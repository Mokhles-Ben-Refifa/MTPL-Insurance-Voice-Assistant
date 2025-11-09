# chat_interface.py
import io
import os
import re
import inspect
import streamlit as st
import speech_recognition as sr
from dotenv import load_dotenv

from src.api_utils import get_api_response

# --- LLM (Gemini via LangChain) for transcript cleanup ---
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# ======== No-sidebar defaults (configurable via environment) ========
ASR_LANG = os.getenv("ASR_LANG", "en-US")
LLM_CLEANUP_ENABLED = os.getenv("LLM_CLEANUP_ENABLED", "1").strip() not in {"0", "false", "False", ""}
DEBUG_CLEANUP = os.getenv("DEBUG_CLEANUP", "0").strip() not in {"0", "false", "False", ""}


# ===================== Utilities =====================

def audio_input_compat(label: str):
    """
    Backward-compatible wrapper for st.audio_input:
    Uses sample_rate=16000 only if the current Streamlit supports it.
    """
    sig = inspect.signature(st.audio_input)
    if "sample_rate" in sig.parameters:
        return st.audio_input(label, sample_rate=16000)
    return st.audio_input(label)


def transcribe_audio_value(audio_value, language=ASR_LANG) -> str | None:
    """
    Transcribe WAV bytes from st.audio_input using SpeechRecognition's Google API.
    Returns text or None on error.
    """
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_value.getvalue())) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data, language=language)
    except sr.UnknownValueError:
        return "(Sorry, I couldn't understand the audio.)"
    except Exception as e:
        st.error(f"ASR error: {e}")
        return None


def cheap_cleanup(text: str) -> str:
    """
    Lightweight, non-LLM cleanup to ensure we never answer and always return something usable.
    - Remove common fillers
    - Normalize whitespace
    - Add trailing '?' for query-like openings if missing
    """
    fillers = r"\b(uh|um|erm|er|mmm|you know|like|uhh|eh|ah)\b[,\s]*"
    text = re.sub(fillers, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    # If looks like a question, ensure punctuation
    if re.match(r"(?i)^(who|what|when|where|why|how|which|do|does|did|is|are|can|could|should|would|may)\b", text):
        if not text.endswith((".", "?", "!")):
            text += "?"
    return text


# ===================== Structured output schema =====================

class CleanedTranscript(BaseModel):
    corrected: str = Field(
        ...,
        description=(
            "The corrected transcript in the SAME language as input. "
            "No added information, no explanations, no answers."
        ),
    )


def correct_prompt_with_llm(raw_text: str, model_name: str, debug: bool = False) -> str:
    """
    Clean ASR text WITHOUT answering.
    Strategy:
      A) Try strict structured output via with_structured_output(..., strict, include_raw)
      B) Fallback to JsonOutputParser with explicit format_instructions
      C) Fallback to cheap_cleanup
    """
    if not GEMINI_API_KEY:
        if debug:
            st.info("DEBUG: GEMINI_API_KEY not set — skipping LLM cleanup; using cheap cleanup.")
        return cheap_cleanup(raw_text)

    # ---- Base LLM (shared across fallbacks) ----
    base_llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=GEMINI_API_KEY,
        temperature=DEFAULT_TEMPERATURE,
        max_output_tokens=128,
        top_p=0.9,
        top_k=40,
    )

    # ========== A) Structured output (strict) ==========
    try:
        # Some LangChain versions support strict/include_raw; if not, this block will raise
        llm_struct = base_llm.with_structured_output(
            CleanedTranscript,
            strict=True,
            include_raw=True,
        )

        promptA = ChatPromptTemplate.from_messages([
            ("system",
             "You are a strict transcript cleaner for a RAG system.\n"
             "NEVER answer questions. ONLY rewrite/clean the input text.\n"
             "Rules:\n"
             "1) Keep the SAME language as input.\n"
             "2) Fix obvious ASR errors and punctuation; preserve names/numbers.\n"
             "3) Remove filler words and false starts.\n"
             "4) Do NOT add information or explanations.\n"
             "5) If the input is a query, output a well-formed, concise query.\n"
             "Return a JSON object that matches the required schema."
            ),
            # Few-shot examples help steer behavior
            ("human", "Input: 'uh what countries are in the eea coverage like green card?'\n"
                      "Output (corrected, same language): 'Which countries are included in the EEA Green Card coverage?'"),
            ("human", "Input: 'Hol ervenyes a zoldkartya uh naaa?'\n"
                      "Output (corrected, same language): 'Hol érvényes a zöldkártya?'"),
            ("human", "{utterance}")
        ])

        resultA = (promptA | llm_struct).invoke({"utterance": raw_text})

        # When include_raw=True, many models return {'parsed': CleanedTranscript(...), 'raw': ...}
        parsed = None
        if isinstance(resultA, dict) and "parsed" in resultA:
            parsed = resultA["parsed"]
            if debug:
                st.info("DEBUG: Structured output (A) returned a dict with 'parsed' and 'raw'.")
        else:
            parsed = resultA

        if debug:
            st.write("DEBUG A result:", resultA)

        corrected = None
        if isinstance(parsed, CleanedTranscript):
            corrected = parsed.corrected
        elif isinstance(parsed, dict):
            corrected = parsed.get("corrected")

        if corrected and corrected.strip():
            return corrected.strip()

    except Exception as e:
        if debug:
            st.warning(f"DEBUG: Structured output (A) failed: {e}")

    # ========== B) JSON parser with format instructions ==========
    try:
        parser = JsonOutputParser(pydantic_object=CleanedTranscript)
        format_instructions = parser.get_format_instructions()

        promptB = ChatPromptTemplate.from_messages([
            ("system",
             "You ONLY return JSON that matches the provided schema. "
             "NEVER answer questions; ONLY clean the text."),
            ("human",
             "Schema:\n{format_instructions}\n\n"
             "Clean this ASR text (same language, fix typos/punctuation, preserve names/numbers, "
             "remove fillers/false starts). Return ONLY JSON.\n\nASR: {utterance}")
        ])

        chainB = promptB | base_llm | parser
        obj = chainB.invoke({"format_instructions": format_instructions, "utterance": raw_text})
        if debug:
            st.write("DEBUG B parsed obj:", obj)

        corrected = None
        if isinstance(obj, CleanedTranscript):
            corrected = obj.corrected
        elif isinstance(obj, dict):
            corrected = obj.get("corrected")

        if corrected and corrected.strip():
            return corrected.strip()

    except Exception as e:
        if debug:
            st.warning(f"DEBUG: JSON parser (B) failed: {e}")

    # ========== C) Fallback heuristic ==========
    cleaned = cheap_cleanup(raw_text)
    if debug:
        st.info("DEBUG: Falling back to cheap_cleanup() result.")
    return cleaned


# ===================== Main UI =====================

def display_chat_interface():
    # ---- Session defaults ----
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "model" not in st.session_state:
        st.session_state.model = DEFAULT_MODEL

    # ---- Render history ----
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = None
    raw_transcript = None
    corrected_transcript = None

    # ---- Mic capture ----
    audio_value = audio_input_compat("🎙️ Record a voice message")
    if audio_value is not None:
        st.audio(audio_value)
        with st.spinner("Transcribing audio..."):
            raw_transcript = transcribe_audio_value(audio_value, language=ASR_LANG)

        if raw_transcript:
            if LLM_CLEANUP_ENABLED:
                with st.spinner("Cleaning up your transcript..."):
                    corrected_transcript = correct_prompt_with_llm(
                        raw_text=raw_transcript,
                        model_name=st.session_state.model,
                        debug=DEBUG_CLEANUP,
                    )
                prompt = corrected_transcript
            else:
                prompt = raw_transcript

    # ---- Text input ----
    text_input = st.chat_input("Type your query here")
    if text_input:
        prompt = text_input
        raw_transcript = None
        corrected_transcript = None

    # ---- Call backend ----
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Generating response..."):
            resp = get_api_response(prompt, st.session_state.session_id, st.session_state.model)

        if resp:
            st.session_state.session_id = resp.get("session_id")
            answer = resp.get("answer", "")
            st.session_state.messages.append({"role": "assistant", "content": answer})

            with st.chat_message("assistant"):
                st.markdown(answer)

            with st.expander("Details"):
                if raw_transcript is not None:
                    st.subheader("ASR (raw)")
                    st.code(raw_transcript)
                if corrected_transcript is not None:
                    st.subheader("ASR (corrected by LLM)")
                    st.code(corrected_transcript)

                st.subheader("Generated Answer")
                st.code(resp.get("answer", ""))
                st.subheader("Model Used")
                st.code(resp.get("model", st.session_state.model))
                st.subheader("Session ID")
                st.code(resp.get("session_id", ""))
        else:
            st.error("Failed to get a response from the API.")