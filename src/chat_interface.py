import io
import os
import re
import inspect
import streamlit as st
import speech_recognition as sr
from dotenv import load_dotenv

from src.api_utils import get_api_response

# --- TTS (Text-to-Speech) ---
from gtts import gTTS
import tempfile

# --- Language detection ---
from langdetect import detect, LangDetectException

# --- LLM (Gemini via LangChain) for transcript cleanup ---
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))  # Lower for more consistent corrections

# ======== No-sidebar defaults (configurable via environment) ========
ASR_LANG = os.getenv("ASR_LANG", "en-US")
TTS_LANG = os.getenv("TTS_LANG", "en")  
TTS_ENABLED = os.getenv("TTS_ENABLED", "1").strip() not in {"0", "false", "False", ""}
LLM_CLEANUP_ENABLED = os.getenv("LLM_CLEANUP_ENABLED", "1").strip() not in {"0", "false", "False", ""}
DEBUG_CLEANUP = os.getenv("DEBUG_CLEANUP", "0").strip() not in {"0", "false", "False", ""}


# ===================== Utilities =====================

def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    Returns ISO 639-1 code (e.g., 'en', 'hu', 'ar', 'fr')
    """
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return "en"  # Default to English if detection fails


def get_language_name(lang_code: str) -> str:
    """
    Get human-readable language name from code.
    """
    lang_map = {
        "en": "English",
        "hu": "Hungarian",
        "ar": "Arabic",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "zh-cn": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "tr": "Turkish",
        "nl": "Dutch",
        "pl": "Polish",
    }
    return lang_map.get(lang_code, lang_code.upper())


def audio_input_compat(label: str, key: str = None):
    """
    Backward-compatible wrapper for st.audio_input:
    Uses sample_rate=16000 only if the current Streamlit supports it.
    """
    sig = inspect.signature(st.audio_input)
    if "sample_rate" in sig.parameters:
        return st.audio_input(label, sample_rate=16000, key=key)
    return st.audio_input(label, key=key)


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


def text_to_speech(text: str, lang: str = None) -> bytes | None:
    """
    Convert text to speech using gTTS (Google Text-to-Speech).
    Auto-detects language if not specified.
    Returns audio bytes or None on error.
    """
    if lang is None:
        lang = detect_language(text)
    
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_path = fp.name
        tts.save(temp_path)
        with open(temp_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        try:
            os.unlink(temp_path)
        except PermissionError:
            import time
            time.sleep(0.1)
            try:
                os.unlink(temp_path)
            except:
                pass    
        return audio_bytes
    except Exception as e:
        st.error(f"TTS error: {e}")
        return None


def cheap_cleanup(text: str) -> str:
    """
    Lightweight, non-LLM cleanup to ensure we never answer and always return something usable.
    - Remove common fillers (multi-language)
    - Normalize whitespace
    - Add trailing '?' for query-like openings if missing
    """
    # Extended filler words for multiple languages
    fillers = r"\b(uh|um|erm|er|mmm|you know|like|uhh|eh|ah|hm|hmm|naaa|szoval|tehat|ugye)\b[,\s]*"
    text = re.sub(fillers, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    if re.match(r"(?i)^(who|what|when|where|why|how|which|do|does|did|is|are|can|could|should|would|may|ki|mi|mikor|hol|miert|hogyan|melyik|van|lesz|lehet)\b", text):
        if not text.endswith((".", "?", "!")):
            text += "?"
    return text


# ===================== Structured output schema =====================

class CleanedTranscript(BaseModel):
    corrected: str = Field(
        ...,
        description=(
            "The corrected transcript in the EXACT SAME language as input. "
            "Fix only ASR errors, punctuation, and remove filler words. "
            "NEVER translate. NEVER add information. NEVER answer questions."
        ),
    )
    detected_language: str = Field(
        ...,
        description="The detected language of the input (e.g., 'English', 'Hungarian', 'Arabic')"
    )


def correct_prompt_with_llm(raw_text: str, model_name: str, debug: bool = False) -> tuple[str, str]:
    """
    Clean ASR text WITHOUT answering or translating.
    Returns: (corrected_text, detected_language)
    
    Enhanced strategy with better prompting for multi-language support.
    """
    if not GEMINI_API_KEY:
        if debug:
            st.info("DEBUG: GEMINI_API_KEY not set — skipping LLM cleanup; using cheap cleanup.")
        detected = detect_language(raw_text)
        return cheap_cleanup(raw_text), get_language_name(detected)

    # Detect language first
    detected_lang_code = detect_language(raw_text)
    detected_lang_name = get_language_name(detected_lang_code)

    # ---- Base LLM with lower temperature for consistency ----
    base_llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=GEMINI_API_KEY,
        temperature=0.0,  # More deterministic
        max_output_tokens=256,
        top_p=0.95,
        top_k=40,
    )

    # ========== Enhanced structured output with language context ==========
    try:
        llm_struct = base_llm.with_structured_output(
            CleanedTranscript,
            strict=True,
            include_raw=True,
        )

        promptA = ChatPromptTemplate.from_messages([
            ("system",
             "You are a professional transcript corrector for speech recognition systems.\n"
             "Your ONLY job is to fix ASR (Automatic Speech Recognition) errors.\n\n"
             "CRITICAL RULES:\n"
             "1. PRESERVE THE ORIGINAL LANGUAGE - If input is Hungarian, output is Hungarian. If English, output is English. NEVER TRANSLATE.\n"
             "2. Fix ONLY these issues:\n"
             "   - Speech recognition errors (wrong words that sound similar)\n"
             "   - Missing or incorrect punctuation\n"
             "   - Capitalization errors\n"
             "   - Remove filler words (uh, um, er, hmm, etc.)\n"
             "   - Remove false starts and repetitions\n"
             "3. PRESERVE:\n"
             "   - All proper nouns, names, and numbers EXACTLY as they appear\n"
             "   - The original meaning and intent\n"
             "   - The language of the input\n"
             "4. NEVER:\n"
             "   - Answer the question\n"
             "   - Add new information\n"
             "   - Translate to another language\n"
             "   - Change the core content\n\n"
             "Return a JSON object with 'corrected' (the cleaned text) and 'detected_language' (the language name)."
            ),
            ("human", 
             "Input: 'uh what countries are in the eea coverage like green card?'\n"
             "Language: English\n"
             "Output: {\"corrected\": \"What countries are in the EEA coverage like green card?\", \"detected_language\": \"English\"}"),
            ("human", 
             "Input: 'Hol ervenyes a zoldkartya uh naaa?'\n"
             "Language: Hungarian\n"
             "Output: {\"corrected\": \"Hol érvényes a zöldkártya?\", \"detected_language\": \"Hungarian\"}"),
            ("human",
             "Input: 'umm ki volt a masodik uh magyar miniszterelnok tehat like who was it?'\n"
             "Language: Hungarian\n"
             "Output: {\"corrected\": \"Ki volt a második magyar miniszterelnök?\", \"detected_language\": \"Hungarian\"}"),
            ("human", 
             "Input: '{utterance}'\n"
             "Detected Language: {language}\n"
             "Remember: Output must be in {language}, NOT translated!")
        ])

        resultA = (promptA | llm_struct).invoke({
            "utterance": raw_text,
            "language": detected_lang_name
        })
        
        parsed = None
        if isinstance(resultA, dict) and "parsed" in resultA:
            parsed = resultA["parsed"]
            if debug:
                st.info("DEBUG: Structured output returned with 'parsed' key.")
        else:
            parsed = resultA

        if debug:
            st.write("DEBUG A result:", resultA)

        corrected = None
        lang = detected_lang_name
        
        if isinstance(parsed, CleanedTranscript):
            corrected = parsed.corrected
            lang = parsed.detected_language
        elif isinstance(parsed, dict):
            corrected = parsed.get("corrected")
            lang = parsed.get("detected_language", detected_lang_name)

        if corrected and corrected.strip():
            return corrected.strip(), lang

    except Exception as e:
        if debug:
            st.warning(f"DEBUG: Structured output failed: {e}")

    # ========== Fallback with explicit format instructions ==========
    try:
        parser = JsonOutputParser(pydantic_object=CleanedTranscript)
        format_instructions = parser.get_format_instructions()

        promptB = ChatPromptTemplate.from_messages([
            ("system",
             f"You are a transcript corrector. CRITICAL: Keep the same language as input.\n"
             f"Fix ASR errors, punctuation, remove fillers. NEVER translate or answer.\n"
             f"Input language detected: {detected_lang_name}. Output MUST be in {detected_lang_name}."),
            ("human",
             "Schema:\n{format_instructions}\n\n"
             "Fix this ASR transcript. Keep it in {language}. Remove fillers, fix errors.\n\n"
             "Input: '{utterance}'\n"
             "Return ONLY valid JSON matching the schema.")
        ])

        chainB = promptB | base_llm | parser
        obj = chainB.invoke({
            "format_instructions": format_instructions,
            "utterance": raw_text,
            "language": detected_lang_name
        })
        
        if debug:
            st.write("DEBUG B parsed obj:", obj)

        corrected = None
        lang = detected_lang_name
        
        if isinstance(obj, CleanedTranscript):
            corrected = obj.corrected
            lang = obj.detected_language
        elif isinstance(obj, dict):
            corrected = obj.get("corrected")
            lang = obj.get("detected_language", detected_lang_name)

        if corrected and corrected.strip():
            return corrected.strip(), lang

    except Exception as e:
        if debug:
            st.warning(f"DEBUG: JSON parser failed: {e}")

    # ========== Final fallback ==========
    cleaned = cheap_cleanup(raw_text)
    if debug:
        st.info("DEBUG: Using cheap_cleanup fallback.")
    return cleaned, detected_lang_name


# ===================== Main UI =====================

def display_chat_interface():
    # ---- Session defaults ----
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "model" not in st.session_state:
        st.session_state.model = DEFAULT_MODEL
    if "audio_enabled" not in st.session_state:
        st.session_state.audio_enabled = TTS_ENABLED
    if "last_played_audio_idx" not in st.session_state:
        st.session_state.last_played_audio_idx = -1
    if "detected_languages" not in st.session_state:
        st.session_state.detected_languages = {}
    
    # Display historical messages (no audio replay)
    for idx, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    
    prompt = None
    raw_transcript = None
    corrected_transcript = None
    detected_language = None
    
    col1, col2 = st.columns([0.1, 0.9])
    
    with col1:
        with st.popover("🎙️", use_container_width=False):
            st.caption("Record your message")
            audio_value = audio_input_compat("Press to record", key="voice_input")
            
            if audio_value is not None:
                st.success("✓ Audio recorded")
                if st.button("🎧 Play recording"):
                    st.audio(audio_value)
    
    with col2:
        audio_toggle = st.checkbox(
            "🔊 Enable audio responses",
            value=st.session_state.audio_enabled,
            key="audio_toggle"
        )
        st.session_state.audio_enabled = audio_toggle
    
    if audio_value is not None:
        with st.spinner("🎙️ Processing audio..."):
            raw_transcript = transcribe_audio_value(audio_value, language=ASR_LANG)

        if raw_transcript and raw_transcript != "(Sorry, I couldn't understand the audio.)":
            if LLM_CLEANUP_ENABLED:
                with st.spinner("✨ Refining transcript..."):
                    corrected_transcript, detected_language = correct_prompt_with_llm(
                        raw_text=raw_transcript,
                        model_name=st.session_state.model,
                        debug=DEBUG_CLEANUP,
                    )
                    if DEBUG_CLEANUP:
                        st.info(f"🗣️ Raw: {raw_transcript}")
                        st.success(f"✅ Corrected ({detected_language}): {corrected_transcript}")
                prompt = corrected_transcript
            else:
                prompt = raw_transcript
                detected_language = get_language_name(detect_language(raw_transcript))
        elif raw_transcript:
            prompt = raw_transcript
    
    text_input = st.chat_input("Type your query here")
    if text_input:
        prompt = text_input
        raw_transcript = None
        corrected_transcript = None
        detected_language = get_language_name(detect_language(text_input))
    
    if prompt:
        # Store detected language for this message
        current_msg_idx = len(st.session_state.messages)
        if detected_language:
            st.session_state.detected_languages[current_msg_idx] = detect_language(prompt)
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Generating response..."):
            WINDOW_MESSAGES = 6  
            history_payload = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-WINDOW_MESSAGES:]
            ]

            resp = get_api_response(
                prompt,
                st.session_state.session_id,
                model=st.session_state.model,
                history=history_payload,   
            )

        if resp:
            st.session_state.session_id = resp.get("session_id")
            answer = resp.get("answer", "")
            
            # Detect answer language for TTS
            answer_lang = detect_language(answer)
            
            # Add new message
            current_assistant_idx = len(st.session_state.messages)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer
            })
            st.session_state.detected_languages[current_assistant_idx] = answer_lang

            with st.chat_message("assistant"):
                st.markdown(answer)
                
                # Play audio only once for the NEW message
                if (st.session_state.audio_enabled and 
                    answer and 
                    current_assistant_idx > st.session_state.last_played_audio_idx):
                    
                    with st.spinner("🔊 Generating audio..."):
                        audio_bytes = text_to_speech(answer, lang=answer_lang)
                        if audio_bytes:
                            st.audio(audio_bytes, format='audio/mp3', autoplay=True)
                            # Mark this message as played
                            st.session_state.last_played_audio_idx = current_assistant_idx
        else:
            st.error("Failed to get a response from the API.")