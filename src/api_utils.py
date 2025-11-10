from dotenv import load_dotenv
import os
import requests
import streamlit as st
from requests.adapters import HTTPAdapter, Retry

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")

def get_api_response(
    question,
    session_id,
    model="gemini-2.5-flash",
    history=None,           
    timeout=20,            
):
    data = {"question": question, "model": model}
    if session_id:
        data["session_id"] = session_id
    if history:
        data["history"] = history  

    try:
       
        s = requests.Session()
        retries = Retry(
            total=2, backoff_factor=0.5,
            status_forcelist=(429, 502, 503, 504),
            allowed_methods=["POST"]
        )
        s.mount("http://", HTTPAdapter(max_retries=retries))
        s.mount("https://", HTTPAdapter(max_retries=retries))

        r = s.post(f"{API_URL}/chat", json=data, timeout=timeout)
        r.raise_for_status()
        return r.json()

    except requests.Timeout:
        st.error("API request timed out. Try again or check the server.")
    except requests.HTTPError as e:
        
        try:
            err_text = e.response.text
        except Exception:
            err_text = str(e)
        st.error(f"API returned an error: {err_text}")
    except requests.RequestException as e:
        st.error(f"Network error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

    return None

def upload_document(file):
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(f"{API_URL}/upload-doc", files=files)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to upload file. Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred while uploading the file: {str(e)}")
        return None

def list_documents():
    try:
        response = requests.get(f"{API_URL}/list-docs")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch document list. Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"An error occurred while fetching the document list: {str(e)}")
        return []

def delete_document(file_id):
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    data = {"file_id": file_id}

    try:
        response = requests.post(f"{API_URL}/delete-doc", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to delete document. Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred while deleting the document: {str(e)}")
        return None