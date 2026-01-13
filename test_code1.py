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
