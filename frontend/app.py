import os, requests, streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Exp Lab Platform", layout="wide")
st.title("Exp Lab Platform — MVP")

st.header("Create project")
code = st.text_input("Project code")
name = st.text_input("Project name")
if st.button("Create"):
    r = requests.post(f"{API}/projects/", json={"code": code, "name": name})
    st.write(r.status_code, r.json() if r.content else {})

st.header("Projects list")
if st.button("Refresh"):
    r = requests.get(f"{API}/projects/")
    st.json(r.json() if r.ok else {})
