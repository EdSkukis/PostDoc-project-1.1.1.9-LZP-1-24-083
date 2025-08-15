import os, requests, streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Exp Lab Platform", layout="wide")
st.title("Exp Lab Platform — MVP")

tab_proj, tab_mat, tab_samp, tab_exp = st.tabs(["Projects", "Materials", "Samples", "Experiments"])

with tab_proj:
    st.subheader("Create project")
    code = st.text_input("Project code", key="p_code")
    name = st.text_input("Project name", key="p_name")
    if st.button("Create project"):
        r = requests.post(f"{API}/projects/", json={"code": code, "name": name})
        st.write(r.status_code, r.json() if r.content else {})
    if st.button("Refresh projects"):
        st.json(requests.get(f"{API}/projects/").json())

with tab_mat:
    st.subheader("Create material")
    mcode = st.text_input("Material code", key="m_code")
    mtype = st.text_input("Material type (e.g., R-glass, Epoxy, GFRP)", key="m_type")
    if st.button("Create material"):
        r = requests.post(f"{API}/materials/", json={"material_code": mcode or None, "type": mtype})
        st.write(r.status_code, r.json() if r.content else {})
    if st.button("Refresh materials"):
        st.json(requests.get(f"{API}/materials/").json())

with tab_samp:
    st.subheader("Create sample")
    proj_id = st.number_input("project_id", min_value=1, step=1)
    mat_ref_type = st.selectbox("material_ref_type", ["material","material_component"])
    mat_ref_id = st.number_input("material_ref_id", min_value=1, step=1)
    samp_code = st.text_input("sample_code", key="s_code")
    if st.button("Create sample"):
        r = requests.post(f"{API}/samples/", json={
            "project_id": int(proj_id),
            "material_ref_id": int(mat_ref_id),
            "material_ref_type": mat_ref_type,
            "sample_code": samp_code or None
        })
        st.write(r.status_code, r.json() if r.content else {})
    if st.button("Refresh samples"):
        st.json(requests.get(f"{API}/samples/").json())

with tab_exp:
    st.subheader("Create experiment")
    sample_id = st.number_input("sample_id", min_value=1, step=1)
    test_type_id = st.number_input("test_type_id", min_value=1, step=1)
    if st.button("Create experiment"):
        r = requests.post(f"{API}/experiments/", json={
            "sample_id": int(sample_id),
            "test_type_id": int(test_type_id)
        })
        st.write(r.status_code, r.json() if r.content else {})
    if st.button("Refresh experiments"):
        st.json(requests.get(f"{API}/experiments/").json())
