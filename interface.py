import streamlit as st
import requests
import time
from datetime import datetime
import pyvis.network as net
import streamlit.components.v1 as components
import tempfile
import os

BASE_URL = "http://localhost:5000"

st.set_page_config(page_title="Eratosthenes Research Assistant", layout="wide")
st.title("📚 Eratosthenes Research Assistant")

# Sidebar: Settings and History
st.sidebar.header("⚙️ Settings")
retrieve_k = st.sidebar.slider("Retrieve K", min_value=5, max_value=30, value=10)
suggest_k = st.sidebar.slider("Suggest K", min_value=1, max_value=20, value=5)
citation_depth = st.sidebar.slider("Citation Depth", min_value=0, max_value=3, value=1)
should_optimize_query = st.sidebar.checkbox("Optimize Query", value=False)
export_graph = st.sidebar.selectbox("Export Graph As", ["None", "HTML", "PNG"])

st.sidebar.markdown("---")
st.sidebar.header("📜 Job History")

try:
    history_resp = requests.get(f"{BASE_URL}/jobs")
    if history_resp.status_code == 200:
        jobs = history_resp.json().get("jobs", [])[:10]
        for job in jobs:
            st.sidebar.markdown(f"**{job['research_query']}**\n{job['status']} — {job['created_at'][:19].replace('T', ' ')}")
except:
    st.sidebar.info("No job history available.")

# --- NLP Paper Search for Seed Selection ---
st.subheader("🔍 Search NLP Papers to Use as Seeds")
search_query = st.text_input("Search NLP paper titles or abstracts")
selected_papers = []

if search_query:
    try:
        r = requests.get(f"{BASE_URL}/search_nlp_papers", params={"q": search_query})
        results = r.json()["results"]
        selected = st.multiselect(
            "Select seed papers:",
            options=[f"{p['title']} ({p['year']})" for p in results],
            format_func=lambda x: x
        )
        selected_papers = [p for p in results if f"{p['title']} ({p['year']})" in selected]
    except:
        st.error("Search failed. Check server log.")

# --- Citation Graph Display for Selected Seed Papers ---
if selected_papers:
    st.subheader("📊 Combined Citation Graph for Selected Seed Papers")
    try:
        G = net.Network(height="600px", width="100%", directed=True)
        seen_nodes = set()

        for paper in selected_papers:
            paper_id = paper["id"].split("/")[-1]
            graph_data = requests.get(f"{BASE_URL}/citation_graph/{paper_id}").json()
            for node in graph_data["nodes"]:
                if node["id"] not in seen_nodes:
                    G.add_node(
                        node["id"],
                        label=node["label"],
                        title=node.get("title", "No Title Provided"),
                        group="seed" if node["id"] == paper["id"] else "neighbor"
                    )
                    seen_nodes.add(node["id"])
            for edge in graph_data["edges"]:
                G.add_edge(edge["from"], edge["to"])

        G.set_options("{}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            G.save_graph(tmp_file.name)
            with open(tmp_file.name, "r", encoding="utf-8") as f:
                graph_html = f.read()
            components.html(graph_html, height=650, scrolling=True)

        if export_graph == "HTML":
            st.download_button(
                "💾 Download Graph as HTML",
                data=graph_html,
                file_name="citation_graph.html",
                mime="text/html"
            )

        elif export_graph == "PNG":
            try:
                from selenium import webdriver
                from PIL import Image
                import base64
                from io import BytesIO

                driver = webdriver.Firefox()
                driver.get(f"file://{tmp_file.name}")
                time.sleep(3)
                png_path = tmp_file.name.replace(".html", ".png")
                driver.save_screenshot(png_path)
                driver.quit()

                with open(png_path, "rb") as f:
                    btn = st.download_button(
                        label="💾 Download Graph as PNG",
                        data=f,
                        file_name="citation_graph.png",
                        mime="image/png"
                    )
            except Exception as e:
                st.error(f"PNG export failed: {e}")

    except Exception as e:
        st.error(f"Failed to load graph: {e}")

# --- Research Query and Submission ---
st.subheader("📥 Submit a Research Query")
research_query = st.text_input("Enter your research question")

if st.button("Submit Query"):
    if not research_query or not selected_papers:
        st.error("Please provide both a research question and select at least one seed paper.")
        st.stop()

    seed_ids = [p["id"] for p in selected_papers]

    try:
        resp = requests.post(
            f"{BASE_URL}/submit",
            json={
                "research_query": research_query,
                "seed_ids": seed_ids,
                "retrieve_k": retrieve_k,
                "suggest_k": suggest_k,
                "citation_depth": citation_depth,
                "should_optimize_query": should_optimize_query
            }
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error during submission: {e}")
        st.stop()

    job_id = resp.json()["job_id"]
    st.success(f"✅ Query submitted! Job ID: {job_id}")
    st.caption("Tracking job status...")

    status_box = st.empty()
    log_output = st.empty()
    result_area = st.empty()
    status = "pending"
    last_log = ""

    with st.spinner("Processing query..."):
        while status not in ["complete", "failed"]:
            time.sleep(2)
            try:
                status_resp = requests.get(f"{BASE_URL}/status/{job_id}")
                status_resp.raise_for_status()
                status = status_resp.json()["status"]
                status_box.info(f"Current status: `{status}`")
            except requests.exceptions.RequestException as e:
                status_box.error(f"❌ Error checking status: {e}")
                break
            try:
                log_resp = requests.get(f"{BASE_URL}/log/{job_id}")
                log_text = log_resp.json().get("log", "")
                if log_text != last_log:
                    log_output.code(log_text[-2000:] if log_text else "[No output yet]", language="bash")
                    last_log = log_text
            except requests.exceptions.RequestException as e:
                log_output.error(f"❌ Error fetching logs: {e}")
                break

    if status == "complete":
        try:
            result_resp = requests.get(f"{BASE_URL}/result/{job_id}")
            result_resp.raise_for_status()
            bib = result_resp.json().get("bibliography", [])
        except requests.exceptions.RequestException as e:
            result_area.error(f"❌ Could not fetch results: {e}")
            st.stop()

        st.success("📖 Annotated bibliography ready!")

        # for i, entry in enumerate(bib, 1):
        #     st.markdown(f"### [{i}] {entry['title']} ({entry['year']})")
        #     st.markdown(f"**Authors:** {entry['authors']}")
            
        #     st.markdown("---")

        st.markdown(f"**Summary:** {bib}")

    elif status == "failed":
        st.error("⚠️ The job failed. Please check the logs above or try again with different inputs.")
