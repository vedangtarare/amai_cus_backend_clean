import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
import os
import pickle
import gdown

st.set_page_config(page_title="amAIcus", layout="wide")
st.title("🤖 amAIcus: Your AI Legal Research Assistant")

# Sidebar
st.sidebar.header("Search Settings")
top_k = st.sidebar.slider("Number of Matches", 1, 10, 3)
view_mode = st.sidebar.radio("Choose View Mode", ["🔍 Case-by-Case Insight", "🧠 Final Summary Answer"])

# Google Drive download links
faiss_link = "https://drive.google.com/uc?id=1t3QjvwMGhN9kXyK_cn8wqosp-SHhFBhf"
pkl_link = "https://drive.google.com/uc?id=1WS2jTJEvO0COQnu5Q52K0MAbwvQjMXVx"

folder_name = "batch_1_index"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

faiss_path = f"{folder_name}/index.faiss"
pkl_path = f"{folder_name}/index.pkl"

if not os.path.exists(faiss_path):
    gdown.download(faiss_link, faiss_path, quiet=False)

if not os.path.exists(pkl_path):
    gdown.download(pkl_link, pkl_path, quiet=False)

with open(pkl_path, "rb") as f:
    store = pickle.load(f)

store.index = FAISS.load_local(folder_name, OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])).index

query = st.text_input("🔎 Enter your legal research question here:")

if query:
    docs_and_scores = store.similarity_search_with_score(query, k=top_k)
    context_blocks = []
    final_answer = ""

    if view_mode == "🔍 Case-by-Case Insight":
        st.info("💡 Follow-up questions, download options, and detailed case matches are below.")
        st.markdown("<h3>📂 Relevant Case Matches</h3>", unsafe_allow_html=True)

        for i, (doc, score) in enumerate(docs_and_scores):
            case_text = doc.page_content.strip()[:2500]
            context_blocks.append(case_text)
            quoted = f""{case_text}""

            analysis_prompt = f"""Analyze the following legal case excerpt in the context of the question: \"{query}\""""

{quoted}"
            significance_prompt = f"Explain the significance of this case in relation to the query: "{query}". Begin and end with a complete sentence.

{quoted}"
            case_name_prompt = f"What is the name of the Indian case this excerpt likely belongs to? Provide only the name and citation if possible.

{quoted}"
            facts_prompt = f"State the facts of this case in 2–3 lines. Begin and end with a complete sentence.

{quoted}"
            holding_prompt = f"State the judgment held in 1–2 lines. What did the court decide? Ensure it starts and ends cleanly.

{quoted}"
            also_lookup_prompt = f"List any legislation, rules, sections, or by-laws that the user should additionally refer to in order to better understand their query: "{query}"

{quoted}"

            model = OpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"])
            analysis = model.invoke(analysis_prompt).strip()
            significance = model.invoke(significance_prompt).strip()
            case_name = model.invoke(case_name_prompt).strip()
            facts = model.invoke(facts_prompt).strip()
            holding = model.invoke(holding_prompt).strip()
            also_lookup = model.invoke(also_lookup_prompt).strip()

            st.markdown(f"<h4>📝 Match {i+1} Summary</h4>", unsafe_allow_html=True)
            st.write("📌 Case Name:", case_name)
            st.write("📖 Facts:", facts)
            st.write("⚖️ Judgment:", holding)
            st.write("🔍 Analysis:", analysis)
            st.write("📎 Significance:", significance)
            st.write("📚 Related Laws:", also_lookup)

    elif view_mode == "🧠 Final Summary Answer":
        combined_context = "\n\n".join([f""{chunk}"" for chunk, _ in docs_and_scores])
        synthesis_prompt = f"""The user asked: "{query}"

Using the following excerpts from legal cases, provide a final detailed answer. End with a full stop.

{combined_context}"""
        model = OpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"])
        final_answer = model.invoke(synthesis_prompt).strip()

        # Ensure the answer ends with a full stop
        if not final_answer.endswith("."):
            final_answer += "."

        st.success("🧩 Final Consolidated Answer")
        st.markdown(f"<div style='color:white;'>{final_answer}</div>", unsafe_allow_html=True)
