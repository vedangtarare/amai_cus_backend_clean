import streamlit as st
import os
import gdown
from fpdf import FPDF
from docx import Document
from io import BytesIO
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA

# --- PAGE CONFIG ---
st.set_page_config(page_title="📚 Legal RAG Assistant", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
        }
        .stApp {
            color: #1a1a1a;
            font-family: 'Segoe UI', sans-serif;
        }
        .chunk-box {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }
        .title {
            text-align: center;
            font-size: 2.5rem;
            color: #3b5737;
            margin-bottom: 1rem;
        }
        .search-box input {
            width: 60% !important;
            margin: auto !important;
            border-radius: 8px;
            padding: 0.5rem !important;
            border: 1px solid #ccc;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        .center-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 1rem;
        }
        h3, h4 {
            color: white !important;
            background-color: #3b5737;
            padding: 0.5rem;
            border-radius: 5px;
        }
        .stDownloadButton button, .stForm button {
            color: white !important;
            background-color: #3b5737 !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🤖 amAIcus – Your Legal Assistant</div>', unsafe_allow_html=True)

# --- INPUTS ---
openai_api_key = st.text_input("🔐 Enter your OpenAI API key", type="password")

with st.form(key='legal_query_form'):
    st.markdown("<div class='center-container'><div class='search-box'>", unsafe_allow_html=True)
    query = st.text_input("💬 Ask a legal question")
    st.markdown("</div></div>", unsafe_allow_html=True)
    view_mode = st.radio("🧩 View Format:", ["🔍 Case-by-Case Insight", "🧠 Final Summary Only"])
    submit_button = st.form_submit_button(label='🔎 Submit')

if submit_button and query and openai_api_key:
    with st.spinner("Thinking..."):
        embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
        if not os.path.exists("batch_1_index"):
            st.info("📥 Downloading vectorstore from Google Drive...")
            gdown.download_folder(id="1EWuxugYvijzp3qlc5APz5rxLifdkxuRD", quiet=False, use_cookies=False)

        vectorstore = FAISS.load_local("batch_1_index", embedding_model, allow_dangerous_deserialization=True)
        raw_docs_and_scores = vectorstore.similarity_search_with_score(query, k=15)

        seen = set()
        docs_and_scores = []
        for doc, score in raw_docs_and_scores:
            if doc.page_content.strip() not in seen:
                seen.add(doc.page_content.strip())
                docs_and_scores.append((doc, score))
            if len(docs_and_scores) == 5:
                break

        context_blocks = []
        case_based_answers = []

        final_answer = ""

if view_mode == "🔍 Case-by-Case Insight":
    st.info("💡 Final summary, follow-up questions, and download options are available in 'Final Summary Only' mode.")
    st.markdown("<h3>📂 Relevant Case Matches</h3>", unsafe_allow_html=True)
            st.markdown("<h3>📂 Relevant Case Matches</h3>", unsafe_allow_html=True)
            for i, (doc, score) in enumerate(docs_and_scores):
                case_text = doc.page_content.strip()[:2500]
                context_blocks.append(case_text)
                quoted = f"\"{case_text}\""

                analysis_prompt = f"Analyze the following legal case excerpt in the context of the question: \"{query}\"\n\n{quoted}\n\nMention key legal doctrines and provide a detailed explanation. End with a complete sentence."
                significance_prompt = f"Explain the significance of this case in relation to the query: \"{query}\". Begin and end with a complete sentence.\n\n{quoted}"
                case_name_prompt = f"What is the name of the Indian case this excerpt likely belongs to? Provide only the name and citation if possible.\n\n{quoted}"
                facts_prompt = f"State the facts of this case in 2-3 lines. Begin and end with a complete sentence.\n\n{quoted}"
                holding_prompt = f"State the judgment held in 1-2 lines. What did the court decide? Ensure it starts and ends cleanly.\n\n{quoted}"
                also_lookup_prompt = f"List any legislation, rules, sections, or by-laws that the user should additionally refer to in order to better understand their query: \"{query}\"\n\n{quoted}"

                also_lookup = OpenAI(openai_api_key=openai_api_key)(also_lookup_prompt)
                case_answer = OpenAI(openai_api_key=openai_api_key)(analysis_prompt)
                significance = OpenAI(openai_api_key=openai_api_key)(significance_prompt)
                case_name = OpenAI(openai_api_key=openai_api_key)(case_name_prompt)
                facts = OpenAI(openai_api_key=openai_api_key)(facts_prompt)
                holding = OpenAI(openai_api_key=openai_api_key)(holding_prompt)

                st.markdown(f"""
                    <div class=\"chunk-box\">
                        <strong>Case #{i+1}: {case_name}</strong> <span style='color:#666'>(Match Score: {round((1 - score)*100, 2):.2f}%)</span><br><br>
                        <b>Excerpt:</b> <em>{quoted}</em><br><br>
                        <b>🧾 Facts:</b> {facts}<br><br>
                        <b>⚖️ Judgment Held:</b> {holding}<br><br>
                        <b>📌 Significance:</b> {significance}<br><br>
                        <b>🧠 Legal Analysis:</b> {case_answer}<br><br>
                        <b>📚 Also Lookup:</b> {also_lookup}
                    </div>
                """, unsafe_allow_html=True)

                case_based_answers.append(f"Match #{i+1}: {case_name}\nFacts: {facts}\nHolding: {holding}\nSignificance: {significance}\nAnalysis: {case_answer}\n")

        if view_mode == "🧠 Final Summary Only":
            st.markdown("<h3>🧩 Final Consolidated Answer</h3>", unsafe_allow_html=True)
            combined_context = "\n\n".join([f"\"{chunk}\"" for chunk in context_blocks])
            synthesis_prompt = f"""The user asked: \"{query}\"\n\nUsing the information from the following 3-5 different legal case excerpts, provide a comprehensive, synthesized legal answer.\n\n{combined_context}\n\nMention the most relevant doctrines and principles and explain how each case contributes to the answer. End with a proper concluding sentence."""

            final_answer = OpenAI(openai_api_key=openai_api_key)(synthesis_prompt)
            st.success(final_answer)

        def generate_pdf(text):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for line in text.split('\n'):
                pdf.multi_cell(0, 10, line)
            return pdf.output(dest='S').encode('latin-1')

        def generate_docx(text):
            doc = Document()
            for line in text.split('\n'):
                doc.add_paragraph(line)
            buffer = BytesIO()
            doc.save(buffer)
            buffer.seek(0)
            return buffer

        st.download_button("📥 Download Final Answer (.pdf)", data=generate_pdf(final_answer),
                           file_name="legal_answer.pdf", mime="application/pdf")

        st.download_button("📥 Download Final Answer (.docx)", data=generate_docx(final_answer),
                           file_name="legal_answer.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

        followup = st.text_input("🤔 Ask a follow-up question")
        if followup:
            followup_prompt = f"""The user previously asked: \"{query}\"\n\nThe assistant responded:\n{final_answer}\n\nNow respond to the follow-up: \"{followup}\"""".strip()

