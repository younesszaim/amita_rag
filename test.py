import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# Fichier uploadé par l'utilisateur
uploaded_files = st.file_uploader(
    "📄 Upload un document", 
    type=["pdf", "pptx", "docx", "xls"],
    accept_multiple_files=True,
    key="rag_docs",
)

# Suppose que tu veux intégrer ce doc dans ta chaîne LLM
extra_docs = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".pdf"):
            # Sauvegarde temporaire en local pour lecture
            with open(f"/tmp/{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Lecture avec PyPDFLoader
            loader = PyPDFLoader(f"/tmp/{uploaded_file.name}")
            pdf_docs = loader.load()

            # Ajout à la liste des docs à injecter
            extra_docs.extend(pdf_docs)

        # ➕ ici tu pourrais aussi gérer .docx, .pptx etc. avec d'autres loaders

