import io
import logging
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter
import os
logging.basicConfig(level=logging.INFO)

def extract_text_with_docling(pdf_path):
    """Charge un PDF en mémoire et extrait son texte avec Docling."""
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        
        pdf_stream = io.BytesIO(pdf_bytes)
        source = DocumentStream(name=pdf_path, stream=pdf_stream)
    
        converter = DocumentConverter()
        doc = converter.convert(source)

        # for page_num in range(len(doc)):
        #     page = doc.load_page(page_num)
        #     text = page.get_text()
        #     logging.info(f"📄 Page {page_num + 1} : {text[:500]}...")  # Aperçu du texte

    except Exception as e:
        logging.error(f"❌ Erreur de lecture du PDF : {e}")

if os.path.exists("2024 12 16_BPCE_Point de vue sur les enjeux IA.pdf"):
    extract_text_with_docling("2024 12 16_BPCE_Point de vue sur les enjeux IA.pdf")
else:
    print(f"❌ Fichier introuvable : {"2024 12 16_BPCE_Point de vue sur les enjeux IA.pdf"}")

