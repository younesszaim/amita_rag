import os
import getpass
from dotenv import load_dotenv
load_dotenv(dotenv_path='.config')


os.environ["MISTRAL_API_KEY"] = os.getenv('MISTRAL_API_KEY', '')
print("API Key set:", os.environ["MISTRAL_API_KEY"][:5], "...")  # optional debug

# ✅ Only now import the LangChain class
from langchain_mistralai import ChatMistralAI

# Initialize LLM
llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0,
    max_retries=2,
)

# Sample input
messages = [
    ("system", "You are a helpful assistant that translates English to French."),
    ("human", "I love programming."),
]

# Run it
ai_msg = llm.invoke(messages)
print(ai_msg.content)

# import os
# import getpass
# from mistralai import Mistral
# #os.environ["MISTRALAI_API_KEY"]= os.getenv('MISTRALAI_API_KEY','')
# #os.environ["MISTRALAI_API_KEY"] = getpass.getpass("Enter your Mistral API key: ")
# api_key = getpass.getpass("Enter your Mistral API key: ")
# # print(api_key)
# model = "mistral-small-latest"

# client = Mistral(api_key=api_key)

# chat_response = client.chat.complete(
#     model= model,
#     messages = [
#         {
#             "role": "user",
#             "content": "What is the best French cheese?",
#         },
#     ]
# )
# print(chat_response.choices[0].message.content)

# import getpass
# import os

# if "MISTRAL_API_KEY" not in os.environ:
#     os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter your Mistral API key: ")

# from langchain_mistralai import ChatMistralAI

# llm = ChatMistralAI(
#     model="mistral-large-latest",
#     temperature=0,
#     max_retries=2,
#     # other params...
# )

# messages = [
#     (
#         "system",
#         "You are a helpful assistant that translates English to French. Translate the user sentence.",
#     ),
#     ("human", "I love programming."),
# ]
# ai_msg = llm.invoke(messages)
# ai_msg

# print(ai_msg.content)

















# import io
# import logging
# from docling.datamodel.base_models import DocumentStream
# from docling.document_converter import DocumentConverter
# import os
# logging.basicConfig(level=logging.INFO)

# def extract_text_with_docling(pdf_path):
#     """Charge un PDF en mémoire et extrait son texte avec Docling."""
#     try:
#         with open(pdf_path, "rb") as f:
#             pdf_bytes = f.read()
        
#         pdf_stream = io.BytesIO(pdf_bytes)
#         source = DocumentStream(name=pdf_path, stream=pdf_stream)
    
#         converter = DocumentConverter()
#         doc = converter.convert(source)

#         # for page_num in range(len(doc)):
#         #     page = doc.load_page(page_num)
#         #     text = page.get_text()
#         #     logging.info(f"📄 Page {page_num + 1} : {text[:500]}...")  # Aperçu du texte

#     except Exception as e:
#         logging.error(f"❌ Erreur de lecture du PDF : {e}")

# if os.path.exists("2024 12 16_BPCE_Point de vue sur les enjeux IA.pdf"):
#     extract_text_with_docling("2024 12 16_BPCE_Point de vue sur les enjeux IA.pdf")
# else:
#     print(f"❌ Fichier introuvable : {"2024 12 16_BPCE_Point de vue sur les enjeux IA.pdf"}")

