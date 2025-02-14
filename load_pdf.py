#from langchain.document_loaders.pdf import  PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from dotenv import load_dotenv
import os
import logging
import time
import io
import fitz 
from office365.sharepoint.client_context import ClientContext
from office365.runtime.auth.user_credential import UserCredential
from office365.sharepoint.files.file import File
import urllib.parse

logging.basicConfig(level=logging.INFO)

# Chargement des variables d'environnement
load_dotenv(dotenv_path='.config')

class LoadAndSplitDocuments:
    def __init__(self):
        self.sharepoint_url = "https://acipartnersparis2.sharepoint.com/sites/POC_RAG"
        self.username = "poc.rag@amitaconseil.com"
        self.password = "Juliepocrag2025"     

        # Connexion à SharePoint
        self.ctx = ClientContext(self.sharepoint_url).with_credentials(UserCredential(self.username, self.password))
        logging.info(f"Connexion à SharePoint avec {self.username}")

        # Vérification de la connexion
        try:
            web = self.ctx.web
            self.ctx.load(web)
            self.ctx.execute_query()
            logging.info(f"Connexion réussie à {web.properties['Title']}")
        except Exception as e:
            logging.error(f"Erreur de connexion : {e}")
            exit()

    def get_pdfs_from_sharepoint(self):
        """Récupère les fichiers PDF directement depuis SharePoint sans les télécharger."""
        try:
            folder = self.ctx.web.get_folder_by_server_relative_url(f"/sites/POC_RAG/Documents%20partages")
            files = folder.files
            self.ctx.load(files)
            self.ctx.execute_query()

            # Filtrer les fichiers PDF
            pdf_files = [file for file in files if file.properties["Name"].endswith(".pdf")]
            logging.info(f"{len(pdf_files)} fichiers PDF trouvés.")
            return pdf_files
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des fichiers PDF : {e}")
            return []

    def load_pdf_from_memory(self, file):
        """Charge un PDF directement en mémoire depuis SharePoint et extrait son contenu avec PyMuPDF (fitz)."""
        try:
            # Obtient l'URL relative du fichier sur SharePoint
            server_relative_url = file.properties["ServerRelativeUrl"]

            # Charge le contenu du fichier PDF depuis SharePoint en binaire
            file_content = File.open_binary(self.ctx, server_relative_url).content

            with io.BytesIO(file_content) as pdf_stream:  # Utilisation du contexte pour s'assurer que le flux est fermé correctement
                doc = fitz.open(stream=pdf_stream)

                # Extraction du texte de chaque page du PDF
                documents = []
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()  # Extraction du texte de la page
                    documents.append(Document(page_content=text, metadata={"page": page_num + 1,"source": file.properties["Name"],"path": server_relative_url}))

                    logging.info(f"Page {page_num + 1} Content: {text[:300]}...")  # Affiche un extrait du contenu

            return documents  # Retourne les documents extraits de chaque page
        except Exception as e:
            logging.error(f"Erreur de lecture du PDF {file.properties['Name']} : {e}")
            return []

    def run_load_and_split_documents(self):
        """Charge et découpe les documents PDF sans les télécharger en local."""
        start_time = time.time()
        logging.info('Démarrage du traitement des documents')

        # Récupérer les fichiers PDF depuis SharePoint
        pdf_files = self.get_pdfs_from_sharepoint()
        all_chunks = []

        # Traiter chaque fichier PDF
        for file in pdf_files:
            logging.info(f"Traitement du fichier : {file.properties['Name']}")
            documents = self.load_pdf_from_memory(file)  # Charger le contenu PDF directement en mémoire
            chunks = self.split_documents(documents)
            all_chunks.extend(chunks)

        end_time = time.time()
        logging.info(f'Traitement terminé en {end_time - start_time} secondes')
        return all_chunks

    @staticmethod
    def split_documents(documents: list[Document]) -> list[Document]:
        """Découpe les documents en chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)


# Exemple d'utilisation
if __name__ == "__main__":
    load_data = LoadAndSplitDocuments()
    
    # Récupère les fichiers PDF depuis SharePoint
    pdf_files = load_data.get_pdfs_from_sharepoint()

    # Charge et extrait le contenu du premier PDF
    if pdf_files:
        first_pdf = pdf_files[0]
        documents = load_data.load_pdf_from_memory(first_pdf)
        logging.info(f"Nombre de pages chargées : {len(documents)}")
    else:
        logging.warning("Aucun fichier PDF trouvé.")


if __name__ == '__main__':
    load_and_split = LoadAndSplitDocuments()
    chunks = load_and_split.run_load_and_split_documents()


