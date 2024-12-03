#from langchain.document_loaders.pdf import  PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from dotenv import load_dotenv, find_dotenv
import os

# load environment variables from .conf file
load_dotenv(dotenv_path='.config')

class LoadAndSplitDocuments:
    def __init__(self):
        self.data_path = os.getenv('DAT A_PATH', './data')

    def run_load_and_split_documents(self):
        document_loader = self.load_documents(data_path = self.data_path)
        document_chunks = self.split_documents(documents = document_loader)
        return document_chunks

    @staticmethod
    def load_documents(data_path : str) -> object:
        document_loader = PyPDFDirectoryLoader(data_path)
        return document_loader.load()

    @staticmethod
    def split_documents(documents : object) -> list[Document]:
        text_spliter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        return text_spliter.split_documents(documents)

if __name__ == '__main__':
    # load = load_documents() # list d'object Document(metadata : dict(source,page), page_content : str)
    # print(load[0])
    load_dotenv(dotenv_path='.config')
    load = LoadAndSplitDocuments()
    print(load.data_path)
    # chunks = load.run_load_and_split_documents()
    # print(chunks)
    # for chunk in chunks :
    #     print(chunk)
    #     break
