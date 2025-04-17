from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
#import ollama
import chromadb
import logging
import time
from _old_load_pdf_2 import LoadAndSplitDocuments
from tenacity import retry, stop_after_attempt, wait_exponential
import os

class EmbeddingsAndVectorStore :
    def __init__(self):
        self.model = 'nomic-embed-text'
        self.load_data = LoadAndSplitDocuments()
        self.document_chunks =  self.load_data.run_load_and_split_documents()
        self.embedding_function = self.get_embedding_function()

    def run_embeddings_and_store_vectors(self):
        return self.generate_vector_db()


    def generate_vector_db(self):
        start_time = time.time()
        logging.info('Generate vector DB')

        # Add metadata (e.g., document names) to documents
        for doc in self.document_chunks:
            doc.metadata["source"] = os.path.basename(doc.metadata.get("source", "Unknown"))

        vector_db = FAISS.from_documents(self.document_chunks, self.embedding_function)

        # vector_db = Chroma.from_documents(
        #     documents=self.document_chunks,
        #     embedding=self.get_embedding_function(),
        #     collection_name='rag',
        #     persist_directory="./chroma_db"
        # )
        # vector_db.persist()
        print(f"Saved {len(self.document_chunks)} chunks to './chroma_db'")
        end_time = time.time()
        logging.info(f'generate_vector_db done in {end_time - start_time}')

        # # using persist storage
        # vector_db = Chroma(
        #     persist_directory="./chroma_db",
        #     embedding_function=self.get_embedding_function(),
        #     collection_name='rag')

        return vector_db

    # @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=1, max=10))
    # def get_embedding_function(self):
    #     start_time = time.time()
    #     logging.info('get_embedding_function')
    #     embeddings = MistralAIEmbeddings(
    #         model="mistral-embed", mistral_api_key='2J2LbKM3hTi9SLjpKwN6kC3bmQF1Vl1M')
    #     end_time = time.time()
    #     logging.info(f'get_embedding_function done in {end_time - start_time}')
    #     return embeddings

    def get_embedding_function(self):
        start_time = time.time()
        logging.info('get_embedding_function')
        embeddings = OllamaEmbeddings(
            model=self.model,
            show_progress=True
        )
        end_time = time.time()
        logging.info(f'get_embedding_function done in {end_time - start_time}')
        return embeddings



    # BedrockEmbeddings
    # def __get_embedding_function():
    #     embeddings = BedrockEmbeddings(
    #         #credentials_profile_name="default",
    #         region_name="us-east-1"
    #     )
    #     return embeddings
    #
    # OpenAIEmbeddings
    # def get_embedding_function_openai():
    #     embeddings_openai = OpenAIEmbeddings(
    #         model="text-embedding-ada-002",
    #         openai_api_key = ''
    #     )
    #     return embeddings_openai

    # ollama
    # def get_embedding_function_ollama(model, chunks):
    #     return [ollama.embeddings(model= model, prompt=chunk)["embedding"] for chunk in chunks]



if __name__ == '__main__':
    emb = EmbeddingsAndVectorStore()
    print(emb.document_chunks)
    # single_vector = emb.get_embedding_function().embed_query(emb.document_chunks[0].page_content)
    # print(single_vector)
    print(emb.run_embeddings_and_store_vectors())

    # load = load_documents()
    # chunks = split_documents(docuemnts=load)
    # single_vector = get_embedding_function().embed_query(chunks[0].page_content)
    # print(single_vector)
    #
    # two_vectors = get_embedding_function().embed_documents([chunk.page_content for chunk in chunks])
    # print(two_vectors)

    # for chunk in chunks :
    #     print(chunk.page_content)
    #     emb = get_embedding_function(model = "mistral", chunks = chunk.page_content)
    #     print(emb)
    #print(emb.embed_query('hello'))
    #print(emb.invoke('hello'))