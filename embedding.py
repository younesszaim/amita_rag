from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
import ollama

from load_pdf import LoadAndSplitDocuments


class EmbeddingsAndVectorStore :
    def __init__(self):
        self.model = 'nomic-embed-text'
        self.load_data = LoadAndSplitDocuments()
        self.document_chunks =  self.load_data.run_load_and_split_documents()

    def run_embeddings_and_store_vectors(self):
        return self.generate_vector_db()


    def generate_vector_db(self):
        vector_db = Chroma.from_documents(
            documents=self.document_chunks,
            embedding=self.get_embedding_function(),
            collection_name='rag'
        )
        return vector_db


    def get_embedding_function(self):
        embeddings = OllamaEmbeddings(
            model=self.model,
            show_progress=True
        )
        return embeddings


# def __get_embedding_function():
#     embeddings = BedrockEmbeddings(
#         #credentials_profile_name="default",
#         region_name="us-east-1"
#     )
#     return embeddings
#
# def get_embedding_function_openai():
#     embeddings_openai = OpenAIEmbeddings(
#         model="text-embedding-ada-002",
#         openai_api_key = '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'
#     )
#     return embeddings_openai

# def get_embedding_function_ollama(model, chunks):
#     return [ollama.embeddings(model= model, prompt=chunk)["embedding"] for chunk in chunks]

# def get_embedding_function():
#     embeddings = OllamaEmbeddings(
#         model = 'nomic-embed-text',
#         show_progress= True
#     )
#     return embeddings

if __name__ == '__main__':
    emb = EmbeddingsAndVectorStore()
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