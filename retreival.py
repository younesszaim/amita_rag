from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from embedding import EmbeddingsAndVectorStore
import logging
import time


class Retrieval:
    def __init__(self):
        self.load_embedding = EmbeddingsAndVectorStore()
        # self.vector_db = self.load_embedding.run_embeddings_and_store_vectors()
        self.local_model = "mistral"
        # self.llm = ChatMistralAI(mistral_api_key='QkVPATWInsy7Tj1viW2zgBUVBLdvnRvf', model='mistral-large-latest')
        self.llm = ChatOllama(model=self.local_model, keep_alive='1h')
        self.QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""
            Vous êtes un assistant intelligent francophone représentant Amita Conseil. 
            Votre rôle est d’accompagner les collaborateurs dans leur recherche d’informations en fournissant 
            des réponses pertinentes et précises.
            Votre mission consiste à reformuler une seul fois la question posée par l’utilisateur afin d’optimiser 
            la récupération de documents pertinents à partir d’une base de données vectorielle, 
            tout en préservant l’intention initiale de la demande.
            
            Question initiale : {question}
            """
        )

    def run_retrieval(self):
        start_time = time.time()
        logging.info('Run run_retrieval')
        retriever = MultiQueryRetriever.from_llm(
            self.load_embedding.run_embeddings_and_store_vectors().as_retriever(search_kwargs={"k": 6, 'lambda_mult': 0.25}),
            self.llm,
            prompt=self.QUERY_PROMPT
        )
        # results = retriever.get_relevant_documents(self.QUERY_PROMPT)
        # for doc in results:
        #     logging.info(f"Retrieved document: {doc.metadata.get('source', 'Unknown')}")

        end_time = time.time()
        logging.info(f'run_retrieval done {end_time - start_time}')
        return retriever


if __name__ == '__main__':
    retrieval = Retrieval()
    print(retrieval.run_retrieval())
