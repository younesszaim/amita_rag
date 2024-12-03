from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

from embedding import EmbeddingsAndVectorStore


class Retrieval :
    def __init__(self):
        self.load_embedding = EmbeddingsAndVectorStore()
        # self.vector_db = self.load_embedding.run_embeddings_and_store_vectors()
        self.local_model = "mistral"
        self.llm = ChatOllama(model=self.local_model)
        self.QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""
            Vous êtes un assistant IA francophone spécialisé dans la recherche d'informations.
            Votre tâche est de générer plusieurs reformulations différentes de la question de l'utilisateur afin 
            d'améliorer la récupération de documents pertinents (sans besoin de préciser de quel documnet
            s'agit-il) depuis une base de données vectorielle. 
            L'objectif est de diversifier les perspectives sur la même question afin d'améliorer les résultats de recherche.

            Question initiale : {question}
            """
    )

    def run_retrieval(self):
        retriever = MultiQueryRetriever.from_llm(
            self.load_embedding.run_embeddings_and_store_vectors().as_retriever(search_kwargs={"k": 6}),
            self.llm,
            prompt=self.QUERY_PROMPT
        )
        return retriever


if __name__ == '__main__':
    retrieval = Retrieval()
    print(retrieval.run_retrieval())

