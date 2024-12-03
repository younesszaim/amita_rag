from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import HumanMessage
from retreival_ import Retrieval



class InteractiveRAG:
    def __init__(self):
        # Étape 1 : Charger et traiter les documents une seule fois
        self.load_data = LoadAndSplitDocuments()
        self.document_chunks = self.load_data.run_load_and_split_documents()

        # Étape 2 : Construire la base vectorielle une seule fois
        self.embeddings_store = EmbeddingsAndVectorStore()
        self.vector_db = self.embeddings_store.generate_vector_db()

        # Étape 3 : Initialiser le modèle et le prompt
        self.local_model = "mistral"  # Assurez-vous qu'il supporte le français
        self.llm = ChatOllama(model=self.local_model)

        # Créer le prompt de génération
        self.rag_prompt = RagPrompt()

        # Initialiser le Query Prompt
        self.QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""Vous êtes un assistant IA conçu pour générer cinq versions différentes de la question utilisateur afin de récupérer des documents pertinents depuis une base de données vectorielle. 
            Votre objectif est d'aider l'utilisateur à surmonter les limites de la recherche par similarité basée sur la distance en fournissant plusieurs perspectives de la question.
            Question initiale : {question}
            Fournissez cinq reformulations de cette question, séparées par des sauts de ligne.
            """
        )

        # Configurer le MultiQueryRetriever
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=self.vector_db.as_retriever(),
            llm=self.llm,
            prompt=self.QUERY_PROMPT,
        )

    def ask_question(self, question: str):
        # Étape 4 : Récupérer les documents pertinents pour la question
        relevant_context = self.retriever.get_relevant_documents(question)

        # Étape 5 : Générer une réponse avec le modèle LLM
        response = self.rag_prompt.run_rag_prompt(retriever=self.retriever, question=question)

        return response
