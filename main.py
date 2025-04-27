import logging
import os
import time

import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv(dotenv_path='.config')
logging.basicConfig(level=logging.INFO)

class InteractiveRAG:
    def __init__(self, vector_db_path, perim):
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY', '')
        self.embedding_function = self.get_embedding_function()
        self.vector_db_path = vector_db_path
        self.perim = perim
        self._load_or_create_vector_db()
        self.QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template=""" """)
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        # self.QUERY_PROMPT = PromptTemplate(
        #     input_variables=["question"],
        #     template="""
        #            Vous êtes un assistant intelligent francophone représentant Amita Conseil.
        #            Votre rôle est d’accompagner les collaborateurs dans leur recherche d’informations en fournissant
        #            des réponses pertinentes et précises.
        #            Votre mission consiste à reformuler une seul fois la question posée par l’utilisateur afin d’optimiser
        #            la récupération de documents pertinents à partir d’une base de données vectorielle,
        #            tout en préservant l’intention initiale de la demande.
        #
        #            Question initiale : {question}
        #            """
        # ) Si le context ne permet pas de repondre, exprimez le et proposer une reponse alternative qui permettera
        #                 de repondre a la question posée
        self.template = """Répondez à la question en se basant uniquement sur le contexte suivant : {context} 
                Question : {question}
                Détaillez la reponse avec precisions et illuster la reponse avec des bullets points si besoin.
                """
        self.retriever = MultiQueryRetriever.from_llm(
            self.db.as_retriever(search_kwargs={"k": 20}),
            self.llm,
            prompt=self.QUERY_PROMPT
        )  #


    def _load_or_create_vector_db(self):
        #vector_db_path = "./faiss_index"
        # vector_db_path = "./expertise_db"
        # vector_db_path = "./business_db"

        if os.path.exists(self.vector_db_path):
            # Load existing vector store
            logging.info(f"Loading VectorDB for {self.perim}")
            self.db = FAISS.load_local(self.vector_db_path, self.embedding_function, allow_dangerous_deserialization=True)
        else:
            # Create and save vector store
            from load_pdf import LoadAndSplitDocuments
            logging.info(f"Preparing VectorDB for {self.perim}")
            load_data = LoadAndSplitDocuments(perim=self.perim)
            document_chunks = load_data.run_load_and_split_documents()

            # Add sourcing metadata
            for doc in document_chunks:
                doc.metadata["source"] = os.path.basename(doc.metadata.get("source", "Unknown"))

            # Create vector store
            self.db = FAISS.from_documents(document_chunks,
                                           self.embedding_function)

            # Persist vector store
            self.db.save_local(self.vector_db_path)
            logging.info(f"Done saving VectorDB for {self.perim}")

    def get_embedding_function(self):
        start_time = time.time()
        logging.info('get_embedding_function')
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        end_time = time.time()
        logging.info(f'get_embedding_function done in {end_time - start_time}')
        return embeddings

    def run_rag_prompt(self, question: str):
        start_time = time.time()
        logging.info('Run run_rag_prompt')
        logging.info('1. prompt')
        prompt = ChatPromptTemplate.from_template(self.template)
        retrieved_docs = self.retriever.get_relevant_documents(question)
        sources = [doc.metadata.get("source", "Unknown") for doc in retrieved_docs]
        sources = set(sources)
        sources = list(sources)[:2]

        end_time = time.time()
        logging.info(f'1. prompt done {end_time - start_time}')
        logging.info('2. chain')
        chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )
        end_time = time.time()
        logging.info(f'2. chain done {end_time - start_time}')
        logging.info('3. result')
        result = chain.invoke(question)
        end_time = time.time()
        logging.info(f'3. result done {end_time - start_time}')

        end_time = time.time()
        logging.info(f'run_rag_prompt done {end_time - start_time}')
        return {"response": result, "resources": sources}


class ExpertiseRAG(InteractiveRAG):
    def __init__(self):
        super().__init__(vector_db_path = "./expertise_db", perim='expertise')
        self.template = """Répondez à la question en se basant uniquement sur le contexte suivant : {context} 
                       Question : {question}
                       Ta mission est d’aider tes collègues à répondre aux questions portant sur l’expertise d’Amita, à travers ses offres de conseil, ainsi que sur l’expérience des consultants via leurs CV.
                       Détaillez la reponse avec precisions et illuster la reponse avec des bullets points si besoin.
                       """
        self.QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""
            Tu es un assistant intelligent francophone dédié aux équipes d’Amita Conseil.

            Ta mission est d’accompagner tes collègues dans l’analyse et la recherche d’informations concernant les CV des candidats et les offres d’emploi de l’entreprise.
            
            Lorsqu’un utilisateur pose une question, tu dois la reformuler une seule fois pour maximiser la pertinence des résultats obtenus depuis notre base de données vectorielle,
            tout en restant fidèle à l’intention initiale.
            
            Ta reformulation doit :
            
            Être claire et précise.
            Respecter l’esprit de la question posée.
            Orienter explicitement la recherche vers les profils candidats ou les postes proposés.
            
            Question initiale : {question}
            """
        )

class BusinessRAG(InteractiveRAG):
    def __init__(self):
        super().__init__(vector_db_path = "./business_db", perim='business')
        self.template = """Répondez à la question en se basant uniquement sur le contexte suivant : {context} 
                       Question : {question}
                       Ta mission est d’aider tes collègues à obtenir des réponses détaillées concernant les missions réalisées par Amita Conseil, les expertises mobilisées, ainsi que les clients accompagnés.                      
                       Détaillez la reponse avec precisions et illuster la reponse avec des bullets points si besoin.
                       """
        self.QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""
            Vous êtes un assistant intelligent francophone représentant le cabinet Amita Conseil, spécialisé dans l'accompagnement stratégique et opérationnel des entreprises.  
            Votre mission est de soutenir efficacement les collaborateurs dans leur recherche d'information, en optimisant l'accès aux connaissances internes à forte valeur ajoutée.

            Pour ce faire, vous devez reformuler une seule fois la question posée par l’utilisateur, dans le but d’optimiser la pertinence de la recherche dans une base de données vectorielle, 
            tout en respectant fidèlement l’intention initiale exprimée.

            Veuillez effectuer cette reformulation avec rigueur et clarté, en conservant le sens et le besoin sous-jacent de la demande d’origine.

            Question initiale : {question}
            """
        )

class EssentielsRAG(InteractiveRAG):
    def __init__(self):
        super().__init__(vector_db_path = "./essentiels_db", perim='essentiels')
        self.template = """Répondez à la question en se basant uniquement sur le contexte suivant : {context} 
                Question : {question}
                Ta mission est d’aider tes collègues à obtenir des réponses précises sur l’organisation interne d’Amita, incluant la présentation institutionnelle, les référentiels de compétences, et les structures internes.
                Détaillez la reponse avec precisions et illuster la reponse avec des bullets points si besoin.
                               """
        self.QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""
            Vous êtes un assistant intelligent francophone représentant le cabinet Amita Conseil, spécialisé dans l'accompagnement stratégique et opérationnel des entreprises.  
            Votre mission est de soutenir efficacement les collaborateurs dans leur recherche d'information, en optimisant l'accès aux connaissances internes à forte valeur ajoutée.

            Pour ce faire, vous devez reformuler plusieurs fois la question posée par l’utilisateur, dans le but d’optimiser la pertinence de la recherche dans une base de données vectorielle, 
            tout en respectant dans la mesure du possible l’intention initiale exprimée.
            
            Detailler bien la reponse avec des bullets points meme en cas de documents confidentiels.

            Veuillez effectuer cette reformulation avec rigueur et clarté, en conservant le sens et le besoin sous-jacent de la demande d’origine.

            Question initiale : {question}
            """
        )

class InterneRAG(InteractiveRAG):
    def __init__(self):
        super().__init__(vector_db_path = "./interne_db", perim='interne')
        self.template = """Répondez à la question en se basant uniquement sur le contexte suivant : {context} 
                        Question : {question}
                    Ta mission est d’aider tes collègues à obtenir des réponses précises sur les activités internes d’Amita, notamment les groupes de travail (GT), les initiatives internes, et les projets collaboratifs.
                        Détaillez la reponse avec precisions et illuster la reponse avec des bullets points si besoin.
                                       """
        self.QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""
            Vous êtes un assistant intelligent francophone représentant le cabinet Amita Conseil, spécialisé dans l'accompagnement stratégique et opérationnel des entreprises.  
            Votre mission est de soutenir efficacement les collaborateurs dans leur recherche d'information, en optimisant l'accès aux connaissances internes à forte valeur ajoutée.

            Pour ce faire, vous devez reformuler une seule fois la question posée par l’utilisateur, dans le but d’optimiser la pertinence de la recherche dans une base de données vectorielle, 
            tout en respectant fidèlement l’intention initiale exprimée.

            Veuillez effectuer cette reformulation avec rigueur et clarté, en conservant le sens et le besoin sous-jacent de la demande d’origine.

            Question initiale : {question}
            """
        )
# expertise_rag = ExpertiseRAG()
# business_rag = BusinessRAG()
# essentiels_rag = EssentielsRAG()
# interne_rag = InterneRAG()
#  # Display the logo at the top
# st.image("./image/img.png", width=200)
# # Initialize session state for chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
#
# # Input field for user question
# question = st.text_input("Wellcome to AmitaGPT! Comment puis-je t'aider ? 😊", "")
# exp_button = st.button("Expertise")
# business_button = st.button("Business")
#
# # When the "Réponse" button is clicked
# if st.button("Réponse"):
#     pass
#     if question.strip():
#         # Add user's question to chat history
#         st.session_state.chat_history.append({"role": "user", "message": question})
#
#         # Display a progress bar
#         with st.spinner('Génération de la réponse...'):
#             progress_bar = st.progress(0)
#
#             # Simulate response generation process
#             for i in range(10):
#                 time.sleep(0.1)  # Simulate time taken to generate response
#                 progress_bar.progress((i + 1) * 10)
#
#             # Generate answer using the RAG system
#             result = business_rag.run_rag_prompt(question=question)
#             answer = result["response"]
#             resources = result["resources"]
#
#             # Add assistant's answer to chat history
#             st.session_state.chat_history.append(
#                 {"role": "assistant", "message": answer, "resources": resources})
#
# # Display chat history in reverse order (latest first)
# st.write("### Conversation :")
# pass
# for chat in reversed(st.session_state.chat_history):
#     if chat["role"] == "user":
#         st.markdown(f"**Vous** : {chat['message']}")
#     else:
#         st.markdown(f"**AmitaGPT** : {chat['message']}")
#         if "resources" in chat:
#             st.markdown("**Ressources** :")
#             for resource in chat['resources']:
#                 st.markdown(f"- {resource}")

if __name__ == "__main__":
    expertise = ExpertiseRAG()
    business = BusinessRAG()
    essentiels_rag = EssentielsRAG()
    interne_rag = InterneRAG()
