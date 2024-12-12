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


class InteractiveRAG:
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY', '')
        self.embedding_function = self.get_embedding_function()
        self._load_or_create_vector_db()
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7
        )
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
        self.template = """Répondez à la question en utilisant UNIQUEMENT le contexte suivant : {context}
                Question : {question}
                """
        self.retriever = MultiQueryRetriever.from_llm(
            self.db.as_retriever(search_kwargs={"k": 10}),
            self.llm,
            prompt=self.QUERY_PROMPT
        )  #

    def _load_or_create_vector_db(self):
        vector_db_path = "./faiss_index"

        if os.path.exists(vector_db_path):
            # Load existing vector store
            self.db = FAISS.load_local(vector_db_path, self.embedding_function, allow_dangerous_deserialization=True)
        else:
            # Create and save vector store
            from load_pdf import LoadAndSplitDocuments

            load_data = LoadAndSplitDocuments()
            document_chunks = load_data.run_load_and_split_documents()

            # Add sourcing metadata
            for doc in document_chunks:
                doc.metadata["source"] = os.path.basename(doc.metadata.get("source", "Unknown"))

            # Create vector store
            self.db = FAISS.from_documents(document_chunks,
                                           self.embedding_function)

            # Persist vector store
            self.db.save_local(vector_db_path)

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

    def run_dag(self):
        while True:
            question = input("Wellcome to AmitaGPT, comment puis-je vous aider ? 😊"
                             "(ou tapez 'exit' pour quitter ) : ")

            if question.lower() == "exit":
                print("Au revoir 👋!")
                break

            response = self.run_rag_prompt(question)
            print(f"Réponse : {response}")

    def main(self):
        # Display the logo at the top
        st.image("./image/img.png", width=200)
        # Initialize session state for chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Input field for user question
        question = st.text_input("Wellcome to AmitaGPT! Comment puis-je t'aider ? 😊", "")

        # When the "Réponse" button is clicked
        if st.button("Réponse"):
            pass
            if question.strip():
                # Add user's question to chat history
                st.session_state.chat_history.append({"role": "user", "message": question})

                # Display a progress bar
                with st.spinner('Génération de la réponse...'):
                    progress_bar = st.progress(0)

                    # Simulate response generation process
                    for i in range(10):
                        time.sleep(0.1)  # Simulate time taken to generate response
                        progress_bar.progress((i + 1) * 10)

                    # Generate answer using the RAG system
                    result = self.run_rag_prompt(question=question)
                    answer = result["response"]
                    resources = result["resources"]

                    # Add assistant's answer to chat history
                    st.session_state.chat_history.append(
                        {"role": "assistant", "message": answer, "resources": resources})

        # Display chat history in reverse order (latest first)
        st.write("### Conversation :")
        pass
        for chat in reversed(st.session_state.chat_history):
            if chat["role"] == "user":
                st.markdown(f"**Vous** : {chat['message']}")
            else:
                st.markdown(f"**AmitaGPT** : {chat['message']}")
                if "resources" in chat:
                    st.markdown("**Ressources** :")
                    for resource in chat['resources']:
                        st.markdown(f"- {resource}")


if __name__ == "__main__":
    rag = InteractiveRAG()
    rag.main()
