from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import HumanMessage

from rag_prompt import RagPrompt
from retreival import Retrieval
import streamlit as st

class InteractiveRAG:
    def __init__(self):
        self.rag = RagPrompt(retriever_instance= Retrieval())

    def run_dag(self):
        while True:
            question = input("Wellcome to AmitaGPT, comment puis-je vous aider ? 😊"
                             "(ou tapez 'exit' pour quitter ) : ")

            if question.lower() == "exit":
                print("Au revoir 👋!")
                break

            response = self.rag.run_rag_prompt(question)
            print(f"Réponse : {response}")

    def main(self):
        st.title("AMITA GPT")

        # Initialize session state for chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Input field for user question
        question = st.text_input("Salut! Comment puis-je t'aider ? 😊", "")

        # When the "Réponse" button is clicked
        if st.button("Réponse"):
            if question.strip():
                # Add user's question to chat history
                st.session_state.chat_history.append({"role": "user", "message": question})

                # Generate answer using the RAG system
                answer = self.rag.run_rag_prompt(question=question)

                # Add assistant's answer to chat history
                st.session_state.chat_history.append({"role": "assistant", "message": answer})

        # Display chat history
        st.write("### Conversation :")
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(f"**Vous** : {chat['message']}")
            else:
                st.markdown(f"**AmitaGPT** : {chat['message']}")

    # def main(self):
    #     st.title("AMITA GPT")
    #
    #     # Initialize session state for chat history
    #     if "chat_history" not in st.session_state:
    #         st.session_state.chat_history = []
    #
    #     while True:
    #         question = st.text_input("Salut! Comment puis-je t'aider ? 😊", "")
    #
    #         if st.button("Réponse"):
    #             answer = self.rag.run_rag_prompt(question=question)
    #             st.write({answer})
    #
    #     # Display chat history
    #     st.write("### Historique de la conversation :")
    #     for chat in st.session_state.chat_history:
    #         st.markdown(f"**Vous** : {chat['question']}")
    #         st.markdown(f"**AmitaGPT** : {chat['response']}")



if __name__ == "__main__":
    rag = InteractiveRAG()
    rag.run_dag()
