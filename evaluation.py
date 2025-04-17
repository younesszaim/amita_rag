import logging
import os
import time
import csv
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mistralai import ChatMistralAI,MistralAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain.retrievers import MultiQueryRetriever 
from langchain.chains import ConversationalRetrievalChain,create_history_aware_retriever
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
#import hashlib
from datetime import datetime

from load_pdf import LoadAndSplitDocuments

load_dotenv(dotenv_path='.config')

class EvaluationRAG:
    def __init__(self):
        #os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY', '')
        os.environ["MISTRAL_API_KEY"] = os.getenv('MISTRAL_API_KEY', '')
        self.embedding_function = self.get_embedding_function()
        self._load_or_create_vector_db()
        # self.llm = ChatOpenAI(
        #     model="gpt-4o-mini",
        # #     model="gpt-3.5-turbo",
        #     temperature=0.7,
        #     max_tokens=None,
        # )
        self.llm = ChatMistralAI(
            model="mistral-small-latest",
            temperature=0.7,
            max_tokens=None,
            #random_seed=1
        )
        
        self.history_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")  
        ])
    
        self.query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
                Vous êtes un agent conversationnel pour les collaborateurs. 
                Reformulez cette question de deux manières différentes pour maximiser la récupération d'informations pertinentes.
                
                Question : {question}
            """
        )

        self.template_context_only = """Répondez à la question en vous appuyant uniquement sur le contexte suivant : {context}
            Question : {question}
            Quand tu ne sais pas, réponds que tu ne sais pas.
            """
        
        self.template_mixed = """Répondez à la question en utilisant vos connaissances ainsi que le contexte suivant : {context}
            Question : {question}
            """
        self.retriever = MultiQueryRetriever.from_llm(
            self.db.as_retriever(search_kwargs={"k": 10}),
            self.llm,
            prompt=self.query_prompt
        )  
        # self.retriever = create_history_aware_retriever(
        #     llm=self.llm,
        #     retriever=multi_query_retriever,
        #     prompt=self.history_prompt
        # )

    def _load_or_create_vector_db(self):
        #vector_db_path = "./faiss_index"
        #vector_db_path = "./chroma_index_openai"
        vector_db_path = "./chroma_index_mistral"

        if os.path.exists(vector_db_path):
            # Load existing vector store
            self.db = Chroma(persist_directory=vector_db_path, embedding_function=self.embedding_function)
            #self.db = FAISS.load_local(vector_db_path, self.embedding_function, allow_dangerous_deserialization=True)
        else:
            # Create and save vector store
            load_data = LoadAndSplitDocuments()
            document_chunks = load_data.run_load_and_split_documents()

            # Add sourcing metadata
            for doc in document_chunks:
                if "source" not in doc.metadata:
                    doc.metadata["source"] = "SharePoint"
                if "last_modified" not in doc.metadata:
                    doc.metadata["last_modified"] = datetime.now().isoformat().strftime("%Y-%m-%d %H:%M:%S")
                #doc.metadata["hash"] = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
                
            # # Create vector store
            # self.db = FAISS.from_documents(document_chunks,
            #                                self.embedding_function)

            # # Persist vector store
            # self.db.save_local(vector_db_path)
     
            self.db = Chroma.from_documents(
            documents=document_chunks,
            embedding=self.embedding_function,
            persist_directory=vector_db_path
            )
    def get_embedding_function(self):
        start_time = time.time()
        logging.info('get_embedding_function')
        #embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        embeddings = MistralAIEmbeddings(model="mistral-embed")
        end_time = time.time()
        logging.info(f'get_embedding_function done in {end_time - start_time}')
        return embeddings

    def run_rag_prompt(self, question: str, chat_history=None, use_model_knowledge=False):
        start_time = time.time()
        logging.info('Run run_rag_prompt')

        chat_history_messages = [
            HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
            for msg in chat_history[-3:]
        ] if chat_history else []

        retrieved_docs = self.retriever.invoke({
            "input": question,
            "chat_history": chat_history_messages
        })

        retrieval_time = time.time() - start_time

        template = self.template_mixed if use_model_knowledge else self.template_context_only
        prompt = ChatPromptTemplate.from_template(template)
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        chain_start_time = time.time()
        chain = (prompt | self.llm | StrOutputParser())
        result = chain.invoke({"context": context_text, "question": question})
        llm_time = time.time() - chain_start_time

        total_time = time.time() - start_time
        sources = list(set([doc.metadata.get("source", "Unknown") for doc in retrieved_docs]))

        return {
            "response": result,
            "retrieval_time": retrieval_time,
            "llm_time": llm_time,
            "total_time": total_time,
            "context_text":context_text,
            "sources": sources
        }

    def run_batch_evaluation(self, questions, output_csv="evaluation_results.csv"):
        with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Question", "Reponse", "Temps Recuperation (s)", "Temps LLM (s)", "Temps Total (s)","Contexte","Sources"])

            for question in questions:
                result = self.run_rag_prompt(question)
                writer.writerow([
                    question,
                    result["response"],
                    round(result["retrieval_time"], 3),
                    round(result["llm_time"], 3),
                    round(result["total_time"], 3),
                    ", ".join(result["context_text"]),
                    ", ".join(result["sources"])
                ])

        print(f"Résultats enregistrés dans {output_csv}")


questions=["Quelle est la date de création d’AMITA Conseil?",
           "Où se situent les différents bureaux d’AMITA Conseil?",
           "Quels sont les profils proposés par AMITA Conseil?",
"Décris-moi en 5 bullets points l’offre data d'AMITA Conseil?"
]

eval_rag = EvaluationRAG()
eval_rag.run_batch_evaluation(questions)