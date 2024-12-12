from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import HumanMessage
from retreival import Retrieval
import logging
import time

class RagPrompt :
    def __init__(self, retriever_instance : Retrieval):
        self.template = """Répondez à la question en utilisant UNIQUEMENT le contexte suivant : {context}
        Question : {question}
        """
        # self.retriever_instance = Retrieval()
        # self.retriever_instance = retriever_instance
        self.llm = retriever_instance.llm
        self.context = retriever_instance.run_retrieval()

    def run_rag_prompt(self, question : str):
        start_time = time.time()
        logging.info('Run run_rag_prompt')
        logging.info('1. prompt')
        prompt = ChatPromptTemplate.from_template(self.template)
        retrieved_docs = self.context.get_relevant_documents(question)
        # Collect context and sources
        # context = " ".join([doc.page_content for doc in retrieved_docs])
        sources = [doc.metadata.get("source", "Unknown") for doc in retrieved_docs]
        sources = set(sources)
        sources = list(sources)[:2]

        end_time = time.time()
        logging.info(f'1. prompt done {end_time - start_time}')
        # print(prompt)
        # print(prompt.invoke({"context": retriever.run_retrieval(), "question": RunnablePassthrough()}))
        logging.info('2. chain')
        chain = (
                {"context": self.context, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )
        end_time = time.time()
        logging.info(f'2. chain done {end_time - start_time}')
        # print({"context": self.retriever_instance, "question": question})
        # return chain.invoke({"context": self.retriever_instance, "question": question})
        logging.info('3. result')
        result = chain.invoke(question)
        end_time = time.time()
        logging.info(f'3. result done {end_time - start_time}')

        end_time = time.time()
        logging.info(f'run_rag_prompt done {end_time - start_time}')
        return {"response": result, "resources": sources} #result



if __name__ == '__main__':
    rag = RagPrompt()
    # retrieval = Retrieval()
    # answer = rag.run_rag_prompt(question= r"parles moi de la data chez amita")
    # print(answer)

    while True:
        question = input("Wellcome to AmitaGPT, comment puis-je vous aider ? 😊 "
                         "(ou tapez 'exit' pour quitter ) : ")

        if question.lower() == "exit":
            print("Au revoir 👋!")
            break

        response = rag.run_rag_prompt(question)
        print(f"Réponse : {response}")


