from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import HumanMessage
from retreival import Retrieval


class RagPrompt :
    def __init__(self, retriever_instance : Retrieval):
        self.template = """Répondez à la question en utilisant UNIQUEMENT le contexte suivant : {context}
        Question : {question}
        """
        # self.retriever_instance = Retrieval()
        self.retriever_instance = retriever_instance
        self.llm = self.retriever_instance.llm
        self.context = self.retriever_instance.run_retrieval()

    def run_rag_prompt(self, question : str):
        prompt = ChatPromptTemplate.from_template(self.template)
        # print(prompt)
        # print(prompt.invoke({"context": retriever.run_retrieval(), "question": RunnablePassthrough()}))
        chain = (
                {"context": self.context, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )
        # print({"context": self.retriever_instance, "question": question})
        # return chain.invoke({"context": self.retriever_instance, "question": question})
        return chain.invoke(question)



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


