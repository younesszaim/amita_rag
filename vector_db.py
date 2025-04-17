from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

from embedding import get_embedding_function
from langchain.vectorstores.chroma import Chroma

from _old_load_pdf_2 import load_documents
from split_documents import split_documents


def add_to_chroma(chunks : list[Document]):
    # db = Chroma(persist_directory='db',
    #             embedding_function=get_embedding_function())
    # ids = [i for i in range(len(chunks))]
    # db.add_documents(chunks, id=ids)
    # db.persist()
    vector_db = Chroma.from_documents(
        documents= chunks,
        embedding=  OllamaEmbeddings(model = 'nomic-embed-text', show_progress= True),
        collection_name= 'rag'
    )
    return vector_db


if __name__ == '__main__':
    load = load_documents()
    chunks = split_documents(docuemnts=load)
    # single_vector = get_embedding_function().embed_query(chunks[0].page_content)
    # print(single_vector)
    #
    # two_vectors = get_embedding_function().embed_documents([chunk.page_content for chunk in chunks])
    # print(two_vectors)

    print(add_to_chroma(chunks))

