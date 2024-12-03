# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.schema.document import Document
#
# from load_pdf import load_documents
#
#
# def split_documents(docuemnts : list[Document]):
#     text_spliter = RecursiveCharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=80,
#         length_function=len,
#         is_separator_regex=False,
#     )
#     return text_spliter.split_documents(docuemnts)
#
# if __name__ == '__main__':
#     load = load_documents()
#     chunks = split_documents(docuemnts=load)
#     print(chunks[0])

