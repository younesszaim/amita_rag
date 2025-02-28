from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from datetime import datetime
import hashlib
import os
from load_pdf import LoadAndSplitDocuments

def update_vector_store_from_sharepoint(): 

    directory = "./chroma_index"
    embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")
    load_data = LoadAndSplitDocuments()
    document_chunks = load_data.run_load_and_split_documents()

    # Ajout métadonnées manquantes
    for doc in document_chunks:
        if "source" not in doc.metadata:
            doc.metadata["source"] = "SharePoint"
        if "last_modified" not in doc.metadata:
            doc.metadata["last_modified"] = datetime.now().isoformat().strftime("%Y-%m-%d %H:%M:%S")
        doc.metadata["hash"] = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()

    if os.path.exists(directory) and len(os.listdir(directory)) > 0:
        print("Chargement du vector store existant")
        db = Chroma(persist_directory=directory, embedding_function=embedding_function)
        # Récupérer les documents existants avec leurs métadonnées
        existing_docs = db._collection.get(include=["metadatas", "documents"])
        existing_docs_metadata = {
            meta["source"]: {"last_modified": meta["last_modified"], "hash": meta.get("hash", "")}
            for meta in existing_docs["metadatas"]
        }
        documents_to_add = []
        documents_to_update = []
        sources_to_remove = []

        #Documents à ajouter ou mettre à jour (date différente)
        for doc in document_chunks:
            source = doc.metadata["source"]            
            hash_value = doc.metadata["hash"]
            last_modified = doc.metadata["last_modified"]
            if source not in existing_docs_metadata:
                documents_to_add.append(doc)
            elif existing_docs_metadata[source]["last_modified"] != last_modified:
                documents_to_update.append(doc)

        # Documents à supprimer
        existing_sources = {meta["source"] for meta in existing_docs["metadatas"]}
        new_sources = {doc.metadata["source"] for doc in document_chunks}
        sources_to_remove = list(existing_sources - new_sources)

        # Mettre à jour le vector store
        print(f"Ajout : {len(documents_to_add)}, Mise à jour : {len(documents_to_update)}, Suppression : {len(sources_to_remove)}")
        if documents_to_add or documents_to_update or sources_to_remove:

            # Suppression des documents obsolètes
            if sources_to_remove:
                db._collection.delete(where={"source":{"$in":sources_to_remove}})
                print("Suppression")
            #Suppression ancienne version
            if documents_to_update:
                sources_to_delete = [doc.metadata["source"] for doc in documents_to_update]
                db._collection.delete(where={"source": {"$in": sources_to_delete}})

            # Ajout des nouveaux documents
            db.add_documents(documents_to_add + documents_to_update)
            print("Ajout et mise à jour")
    else:
        print("Création d'un nouveau vector store avec persistance...")
        # Créer un nouveau vector store avec persistance locale
        db = Chroma.from_documents(
        documents=document_chunks,
        embedding=embedding_function,
        persist_directory=directory
        )

    #db.persist()
    print(" Vector store enregistré localement avec succès !")

update_vector_store_from_sharepoint()

