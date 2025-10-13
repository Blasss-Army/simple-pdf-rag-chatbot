import os
from dataclasses import dataclass
from .conf import RetrieverConfig
import uuid

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter,
    TokenTextSplitter,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore as Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Filter, FilterSelector
from langchain.schema import Document as LCDocument


class Retriever():
    def __init__(self, cfg: RetrieverConfig):
        # 1) Retriever configuration
        self.cfg = cfg              
        # 2) Load documents from the 'data' directory
        self.all_document_pages = self.load_documents_from_directory(self.cfg.data_path)
        # 3) Create chunks from the loaded documents
        self.chunks = self.create_chunks_splits(self.all_document_pages, chunk_size=1000, chunk_overlap=200, splitter='recursive')
        # 4) Build the Qdrant retriever
        self.client, self.retriever = self.build_retriever(chunks=self.chunks, collection_name=self.cfg.collection_name, path=self.cfg.index_path)
                    
    def close(self):
            '''Close the client sesion'''
            self.client.close() # Finally, close the client connection -->>> Good practice , avoiding resource leaks and warning messages.


    def load_documents_from_directory(self, directory_path): # Ruta absoluta a /data junto a este archivo 

        '''
        Load all PDF documents from a specified directory.
        Args:
            directory_path (str): Path to the directory containing PDF documents.
        Outputs:
            list: List of loaded PDF documents.
        '''

        all_document_pages = []
        loader = DirectoryLoader(
                str(directory_path),  # Convertir a cadena
                glob="**/*.pdf",          # todos los PDFs (subcarpetas incluidas)
                loader_cls=PyPDFLoader
            )
        all_document_pages = loader.load()
        return all_document_pages
    
    def load_documents_from_gradio(self, documents):
        docs = []
        for i in documents:
            loader = PyPDFLoader(i)
            pages = loader.load()
            docs.extend(pages
                        )
        return docs

    def create_chunks_splits(self, documents, chunk_size=1000, chunk_overlap=200, splitter = 'recursive'):  # By default, use recursive splitter

        '''
        Split documents into smaller chunks for better processing.

        Args:
            documents (list): List of documents to be split.
            chunk_size (int): Size of each chunk.
            chunk_overlap (int): Overlap between chunks.
            splitter (str): Type of text splitter to use ('recursive', 'character', 'markdown', 'token').
        Outputs:
            list: List of document chunks.
        '''

        if splitter == 'recursive':
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        elif splitter == 'character':
            text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
        elif splitter == 'markdown':
            text_splitter = MarkdownTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)

        elif splitter == 'token':
            text_splitter = TokenTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        else:
            raise ValueError("Invalid splitter type. Choose from 'recursive', 'character', 'markdown', or 'token'.")
        
        chunks = text_splitter.split_documents(documents)
        return chunks
        
    def build_retriever(self, chunks, collection_name, path):

        '''
        Build a Qdrant retriever from document chunks, using Google Generative AI embeddings and local Qdrant vector Database.

        Args:
            chunks (list): List of document chunks.
            collection_name (str): Name of the Qdrant collection.
            path (str): Path to store the Qdrant local database.
        Outputs:
                Qdrant retriever object.
        '''

        # 1) Initialize Google Generative AI Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model=self.cfg.embed_model)

        # 2) Initialize Qdrant client (local instance) -->>> Good practice
        client = QdrantClient(
            path=str(path),                          # Convertir a cadena
            prefer_grpc=self.cfg.prefer_grpc,        # Cambiar a False si hay problemas con gRPC
        )

        # 3) Create the collection (Only if it doesnt exits)
        if not client.collection_exists(collection_name):
            embedding_dim = len(embeddings.embed_query("Hello, world!"))
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
            )
            
        # 3.1) If reset_collection=True, delete all points in the collection
        if self.cfg.reset_collection:
            client.delete(
                collection_name=collection_name,
                points_selector=FilterSelector(
                    filter=Filter(must=[])
                ),
                wait=True,  
            )

        # 4) Create Qdrant vectorstore from chunks 
        vectorestore = Qdrant(
            embedding=embeddings,
            client=client,
            collection_name=collection_name
        )

        # 5) Add chunks and metadatas to vectorestore (It only indexes if the collection is empty)
        if (client.count(collection_name=self.cfg.collection_name,exact=True,).count == 0):
             ids = self._make_ids(chunks)
             vectorestore.add_documents(chunks, ids=ids)

        # 6) kwargs seguros según search_type
        kwargs = {"k": self.cfg.vectore_store_k}
        if self.cfg.vectore_store_search_type == "mmr":
            kwargs.update({
                "fetch_k": self.cfg.vectore_store_fetch_k,
                "lambda_mult": float(self.cfg.vectore_store_lambda_mult),
            })

        # 7) Create retriever from vectorestore
        retriever = vectorestore.as_retriever(search_type= self.cfg.vectore_store_search_type,
                                                search_kwargs=kwargs,
                                             )
        
        if (client.count(collection_name=self.cfg.collection_name,exact=True,).count == 0):
             ids = self._make_ids(chunks)
             retriever.add_documents(chunks, ids=ids)

        return client,retriever
    
    def add_chunks_into_vectorestorage(self,chunks):
        ids = self._make_ids(chunks)
        self.retriever.add_documents(chunks, ids=ids)
        return 'New chunks have been uploaded into the Retriever'

    
    def _make_ids(self, chunks):
        '''Generate **deterministic, human-traceable IDs** for document chunks.
            Args:
                chunks (list[langchain.schema.Document]): Chunked documents containing metadata
                    such as:
                    - "source": logical path or filename (prefer a content hash for robustness)
                    - "page": page number in the source PDF
                    - "start_index": character offset used by the splitter

            Returns:
                list[str]: A list of UUIDv5 strings aligned positionally with `chunks`. These IDs
                are intended to be passed to the vector store's add/upsert operation.'''
        ids = []
        for i, d in enumerate(chunks):
            src   = d.metadata.get("source", "doc")
            page  = d.metadata.get("page", -1)
            start = d.metadata.get("start_index", i)
            key = f"{src}|p{page}|s{start}"
            ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, key)))

        return ids

    def get_relevant_documents(self, query):

        '''
        Retrieve relevant documents based on a query using the provided retriever.

        Args:
            query (str): The query string to search for relevant documents.
        Outputs:
            list: List of relevant documents.
        '''
        return self.retriever.invoke(query)
    

    def vector_count(self) -> int:
        """Número de vectores (points) en la colección actual."""
        return self.client.count(
            collection_name=self.cfg.collection_name,
            exact=True,   # True = conteo exacto
        ).count
