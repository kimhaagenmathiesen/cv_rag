import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import os

# Model for embedding
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# LLM for generating text
llm = Ollama(model="gemma2:9b")

# Database location
db_path = "./chroma_db"

# If ChromaDB already exists
if os.path.exists(db_path):
    print("ChromaDB already exists. Reusing existing database.")
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(    
        vector_store=vector_store,
        embed_model=embed_model
    )
# If ChromaDB does not exist, create it
else:
    print("ChromaDB does not exist. Creating new database.")
    documents = SimpleDirectoryReader("./data").load_data()
    print(f"Loaded {len(documents)} documents")

    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,  
        storage_context=storage_context, 
        embed_model=embed_model
    )

# Create query engine
query_engine = index.as_query_engine(llm=llm, verbose=True)
# Get response from LLM
response = query_engine.query("Write a job application based on my CV for the AI Software Engineer position.")
print(response)

# Retrieve and preview context nodes
retriever = index.as_retriever()
results = retriever.retrieve("Write a job application based on my CV for the AI Software Engineer position.")

