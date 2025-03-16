import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import os

# Model til embedding af 
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2",  #Anden mulighed "BAAI/bge-small-en-v1.5"
                                   local_files_only=True)
# LLM til generering af tekst
llm = Ollama(model="gemma2:9b")

# Database placering
db_path = "./chroma_db"

# Hvis Choma database eksistere
if os.path.exists(db_path):
    print("ChromaDB already exists. Reusing existing database.")
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(     # from_vector_store var løsningen ift. at hente fra eksisterende db
        vector_store=vector_store,
        embed_model=embed_model
    )
# Hvis Chroma database ikke eksistere, så opret den.
else:
    print("ChromaDB does not exist. Creating new database.")
    documents = SimpleDirectoryReader("./data").load_data()
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context, 
        embed_model=embed_model
    )

# Opret query engine
query_engine = index.as_query_engine(llm=llm)
# Hent svar fra LLM
response = query_engine.query("Skriv en ansøgning ud jobopslaget, med udgangspunkt i mit CV. Skriv i samme tone. Præsenter dig som Gemma2 der skriver ansøgningen på mine vegne, forklar du er implementeret i Python og koden kan findes på følgende github link: https://github.com/kimhaagenmathiesen/cv_rag/blob/main/data/cv.pdf.")
print(response)