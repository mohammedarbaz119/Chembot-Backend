import pymongo
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.embeddings.cloudflare_workersai import CloudflareEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv
import os

load_dotenv()




Embed_Model = CloudflareEmbedding(
    account_id=os.getenv("cloudFareAccountID"),
    auth_token=os.getenv("cloudfareToken"),
    model="@cf/baai/bge-small-en-v1.5",
)

mongodb_client = pymongo.MongoClient(os.getenv("MongoConn"))

store = MongoDBAtlasVectorSearch(
    mongodb_client,
    db_name = os.getenv("DB_NAME"),
    collection_name =os.getenv("COLLECTION_NAME") ,
    vector_index_name =os.getenv("VECTOR_INDEX_NAME") 
)
storage_context = StorageContext.from_defaults(vector_store=store)

index = VectorStoreIndex.from_vector_store(
   store,embed_model=Embed_Model
)