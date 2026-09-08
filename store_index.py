from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, load_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not configured.")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not configured.")

extracted_data=load_pdf_files(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks=text_split(filter_data)

embeddings = load_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"
namespace = "medical-encyclopedia"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension=384,  # Dimension of the embeddings
        metric= "cosine",  # Cosine similarity
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,
    namespace=namespace
)