from langchain_core.documents import Document
#from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_pinecone import PineconeVectorStore
#from langchain_openai import ChatOpenAI
from typing import List

# Extract text from PDF files
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs = []

    for doc in docs:

        text = doc.page_content.strip()

        if not text:
            continue

        minimal_docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                },
            )
        )

    return minimal_docs


# Split the documents into smaller chunks
def text_split(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    return splitter.split_documents(documents)


def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )