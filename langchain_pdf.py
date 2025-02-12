import os
import asyncio
import bs4
import faiss
from langchain.chat_models import init_chat_model
from langchain_google_vertexai import VertexAiEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter

API_KEY = "HxJUk7npOHvvLjQ08gIYvMHNu4BPbR00"
llm_model = "mistral-small-latest"
embed_model = "mistral-embed"


# Creating a LLM model
#model = init_chat_model(llm_model)

data = [
    "RAG is a new and upcoming architecture",
    "It helps by making sure the responses of the LLM is factually grounded", 
    "It is a useful propject to showcase on your resume"
]

# Creating a pdf loader

file_path = "/Users/vaibhavvemani/Downloads/test-book.pdf"
loader = PyPDFLoader(file_path)
async def pdf_loader(loader):
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages

output = asyncio.run(pdf_loader(loader))

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=300)
splits = text_splitter.split_documents(output)

# Creating Embeddings of the pdf pages
embeddings = VertexAiEmbeddings(model = embed_model )

vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever()

# index = faiss.IndexFlatL2(1024) 
# vectorstore = FAISS(
#     embedding_function = embeddings, 
#     index = index, 
#     docstore = InMemoryDocstore(),
#     index_to_docstore_id = {}
# )

# store = vectorstore.add_documents(documents = splits)

retrieved_docs = vectorstore.similarity_search("Who is Gregor", k=2)
print(retrieved_docs) 



