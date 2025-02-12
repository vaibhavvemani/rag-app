import bs4
import os
import getpass
from langchain_community.document_loaders import WebBaseLoader
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from pinecone import Pinecone as pc
web_loader = WebBaseLoader(
    web_paths = ("https://blog.google/technology/ai/google-gemini-ai/", ),
)
#os.environ['GOOGLE_API_KEY'] = getpass.getpass('Gemini API Key:')
#os.environ['PINECONE_API_KEY'] = getpass.getpass('Pinecone API Key:')

docs = web_loader.load()
text_content = docs[0].page_content
text_content_1 = text_content.split("code, audio, image and video.",1)[1]
final_text = text_content_1.split("Cloud TPU v5p",1)[0]

docs = [Document(page_content=final_text, metadata={"source": "local"})]

gemini_embeddings = GoogleGenerativeAIEmbeddings(model = 'models/embeddings-001')

pinecone_client = pc(api_key = os.getenv("PINECONE_API_KEY"))

index_name = "rag-testing"

if index_name not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(name = index_name, metric = "cosine", dimension=768)

vectorstore = pc.from_documents(docs, gemini_embeddings, index_name=index_name)

