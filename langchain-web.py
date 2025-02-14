from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PDFBaseLoader
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAI

from typing_extensions import TypedDict, List

from pinecone import Pinecone as pc

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

pinecone_client = pc(api_key = PINECONE_API_KEY)

def load_website(vectorstore, url: str):
    loader = WebBaseLoader(web_paths = [url])
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    splits = text_splitter.split_documents(docs)
    vectorstore.add_documents(splits)

def load_pdf(vectorstore, path: str):
    loader = PDFBaseLoader(pdf_paths = [path])
    docs = loader.load()
    text_spliiter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=50)
    splits = text_spliiter.split_documents(docs)
    vectorstore.add_documents(splits)

def pinecone_init(embedding_model, host):
    index = pinecone_client.Index(host = host)
    vectorstore = PineconeVectorStore(embedding = embedding_model, index = index)
    return vectorstore

    

if __name__ == "__main__":
    website_url = "https://en.wikipedia.org/wiki/Mahabrahma"
    pinecone_host = "https://rag-testing-m4lr6ld.svc.aped-4627-b74a.pinecone.io"
    llm_model = "models/gemini-2.0-flash-thinking-exp-01-21"
    embedding_model = "models/embedding-001"

    llm = GoogleGenerativeAI(model=llm_model, temperature=0.7, top_p=0.85)
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model = embedding_model)

    prompt_template = """You are an assistant for question answering tasks.
    Use only the given context to answer the question.
    If you don't know the answer just say I don't know.
    Use a maximum of 3 sentences to answer the question.

    Question: {input}
    Context: {context}
    Answer: """

    PROMPT = PromptTemplate.from_template(prompt_template)

    vectorstore = pinecone_init(gemini_embeddings, pinecone_host)
    retriever = vectorstore.as_retriever()
    chain = load_qa_chain(llm, chain_type="stuff")

    while True:
        question = input("Enter your question: ")
        if question.lower() == "exit":
            print("Exiting...")
            break
        elif question.lower() == "pdf":
            path = input("Enter the path to the pdf: ")
            docs = load_pdf(vectorstore, path)
        elif question.lower() == "web":
            url = input("Enter the url: ")
            docs = load_website(vectorstore, url)
        docs = retriever.invoke(question)
        response = chain.invoke({"input_documents": docs, "question": question})
        print(response["answer"])


