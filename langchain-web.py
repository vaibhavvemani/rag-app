import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAI

from pinecone import Pinecone as pc
from pinecone import ServerlessSpec

os.environ["GOOGLE_API_KEY"] = "AIzaSyBF-wxgd4Fm_3sTFcAL3u-WaZQh7aLzqAM"
os.environ["PINECONE_API_KEY"] = "pcsk_3Mwc1n_LMUVRZ6GzBwi9XzeB4Wruha3rDgghsuxRBquutiiRqpva1Miydd4SAyYTzGwekg"

website_url = "https://en.wikipedia.org/wiki/Mahabrahma"
pinecone_host = "https://rag-testing-m4lr6ld.svc.aped-4627-b74a.pinecone.io"
llm_model = "models/gemini-2.0-flash-thinking-exp-01-21"
embedding_model = "models/embedding-001"

pinecone_client = pc(api_key = os.getenv("PINECONE_API_KEY"))

web_loader = WebBaseLoader(web_paths = "https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = web_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
splits = text_splitter.split_documents(docs)

llm = GoogleGenerativeAI(model=llm_model, temperature=0.7, top_p=0.85)
gemini_embeddings = GoogleGenerativeAIEmbeddings(model = embedding_model)

if not "rag-testing" in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name = "rag-testing",
        dimension = 768,
        metric = "cosine",
        spec = ServerlessSpec(

        )
    )

index = pinecone_client.Index(host = pinecone_host)
vectorstore = PineconeVectorStore(embedding = gemini_embeddings, index = index)
vectorstore.add_documents(documents=splits)

prompt_template = """You are an assistant for question answering tasks.
Use only the given context to answer the question.
If you don't know the answer just say I don't know.
Be detailed in your answers.

Question: {question}
Context: {context}
Answer: """

prompt = PromptTemplate.from_template(prompt_template)

question = "What is gemini?"
retrieved_docs = vectorstore.similarity_search(question, k=5)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
prompt = prompt.invoke({"question": question, "context": docs_content} )

response = llm.generate([prompt.text])
print(response[0].text)
