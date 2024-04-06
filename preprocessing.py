from PyPDF2 import PdfReader #library to read pdf files
from langchain.text_splitter import RecursiveCharacterTextSplitter#library to split pdf files
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings #to embed the text
import google.generativeai as genai

from langchain_community.vectorstores import FAISS #for vector embeddings
from langchain.chains.question_answering import load_qa_chain #to chain the prompts
from dotenv import load_dotenv

load_dotenv()



genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    # iterate over all pdf files uploaded
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        # iterate over all pages in a pdf
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    # create an object of RecursiveCharacterTextSplitter with specific chunk size and overlap size
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 1000)
    # now split the text we have using object created
    chunks = text_splitter.split_text(text)

    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001") # google embeddings
    vector_store = FAISS.from_texts(text_chunks,embeddings) # use the embedding object on the splitted text of pdf docs
    vector_store.save_local("faiss_index") # save the embeddings in local

pdf_docs = []
raw_text = get_pdf_text()
text_chunks = get_text_chunks(raw_text)
get_vector_store(text_chunks)



