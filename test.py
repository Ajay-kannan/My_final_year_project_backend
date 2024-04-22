import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import csv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


llm = ChatGoogleGenerativeAI(model="gemini-pro")
result = llm.invoke("Write a ballad about LangChain")
print(result.content)

# def get_pdf_text(pdf_docs):
#     text = " "
#     # Iterate through each PDF document path in the list
#     for pdf in pdf_docs:
#         # Create a PdfReader object for the current PDF document
#         pdf_reader = PdfReader(pdf)
#         # Iterate through each page in the PDF document
#         for page in pdf_reader.pages:
#             # Extract text from the current page and append it to the 'text' string
#             text += page.extract_text()

#     # Return the concatenated text from all PDF documents
#     return text



# def read_csv_file(csv_file):
#     """Reads a CSV file and returns the text from the first column."""
#     try:
#         with open(csv_file, 'r', encoding='utf-8') as file:
#             csv_reader = csv.reader(file)
#             # Initialize an empty string to store the text
#             text = ""
#             # Iterate over each row in the CSV file
#             for row in csv_reader:
#                 # Add the text from the first column to the 'text' variable
#                 text += row[0] + " "
#             return text.strip()  # Remove leading and trailing whitespace
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None



# # The RecursiveCharacterTextSplitter takes a large text and splits it based on a specified chunk size.
# def get_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):     
#     # Create embeddings using a Google Generative AI model
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     # Create a vector store using FAISS from the provided text chunks and embeddings
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

#     # Save the vector store locally with the name "faiss_index"
#     vector_store.save_local("faiss_index")


# def get_conversational_chain():
#     # Define a prompt template for asking questions based on a given context
#     prompt_template = """
#      You are an AI Chatbot developed to help users by suggesting eco-friendly farming methods, fertilizers, and maximizing profits. Use the following pieces of context to answer the question at the end. Greet Users!!
#     {context}
#     {question}
#     """

#     # Initialize a ChatGoogleGenerativeAI model for conversational AI
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

#     # Create a prompt template with input variables "context" and "question"
#     prompt = PromptTemplate(
#         template=prompt_template, input_variables=["context", "question"]
#     )

#     # Load a question-answering chain with the specified model and prompt
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain



# def user_input(user_question):
#     # Create embeddings for the user question using a Google Generative AI model
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     # Load a FAISS vector database from a local file
#     new_db = FAISS.load_local("faiss_index", embeddings)

#     # Perform similarity search in the vector database based on the user question
#     docs = new_db.similarity_search(user_question)

#     # Obtain a conversational question-answering chain
#     chain = get_conversational_chain()

#     # Use the conversational chain to get a response based on the user question and retrieved documents
#     response = chain(
#         {"input_documents": docs, "question": user_question}, return_only_outputs=True
#     )

#     # Print the response to the console
#     print(response)

#     # Display the response in a Streamlit app (assuming 'st' is a Streamlit module)
#     # st.write("Reply: ", response["output_text"])

# pathoffile = "C:/Users/Ajay kannan/Desktop/reactprogram/practice/DataSet/formatted-text.pdf"

# raw_text = get_pdf_text(pathoffile)
# text_chunks = get_chunks(raw_text)
# get_vector_store(text_chunks)



# def main():
#     st.set_page_config("Chat PDF")
#     st.header("Chat with PDF using Gemini")

#     user_question = st.text_input("Ask a Question from the PDF Files")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader(
#             "Upload your PDF Files and Click on the Submit & Process Button",
#             accept_multiple_files=True,
#         )
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 print(raw_text)
#                 text_chunks = get_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done")


# if __name__ == "__main__":
#     main()

   # if not embeddings:
    #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # if not vector_store:
    #     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

   # Create embeddings for the user question using a Google Generative AI model




# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory

# conversation = ConversationChain(
#     llm=llm,
#     verbose=True,
#     memory=ConversationBufferMemory()
# )
    
#     PROMPT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

#     Current conversation:
#     {history}
#     Human: {input}
#     AI Assistant:
# """

#     PROMPT = PromptTemplate( input_variables=["history", "input"], template=PROMPT_TEMPLATE)

#     conversation = ConversationChain(
#     llm=llm,
#     verbose=True,
#     prompt=PROMPT,
#     memory=ConversationBufferMemory(ai_prefix="AI Assistant")
# )

#   result = conversation.predict(input=user_question)

#     print(result)


# def get_text_from_pdf(pdf_file):
#     """Extracts text from a PDF document."""
#     with open(pdf_file, 'rb') as f:
#         pdf_reader = PdfReader(f)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text


# def get_chunks(text):
#     """Splits text into chunks using RecursiveCharacterTextSplitter."""
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def preprocess_text(text):
#     """Preprocesses the text (e.g., cleaning, normalization)."""
#     # Implement your specific preprocessing steps here
#     return text.lower().strip()

# def create_vector_store(text_chunks):
#     """Creates a FAISS vector store from text chunks and embeddings."""
#     # Ensure vector store initialization is done only once or conditionally
#     global embeddings, vector_store
#     if not embeddings:
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     if not vector_store:
#         vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
#     return vector_store

   