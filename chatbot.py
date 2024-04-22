from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
import chromadb
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import websearch
import re
from deep_translator import GoogleTranslator


load_dotenv()

def instantiate_embedding_model():

    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",device="cpu") # choose the device to load the model to (note: GPU will often be *much* faster than CPU)
    # Make sure the model is on the CPU
    embedding_model.to("cpu")
    return embedding_model


def instantiate_DB():
    chroma_client = chromadb.PersistentClient(path="C:/Users/Ajay kannan/Desktop/reactprogram/practice/database")
    # Load collection, or create new on first run. Specify the model that we want to use to do the embedding.
    chroma_collection = chroma_client.get_or_create_collection(name='vectorstore')

    return chroma_collection

def instantiate_LLM(temperature=0.5,top_p=0.95):

    llm = ChatGoogleGenerativeAI(
            google_api_key= os.getenv("GOOGLE_API_KEY") ,
            model="gemini-pro",
            temperature=temperature,
            top_p=top_p,
            convert_system_message_to_human=True
        )
    return llm

def text_preprocessing(text):
    if text.startswith("content='"):
        # Remove if the text starts with content = "
        text = re.sub(r"^content='", '', text)

    # Replace \\n with \n
    text = text.replace("\\n", "\n")

    # Remove double stars but not single stars
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)

    return text


def generate_response(question):
    
    embedding = instantiate_embedding_model()
    db = instantiate_DB()
    llm = instantiate_LLM()
    
    # Encode the query sentence using the model
    query_embedding = embedding.encode(question)

    results = db.query(
    query_embeddings=query_embedding.tolist(),
    n_results= 5 ,
    )
    context = ""
    doc_results = results["metadatas"]
    doc_results = doc_results[0]
    for idx in range(len(doc_results)):
        context = context + doc_results[idx]["text"] + "\n\n"

    context_from_web = websearch.web_search(question)

    # print("context_from_web :", context_from_web["context_text"])
    # print("\n")
    # print("context from db", context)
    context = context_from_web["context_text"]+ "\n\n" + context

    print(context)

    
    prompt_template = """ 
    You are an AI Chatbot developed to help users by suggesting verbosely eco-friendly farming methods. Use the following pieces of context to answer the question . Greet Users!!
   
    User query:  {question}

    use the following context items to answer the user query:
    {context}

    
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    prompt_result = PROMPT.format(question=question , context = context )

    results_answer = llm.invoke(prompt_result)

    answer_in_english =  text_preprocessing(str(results_answer))

    answer_in_tamil = GoogleTranslator(source='en', target='ta').translate(answer_in_english)
    
    results_all = {"messageContent" : answer_in_english ,"messageContent_tamil" : answer_in_tamil  ,  "url_link": context_from_web["url_link"] , "image_links" : context_from_web["image_links"] , "video_links" : context_from_web["video_links"]}

    return results_all



#  please answer the query.
#     Give yourself room to think by extracting relevant passages from the context before answering the query.
#     Don't return the thinking, only return the answer.
#     Make sure your answers are as explanatory as possible.
#     Use the following examples as reference for the ideal answer style.

# Example 1:
# Query: What are some cultivation tips for tomatoes?
# Answer: Cultivation tips for tomatoes include choosing the right variety for your climate, providing ample sunlight, proper watering to keep the soil consistently moist but not waterlogged, using mulch to retain moisture and prevent weeds, providing support for the plants as they grow, and regularly pruning to promote air circulation and reduce the risk of disease.

 