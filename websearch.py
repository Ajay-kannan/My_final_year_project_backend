import requests
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import shutil

load_dotenv()

def get_search_results_text(search_term):
    subscription_key = os.getenv("SUBSCRIPTION_KEY")

    search_url = "https://api.bing.microsoft.com/v7.0/search"

    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {"q": search_term, "textDecorations": True, "textFormat": "HTML"}

    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()

    # dict_keys(['_type', 'queryContext', 'webPages', 'images', 'relatedSearches', 'videos', 'rankingResponse'])

    count = 0
    scrape_text = ""
    url_links = []
    image_links = []
    videos_links = []

    pages = search_results["webPages"]
    text_results = pages["value"]


    if "images" in search_results :
        image = search_results["images"]
        image_results = image["value"]
        for result in image_results[:10]:
            if not "contentUrl" in result:
                continue
            else:
                image_links.append(result["contentUrl"])

    if "videos" in search_results:
        video = search_results["videos"]
        videos_results = video["value"]
        for result in videos_results[:10]:
            if not "contentUrl" in result:
                continue
            else:
                videos_links.append(result["contentUrl"])

    
    for result in text_results[:10]:
        try:
            response = requests.get(result['url'])
            content = response.content
            soup = BeautifulSoup(content, "html.parser")
            body = soup.find('body')
            if body:
                text = body.get_text().strip()
                cleaned_text = ' '.join(text.split())
                scrape_text = scrape_text + cleaned_text + "\n"
                url_links.append(result['url'])
                count = count + 1
                if(count > 3):
                    break
            else:
                print("No body element found on this page.")
        except Exception as e:
            print(f"Error occurred while processing {result['url']}: {e}")
    return scrape_text , url_links , image_links , videos_links



# The RecursiveCharacterTextSplitter takes a large text and splits it based on a specified chunk size.
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):     

    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",device="cpu") # choose the device to load the model to (note: GPU will often be *much* faster than CPU)
    # # Make sure the model is on the CPU
    embedding_model.to("cpu")

    chroma_client = chromadb.PersistentClient(path="C:/Users/Ajay kannan/Desktop/reactprogram/practice/database_web")
    # Load collection, or create new on first run. Specify the model that we want to use to do the embedding.
    chroma_collection = chroma_client.get_or_create_collection(name='vectorstore')
    
    for idx, item in enumerate(text_chunks):
  # Encode sentence chunk
        embedding = embedding_model.encode(item)

  # Generate a unique ID for each chunk
        chunk_id = str(uuid.uuid1())
        meta = {
      'text': item,
    }
        chroma_collection.add(
    documents=[item],
    embeddings=embedding.tolist(),
    metadatas=[meta],
    ids=[chunk_id]
)  

def  search_chroma(query):
    # Initialize Sentence Transformer model
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cpu")

    # Ensure the model is on the CPU
    embedding_model.to("cpu")

    # embedding_model.to("cuda")

    chroma_client = chromadb.PersistentClient(path="C:/Users/Ajay kannan/Desktop/reactprogram/practice/database_web")
    # Load collection, or create new on first run. Specify the model that we want to use to do the embedding.
    chroma_collection = chroma_client.get_or_create_collection(name='vectorstore')
    
    # Encode the query sentence using the model
    query_embedding = embedding_model.encode(query)

    res = chroma_collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=10,
    )
    # chroma_client.delete_collection(name="vectorstore")
   
    return res["metadatas"][0]



def web_search(query):
    scrape_text , url_links , image_links , videos_links = get_search_results_text(query)
    text_chunks = get_chunks(scrape_text)
    get_vector_store(text_chunks)
    results_text = search_chroma(query)
    context_text = ""
    for result in results_text :
        context_text = context_text + result["text"] + "\n\n"

    results_all = {"context_text" : context_text ,  "url_link": url_links , "image_links" : image_links , "video_links" : videos_links}
    
    
    return results_all

