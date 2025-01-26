import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Ollama
import streamlit as st
import traceback
import logging

logging.basicConfig(level=logging.INFO)

def search_bing(query):
    service = Service(r'C:\Users\Ziaulhaq Parsa\OneDrive\Desktop\Advance-Programming\assignment4\chromedriver-win64\chromedriver.exe')
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=service, options=options)

    try:
        logging.info("Opening Bing with query.")
        driver.get(f"https://www.bing.com/search?q={query.replace(' ', '+')}")

        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.b_algo'))
        )
        results = []
        search_results = driver.find_elements(By.CSS_SELECTOR, '.b_algo')
        for result in search_results:
            try:
                title_element = result.find_element(By.CSS_SELECTOR, 'h2 a')
                link = title_element.get_attribute('href')
                title = title_element.text
                if title and link: 
                    results.append({'title': title, 'link': link})
            except Exception as e:
                logging.error(f"Error processing a result: {e}")
                continue

    except Exception as e:
        logging.error(f"Error during Bing search: {traceback.format_exc()}")
        raise e
    finally:
        driver.quit()

    return results

def generate_response_with_langchain(query, search_results):
    formatted_results = "\n".join([f"{i+1}. {result['title']} - {result['link']}" for i, result in enumerate(search_results[:5])])
    prompt = PromptTemplate(
        input_variables=["query", "results"],
        template="The user asked: {query}\nBased on the following search results:\n{results}\nGenerate a concise and meaningful response."
    )
    llm = Ollama(model="llama2:latest")
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    response = llm_chain.run({"query": query, "results": formatted_results})
    return response
try:
    chroma_client = Client(Settings())
except Exception as e:
    logging.error(f"Error initializing ChromaDB: {traceback.format_exc()}")
    chroma_client = None

try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    logging.error(f"Error loading SentenceTransformer model: {traceback.format_exc()}")
    embedding_model = None

def create_and_store_embeddings(data):
    try:
        embeddings = embedding_model.encode([item['title'] for item in data if item['title']])
        chroma_client.add(
            collection_name="search_results",
            documents=[
                {"id": str(i), "metadata": {"title": item['title'], "link": item['link']}}
                for i, item in enumerate(data)
            ],
            embeddings=embeddings,
        )

        logging.info("Embeddings created and stored successfully.")
    except Exception as e:
        logging.error(f"Error creating/storing embeddings: {traceback.format_exc()}")

def retrieve_similar(query):
    try:
        query_embedding = embedding_model.encode([query])
        results = chroma_client.query(
            collection_name="search_results",
            query_embeddings=query_embedding,
            n_results=5,
        )
        return results
    except Exception as e:
        logging.error(f"Error retrieving similar results: {traceback.format_exc()}")
        return []

st.title("Conversational Chatbot with LangChain and Ollama")
user_query = st.text_input("Ask me anything...")
if user_query:
    st.markdown(f"**You:** {user_query}")
    with st.spinner("thinking🤔..."):
        try:
            search_results = search_bing(user_query)
            if search_results:
                if chroma_client and embedding_model:
                    create_and_store_embeddings(search_results)
                chatbot_response = generate_response_with_langchain(user_query, search_results)
                st.markdown(f"**🤖 Chatbot:**\n{chatbot_response}")
            else:
                st.warning("🤷‍♂️ Sorry, I couldn't find anything.")
        except Exception as e:
            st.error(f"An error occurred: {traceback.format_exc()}")

            st.markdown("---")
            st.markdown("**Developed by Ziaulhaq Parsa**")