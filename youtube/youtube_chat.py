#%%
# Setup and Imports
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser  # Added for modern chain construction
from dotenv import find_dotenv, load_dotenv
import textwrap
import os

load_dotenv(find_dotenv())
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  # Explicit model name

def create_db_from_youtube_video_url(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    # Split the transcript into chunks so that we don't exceed the maximum token limit
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query, k=8):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    # Modern ChatAnthropic initialization with extended thinking
    chat = ChatAnthropic(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-3-7-sonnet-latest",
        max_tokens=8000,
        temperature=1,
        thinking={"type": "enabled", "budget_tokens": 2000},
    )

    template = """
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript: {docs}
        
        Only use the factual information from the transcript to answer the question, but you are obviously allowed to follow instructions you are given.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # Modern chain construction using pipe syntax
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    
    chain = chat_prompt | chat | StrOutputParser()
    
    # Modern invoke method instead of run
    response = chain.invoke({"question": query, "docs": docs_page_content})
    response = response.replace("\n", "")
    return response, docs

#%%
# YouTubeGPT:
video_url = "https://youtu.be/lmCaQxk4b8c?si=8Q1e-hpx80sgtcbe"
db = create_db_from_youtube_video_url(video_url)

query = "What are the 5 habits mentioned in this video?"
response, docs = get_response_from_query(db, query)
print("\n")
print(textwrap.fill(response, width=50))
# %%