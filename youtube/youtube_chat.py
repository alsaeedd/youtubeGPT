#%%
# Setup and Imports
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap
import os

load_dotenv(find_dotenv())
embeddings = HuggingFaceEmbeddings()

def create_db_from_youtube_video_url(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    # Split the transcript into chunks so that we don't exceed the maximum token limit
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query, k=4): # 4 because of Claude-3's max token limit
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatAnthropic(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-3-7-sonnet-20250219",
        temperature=0.2,
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

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    
    # Changed to handle Anthropic's response format
    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs

#%%
# YouTubeGPT:
video_url = "https://youtu.be/lmCaQxk4b8c?si=8Q1e-hpx80sgtcbe"
db = create_db_from_youtube_video_url(video_url)

query = "What is this video about?"
response, docs = get_response_from_query(db, query)
print("\n")
print(textwrap.fill(response, width=50))
# %%