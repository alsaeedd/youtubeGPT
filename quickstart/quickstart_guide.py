#%%
# Setup and Imports
from dotenv import find_dotenv, load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools, get_all_tool_names
import os

# Load environment variables
load_dotenv(find_dotenv())

# Initialize LLM
llm = ChatAnthropic(
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    model="claude-3-sonnet-20240229"
)

#%%
# Basic LLM Example
prompt = "Write a poem about Bahrain and ai"
response = llm.invoke(prompt)
print(response.content)
#%%
# Prompt Templates Example
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

result = prompt.format(product="Smart Apps using Large Language Models (LLMs)")
print(result)

#%%
# Chains Example 1
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run("AI Chatbots for Dental Offices"))

#%%
# Chains Example 2
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write an email subject for this topic {topic}?",
)

chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run("AI Session"))

#%%
# Agents Example
tools = load_tools(["wikipedia", "llm-math"], llm=llm)

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

result = agent.run(
    "In what year was python released and who is the original creator? Multiply the year by 3"
)
print(result)

#%%
# Memory Example
# Allows the agent to remember information from previous interactions
conversation = ConversationChain(llm=llm, verbose=True)

output = conversation.predict(input="Hi there!")
print(output)

output = conversation.predict(
    input="I'm doing well! Just having a conversation with an AI."
)
print(output)