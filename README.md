# YouTubeGPT

Ironically, this repo doesn't actually use GPT. It uses Anthropic's Claude 3.5 Sonnet Model. But the name isn't as appealing🤣.

This repository focuses on experimenting with the LangChain library for building powerful applications with large language models (LLMs).

## LangChain (Quickstart)

This part of the code is to mess around and better understand LangChain. Under the quickstart folder, you will find a python file with code that will explain each of the following modules with an example. Please refer to [this](https://youtu.be/NYSWn1ipbgg?si=4YgkjzfuG3ieG1n4) for an explanation that you can follow along to with the quickstart file.

The Python-specific portion of LangChain's documentation covers several main modules, each providing examples, how-to guides, reference docs, and conceptual guides. These modules include:

1. Models: Various model types and model integrations supported by LangChain.
2. Prompts: Prompt management, optimization, and serialization.
3. Memory: State persistence between chain or agent calls, including a standard memory interface, memory implementations, and examples of chains and agents utilizing memory.
4. Indexes: Combining LLMs with custom text data to enhance their capabilities.
5. Chains: Sequences of calls, either to an LLM or a different utility, with a standard interface, integrations, and end-to-end chain examples.
6. Agents: LLMs that make decisions about actions, observe the results, and repeat the process until completion, with a standard interface, agent selection, and end-to-end agent examples.

## Requirements

- [Python 3.6 or higher](https://www.python.org/downloads/)
- [LangChain library](https://python.langchain.com/en/latest/index.html)
- [Anthropic API key](https://www.anthropic.com/api)

## Anthropic API Models

You will have to pay at least 5 dollars to use their API. It's very cheap per usage and probably really worth it.

## Installation

#### 1. Clone the repository

```bash
git clone https://github.com/alsaeedd/youtubeGPT.git
```

#### 2. Create a Python environment

Python 3.6 or higher using `venv`.

```bash
cd youtubeGPT
python3 -m venv env
source env/bin/activate
```

#### 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

#### 4. Set up the keys in a .env file

First, create a `.env` file in the root directory of the project. Inside the file, add your OpenAI API key:

```makefile
ANTHROPIC_API_KEY="your_api_key_here"
```

Save the file and close it.

##### NOTE: This next part is already in the code, but it's here for you to know how the API key is imported into the code.

In your Python script or Jupyter notebook, load the `.env` file using the following code:

```python
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
```

By using the right naming convention for the environment variable, you don't have to manually store the key in a separate variable and pass it to the function. The library or package that requires the API key will automatically recognize the `ANTHROPIC_API_KEY` environment variable and use its value.

When needed, you can access the `ANTHROPIC_API_KEY` as an environment variable:

```python
import os
api_key = os.environ['ANTHROPIC_API_KEY']
```

Now your the project environment is all set up and ready to go.

#### 5. Install Extensions for a Smooth Experience

In VSCode, install the Jupyter and Python extensions. This way, you will see a "Run Cell" button on top of each of the prompts and pieces of code you want to mess around with.

##### Credits

Credits to Dave Ebbelaar on YouTube for this project.

Please check out his youtube channel [here](https://www.youtube.com/@daveebbelaar)
