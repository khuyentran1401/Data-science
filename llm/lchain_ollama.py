# /// script
# dependencies = [
#     "marimo",
# ]
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.13.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # How to Use LangChain with DeepSeek and Ollama: Step-by-Step Integration Tutorial

    ## What Is LangChain? (Quick Overview)

    [LangChain](https://www.langchain.com/) is an open-source framework designed to make it easier to build applications powered by large language models (LLMs). It allows you to connect different components — like language models, tools, prompts, and memory — into structured, reusable pipelines.

    In this guide, we'll walk you through how to use LangChain in combination with two powerful open-source model platforms:

    - **[Ollama](https://ollama.com/)** – a local LLM runner for models like LLaMA 3, Mistral, and others, perfect for running models on your own machine.
    - **[DeepSeek](https://www.deepseek.com/)** – a family of advanced transformer models known for strong reasoning, coding, and general-purpose performance.

    You’ll learn how to:

    - Integrate **LangChain with Ollama** for local model serving
    - Connect **LangChain to DeepSeek** via API or local inference
    - Optionally combine **LangChain, Ollama, and DeepSeek together** in a single workflow for advanced use cases

    Whether you came here for a **LangChain Ollama tutorial**, a **LangChain DeepSeek tutorial**, or a hybrid integration of all three, this article will guide you step-by-step.

    ## Introduction to Ollama and DeepSeek

    Before diving into integration steps, let's understand the two key technologies we'll be working with in this tutorial.

    ### What is Ollama?

    ![Ollama logo](https://ollama.com/public/blog/embedding-models.png)

    Ollama is an open-source framework designed to run large language models locally on your machine. It provides a simplified interface for downloading, running, and interacting with various open-source LLMs without needing extensive technical setup. Ollama handles the complex infrastructure requirements so developers can focus on using LLMs rather than managing them.

    - Provides a simple CLI and REST API for running models locally
    - Supports popular open-source models like DeepSeek, Llama, Mistral, and Gemma
    - Optimized for consumer hardware with minimal setup requirements
    - Offers customization through Modelfiles for fine-tuning behavior

    With its focus on local execution, Ollama enables privacy-conscious applications and development workflows that don't depend on external APIs. This makes it particularly valuable for scenarios where data sensitivity is a concern or when working in environments with limited internet connectivity.

    ### What is DeepSeek?

    ![DeepSeek AI logo and visualization showing advanced language model capabilities for reasoning and code generation](https://platform.theverge.com/wp-content/uploads/sites/2/chorus/uploads/chorus_asset/file/25848982/STKB320_DEEPSEEK_AI_CVIRGINIA_A.jpg?quality=90&strip=all&crop=0,0,100,100)

    DeepSeek represents a family of transformer-based language models developed with a focus on reasoning capabilities and coding performance. With DeepSeek [R1](https://github.com/deepseek-ai/DeepSeek-R1) and [V3](https://github.com/deepseek-ai/DeepSeek-V3) garnering almost 200k stars on GitHub, DeepSeek AI has established itself as one of the leading open-source model ecosystems in the AI community. The project's popularity stems from its impressive performance across various benchmarks.

    - DeepSeek-R1 models include both base versions and specialized distilled variants
    - Achieves performance comparable to proprietary models on math, code, and reasoning tasks
    - Released under MIT license that supports commercial use and modifications
    - Available through Hugging Face and the DeepSeek platform

    The DeepSeek model family continues to evolve, with ongoing research focused on improving reasoning capabilities through reinforcement learning approaches. By incorporating these models into LangChain workflows, developers can use their strengths in applications requiring complex reasoning or code generation while maintaining the flexibility of the LangChain framework.

    ## LangChain + Ollama: Integration Tutorial

    Now that we understand the core technologies, let's explore how to integrate LangChain with Ollama to run models locally.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Installation and Setup

    First, install the required packages:

    ```python
    pip install langchain langchain-community langchain-ollama
    ```

    Ollama needs to be installed separately since it's a standalone service that runs locally:

    - For macOS: Download from [ollama.com](https://ollama.com) - this installs both the CLI tool and service
    - For Linux: `curl -fsSL https://ollama.com/install.sh | sh` - this script sets up both the binary and system service
    - For Windows: Download Windows (Preview) from [ollama.com](https://ollama.com) - still in preview mode with some limitations

    Ollama runs as a local server process that exposes a REST API on port 11434. This architecture allows any application to connect to it, not just the command line interface. When you start Ollama, it:

    1. Loads the GGUF model files from your local storage
    2. Creates an inference engine using optimized libraries like llama.cpp
    3. Exposes HTTP endpoints that LangChain will communicate with

    Start the Ollama server:

    ```bash
    ollama serve
    ```

    The server will run in the background, handling model loading and inference requests. You can configure Ollama with environment variables:

    ```bash
    # Set custom model directory location
    OLLAMA_MODELS=/path/to/models ollama serve

    # Limit GPU memory usage (in MiB)
    OLLAMA_GPU_LAYERS=35 ollama serve
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Pulling Models with Ollama

    Before using any model with LangChain, you need to pull it to your local machine:

    ```bash
    ollama pull qwen3:0.6b
    ```

    When you run this command, Ollama:

    1. Downloads the model weights (often several GB in size)
    2. Optimizes the model for your specific hardware
    3. Stores the model in your local Ollama library (typically in `~/.ollama/models`)

    Once it is downloaded, you can serve the model with the following command:

    ```bash
    ollama run qwen3:0.6b
    ```

    Popular models with their characteristics:

    - `llama3` - Meta's Llama 3 model (8B parameters, good general purpose)
    - `llama3:70b` - Larger 70B parameter variant with stronger reasoning
    - `mistral` - Mistral AI's 7B parameter base model (efficient for its size)
    - `gemma:7b` - Google's Gemma model optimized for various tasks
    - `deepseek` - DeepSeek's model with strong code generation capabilities
    - `codellama` - Specialized for programming tasks and code completion
    - `nomic-embed-text` - Designed specifically for text embeddings

    The model size has significant impact on performance and resource requirements:

    - Smaller models (7B-8B) run well on most modern computers with 16GB+ RAM
    - Medium models (13B-34B) need more RAM or GPU acceleration
    - Large models (70B+) typically require a dedicated GPU with 24GB+ VRAM

    For a full list of models you can serve locally, check out [the Ollama model library](https://ollama.com/search). Before pulling a model and potentially waste your hardware resources, check out [the VRAM calculator](https://apxml.com/tools/vram-calculator) that tells you if you can run a specific model on your machine:

    ![VRAM Calculator showing memory requirements for different LLM models across various quantization levels](images/vram.png)

    ### Basic Chat Integration - continue from here

    LangChain provides dedicated classes for working with Ollama chat models:

    ```python
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage

    # Initialize the chat model with specific configurations
    chat_model = ChatOllama(
        model="qwen3:0.6b",
        temperature=0.5,
        base_url="http://localhost:11434",  # Can be changed for remote Ollama instances
    )

    # Create a conversation with system and user messages
    messages = [
        SystemMessage(content="You are a helpful coding assistant specialized in Python."),
        HumanMessage(content="Write a recursive Fibonacci function with memoization.")
    ]

    # Invoke the model
    response = chat_model.invoke(messages)
    print(response.content)
    ```

    Make sure you have Ollama running on your machine and the model is already pulled and being served. If the model isn't available locally, Ollama will attempt to download it first, which may take some time depending on your internet connection and the model size.

    Under the hood, `ChatOllama`:

    1. Converts LangChain message objects into Ollama API format
    2. Makes HTTP POST requests to the `/api/chat` endpoint
    3. Processes streaming responses when enabled
    4. Parses the response back into LangChain message objects

    The `ChatOllama` class supports both synchronous and asynchronous operations, so for high-throughput applications, you can use:

    ```python
    async def generate_async():
        response = await chat_model.ainvoke(messages)
        return response.content

    # In async context
    result = await generate_async()
    ```

    ```python
    print( result[:200])
    ```

    ```plaintext
    <think>
    Okay, I need to write a recursive Fibonacci function with memoization. Let me think about how to approach this.

    First, the Fibonacci sequence is defined such that each number is the sum of t
    ```

    ### Using Completion Models

    For traditional completion-style interactions, use the `Ollama` class:

    ```python
    from langchain_ollama import OllamaLLM

    # Initialize the LLM with specific options
    llm = OllamaLLM(
        model="qwen3:0.6b",
    )

    # Generate text from a prompt
    text = \"""
    Write a quick sort algorithm in Python with detailed comments:
    ```python
    def quicksort(
    \"""

    response = llm.invoke(text)
    print(response[:500])
    ```

    ```plaintext
    <think>
    Okay, I need to write a quicksort algorithm in Python with detailed comments. Let me start by recalling how quicksort works. The basic idea is to choose a pivot element, partition the array into elements less than the pivot and greater than it, and then recursively sort each partition. The pivot can be chosen in different ways, like the first element, middle element, or random element.

    First, I should define the function signature. The parameters are the array, and maybe a left and ...
    ```

    The difference between `ChatOllama` and `OllamaLLM` classes:

    - `OllamaLLM` uses the `/api/generate` endpoint for text completion
    - `ChatOllama` uses the `/api/chat` endpoint for chat-style interactions
    - Completion is better for code continuation, creative writing, and single-turn prompts
    - Chat is better for multi-turn conversations and when using system prompts

    For streaming responses (showing tokens as they're generated):

    ```python
    for chunk in llm.stream("Explain quantum computing in three sentences:"):
        print(chunk, end="", flush=True)
    ```

    ```plaintext
    <think>
    Okay, the user wants me to explain quantum computing in three sentences. Let me start by recalling what I know. Quantum computing uses qubits instead of classical bits. So first sentence should mention qubits and the difference from classical bits. Maybe say "Quantum computing uses qubits, which can exist in multiple states at once, unlike classical bits that are either 0 or 1."

    ...
    ```

    ### Customizing Model Parameters

    Ollama offers fine-grained control over generation parameters:

    ```python
    llm = OllamaLLM(
        model="deepseek",
        temperature=0.7,      # Controls randomness (0.0 = deterministic, 1.0 = creative)
        top_p=0.9,            # Nucleus sampling parameter (lower = more focused)
        top_k=40,             # Limits vocabulary to top K tokens
        num_ctx=4096,         # Context window size in tokens
        num_predict=100,      # Maximum number of tokens to generate
        stop=["```", "###"],  # Stop sequences to end generation
        repeat_penalty=1.1,   # Penalizes repetition (>1.0 reduces repetition)
        num_thread=4,         # CPU threads for computation
        num_gpu=1,            # Number of GPUs to use
        mirostat=0,           # Alternative sampling method (0=disabled, 1=v1, 2=v2)
        mirostat_eta=0.1,     # Learning rate for mirostat
        mirostat_tau=5.0,     # Target entropy for mirostat
    )
    ```

    Parameter recommendations:

    - For factual or technical responses: Lower temperature (0.1-0.3) and higher `repeat_penalty` (1.1-1.2)
    - For creative writing: Higher temperature (0.7-0.9) and lower `top_p` (0.8-0.9)
    - For code generation: Medium temperature (0.3-0.6) with specific stop tokens like "```"
    - For long-form content: Increase `num_predict` and use a larger `num_ctx`

    The model behavior changes dramatically with these settings. For example:

    ```python
    # Scientific writing with precise output
    scientific_llm = Ollama(model="deepseek", temperature=0.1, top_p=0.95, repeat_penalty=1.2)

    # Creative storytelling
    creative_llm = Ollama(model="deepseek", temperature=0.9, top_p=0.8, repeat_penalty=1.0)

    # Code generation
    code_llm = Ollama(model="codellama", temperature=0.3, top_p=0.95, stop=["```", "def "])
    ```

    ### Creating LangChain Chains

    LangChain's power comes from composing components into chains:

    ```python
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain.schema import StrOutputParser
    import json

    # Create a structured prompt template
    prompt = PromptTemplate.from_template(\"""
    You are an expert educator.
    Explain the following concept in simple terms that a beginner would understand.
    Make sure to provide:
    1. A clear definition
    2. A real-world analogy
    3. A practical example

    Concept: {concept}
    \""")

    # Create a parser that extracts structured data
    class JsonOutputParser:
        def parse(self, text):
            try:
                # Find JSON blocks in the text
                if "```json" in text and "```" in text.split("```json")[1]:
                    json_str = text.split("```json")[1].split("```")[0].strip()
                    return json.loads(json_str)
                # Try to parse the whole text as JSON
                return json.loads(text)
            except:
                # Fall back to returning the raw text
                return {"raw_output": text}

    # Build a more complex chain
    chain = (
        {"concept": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Execute the chain with detailed tracking
    result = chain.invoke("Recursive neural networks")
    print(result[:500])
    ```

    ```plaintext
    <think>
    Okay, so the user is asking for a simple explanation of recursive neural networks. Let me start by breaking down the concept. First, I need to define it clearly. Recursive neural networks... Hmm, I remember they're a type of neural network that can process data in multiple steps. Wait, maybe I should explain it as networks that can be broken down into smaller parts. Like, they can have multiple layers or multiple levels of processing.

    Now, the user wants a real-world analogy. Let me th...
    ```

    The chain architecture allows you to:

    1. Pre-process inputs before sending to the model
    2. Transform model outputs into structured data
    3. Chain multiple models together
    4. Add memory and context management

    For more advanced use cases, create multi-step reasoning chains:

    ```python
    from langchain.chains import SequentialChain
    from langchain.chains import LLMChain

    # First chain summarizes a concept
    summarize_prompt = PromptTemplate.from_template("Summarize this concept: {concept}")
    summary_chain = LLMChain(llm=llm, prompt=summarize_prompt, output_key="summary")

    # Second chain explains it with examples
    explain_prompt = PromptTemplate.from_template("Explain {summary} with examples")
    explanation_chain = LLMChain(llm=llm, prompt=explain_prompt, output_key="explanation")

    # Connect chains
    full_chain = SequentialChain(
        chains=[summary_chain, explanation_chain],
        input_variables=["concept"],
        output_variables=["summary", "explanation"]
    )

    result = full_chain({"concept": "Vector databases"})
    print(f"Summary: {result['summary']}\n\nExplanation: {result['explanation']}")
    ```

    ### Working with Embeddings

    Embeddings convert text into numerical vectors that capture semantic meaning:

    ```python
    from langchain_ollama import OllamaEmbeddings
    import numpy as np

    # Initialize embeddings model with specific parameters
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",  # Specialized embedding model that is also supported by Ollama
        base_url="http://localhost:11434",
    )

    # Create embeddings for a query
    query = "How do neural networks learn?"
    query_embedding = embeddings.embed_query(query)
    print(f"Embedding dimension: {len(query_embedding)}")
    ```

    ```plaintext
    Embedding dimension: 768
    ```

    ```python
    # Create embeddings for multiple documents
    documents = [
        "Neural networks learn through backpropagation",
        "Transformers use attention mechanisms",
        "LLMs are trained on text data"
    ]

    doc_embeddings = embeddings.embed_documents(documents)

    # Calculate similarity between vectors
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # Find most similar document to query
    similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]
    most_similar_idx = np.argmax(similarities)
    print(f"Most similar document: {documents[most_similar_idx]}")
    ```

    ```plaintext
    Most similar document: Neural networks learn through backpropagation
    ```

    These embeddings can be used for:

    1. **Semantic search**: Find documents related to a query by meaning, not just keywords
    2. **Document clustering**: Group similar documents together
    3. **Retrieval-Augmented Generation (RAG)**: Retrieve relevant context before generating responses
    4. **Recommendation systems**: Suggest similar items based on embeddings

    When working with Ollama embeddings, it's important to understand:

    - Different models produce embeddings with different dimensions (384, 768, 1024, etc.)
    - The quality of embeddings varies by model (specialized embedding models usually perform better)
    - Embedding generation is usually faster than text generation

    ### Common Issues and Troubleshooting

    1. **Ollama Server Not Running**
       - Error: `Failed to connect to Ollama server at http://localhost:11434/api/generate`
       - Diagnosis: Check if the server is running with `ps aux | grep ollama`
       - Solution: Start Ollama with `ollama serve` and ensure no firewall is blocking port 11434

    2. **Model Not Found**
       - Error: `no model found with name 'xyz'`
       - Diagnosis: List available models with `ollama list`
       - Solution: Pull the model first with `ollama pull xyz`
       - Check model naming: Ollama is case-sensitive, so 'Llama' and 'llama' are different

    3. **Out of Memory**
       - Error: `Failed to load model: out of memory`
       - Diagnosis: Check available system memory with `free -h` (Linux) or Activity Monitor (Mac)
       - Solution:
         - Try a smaller model (`llama3:8b` instead of `llama3:70b`)
         - Reduce `num_ctx` parameter to use less memory
         - Set `OLLAMA_GPU_LAYERS=0` to run only on CPU if GPU memory is insufficient
         - Close other memory-intensive applications

    4. **Slow Responses**
       - Issue: First response after starting Ollama takes a long time
       - Diagnosis: This is expected behavior as the model is loaded from disk into RAM/VRAM
       - Solutions:
         - For frequent use, keep Ollama running in the background
         - Use `OLLAMA_KEEP_ALIVE=1h` to keep models in memory for one hour after last use
         - For faster startup, use smaller models or quantized versions (e.g., `deepseek:7b-q4_0`)

    5. **Port Already in Use**
       - Error: `listen tcp 0.0.0.0:11434: bind: address already in use`
       - Diagnosis: Find what's using the port with `lsof -i :11434` (Mac/Linux) or `netstat -ano | findstr 11434` (Windows)
       - Solutions:
         - Stop the existing Ollama process
         - Use a different port with `OLLAMA_HOST=127.0.0.1:11435 ollama serve`
         - Update your LangChain code to use the new port: `base_url="http://localhost:11435"`

    6. **Incorrect Response Formats**
       - Issue: Model returns text when JSON was expected or vice versa
       - Diagnosis: Check if your model supports the requested format and if your prompt clearly specifies the format
       - Solution:
         - Set `format="json"` in ChatOllama constructor if the model supports it
         - Add explicit format instructions in your prompt
         - Use output parsers to structure the response regardless of format

    7. **Token Context Length Exceeded**
       - Error: `context window is full` or truncated responses
       - Diagnosis: Your input + generated text is exceeding the model's context window
       - Solutions:
         - Reduce input prompt length
         - Use a model with larger context window
         - Split long documents into chunks
         - Set appropriate `num_ctx` parameter (but requires more memory)

    By following this comprehensive tutorial, you now have the knowledge to use Ollama's local inference capabilities within LangChain's flexible framework for building sophisticated LLM-powered applications.

    ## LangChain + DeepSeek: Integration Tutorial

    Now that we've explored how to use LangChain with Ollama for local model serving, let's turn our attention to DeepSeek. While Ollama gives us local model running capabilities, integrating with DeepSeek offers access to its specialized models known for reasoning capabilities and coding performance.

    ### Installation and Setup For Deepseek

    First, install the required packages:

    ```python
    pip install langchain langchain-deepseek python-dotenv
    ```

    DeepSeek is primarily accessed through their API. You'll need to create a [DeepSeek account](https://platform.deepseek.com), generate an API key, and add a couple of dollars as balance. Once you have your API key, set it as an environment variable in a `.env` file:

    ```python
    touch .env

    # In the .env file
    DEEPSEEK_API_KEY="sk-your-api-key-here"
    ```

    ### Using DeepSeek Chat Models

    DeepSeek offers two main model families:

    - **DeepSeek-V3** (specified via `model="deepseek-chat"`) - general purpose model with tool calling and structured output
    - **DeepSeek-R1** (specified via `model="deepseek-reasoner"`) - specialized for reasoning tasks

    Here's how to use them with LangChain:

    ```python
    from dotenv import load_dotenv
    from langchain_deepseek import ChatDeepSeek

    load_dotenv()  # Load your api key

    # Initialize the chat model
    llm = ChatDeepSeek(
        model="deepseek-chat",  # Can also use "deepseek-reasoner"
        temperature=0,  # 0 for more deterministic responses
        max_tokens=None,  # None means model default
        timeout=None,  # API request timeout
        max_retries=2  # Retry failed requests
    )

    # Create a conversation with system and user messages
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="Translate this sentence: I love programming.")
    ]

    # Generate a response
    response = llm.invoke(messages)
    print(response.content)
    ```

    For asynchronous operation, which is useful for handling multiple requests:

    ```python
    async def generate_async():
        response = await llm.ainvoke(messages)
        return response.content

    # In async context
    result = await generate_async()
    ```

    ### Building Chains with DeepSeek

    LangChain's power comes from composing components into chains. Here's how to create a translation chain with DeepSeek:

    ```python
    from langchain_core.prompts import ChatPromptTemplate

    # Create a structured prompt template
    prompt = ChatPromptTemplate(
        [
            (
                "system",
                "You are a helpful assistant that translates {input_language} to {output_language}."
            ),
            ("human", "{input}"),
        ]
    )

    # Build the chain
    chain = prompt | llm

    # Execute the chain
    result = chain.invoke({
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming."
    })

    print(result.content)
    ```

    ### Streaming Responses

    For long-running generations, you might want to stream tokens as they're generated:

    ```python
    from langchain_core.output_parsers import StrOutputParser

    streamed_chain = prompt | llm | StrOutputParser()

    for chunk in streamed_chain.stream({
        "input_language": "English",
        "output_language": "Italian",
        "input": "Machine learning is transforming the world."
    }):
        print(chunk, end="", flush=True)
    ```

    ### Structured Output

    When you need structured data instead of free text, you can use DeepSeek's structured output capability:

    ```python
    from langchain_core.pydantic_v1 import BaseModel, Field
    from typing import List

    # Define the output schema
    class MovieReview(BaseModel):
        title: str = Field(description="The title of the movie")
        year: int = Field(description="The year the movie was released")
        rating: float = Field(description="Rating from 0-10")
        pros: List[str] = Field(description="List of positive aspects")
        cons: List[str] = Field(description="List of negative aspects")

    # Create a structured LLM
    from langchain.output_parsers.openai_functions import PydanticOutputFunctionsParser

    structured_llm = llm.bind(
        functions=[MovieReview],
        function_call={"name": "MovieReview"}
    )

    # Create a chain
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a movie critic."),
            ("human", "Write a review for {movie_title}")
        ])
        | structured_llm
        | PydanticOutputFunctionsParser(pydantic_schema=MovieReview)
    )

    # Get structured output
    review = chain.invoke({"movie_title": "The Matrix"})
    print(f"Title: {review.title}")
    print(f"Year: {review.year}")
    print(f"Rating: {review.rating}/10")
    print("Pros:")
    for pro in review.pros:
        print(f"- {pro}")
    print("Cons:")
    for con in review.cons:
        print(f"- {con}")
    ```

    ### Common Issues and Troubleshooting With DeepSeek

    1. **API Rate Limits**
       - Error: `429 Too Many Requests`
       - Solution: Implement rate limiting in your code or upgrade your API plan

    2. **Authentication Errors**
       - Error: `401 Unauthorized`
       - Solution: Check your API key is correct and properly set in the environment

    3. **Context Length Exceeded**
       - Error: `400 Bad Request: Maximum context length exceeded`
       - Solution: Reduce your input length or try a model with a larger context window

    4. **Model Loading Time with Local Inference**
       - Issue: First request takes a long time
       - Solution: This is normal as the model is loaded into memory; subsequent requests will be faster

    5. **Memory Issues with Local Models**
       - Error: `out of memory`
       - Solution: Use a smaller model, reduce batch size, or upgrade your hardware

    6. **Structured Output Format Errors**
       - Issue: Model returns free text instead of structured output
       - Solution: Use more explicit instructions in your prompt or try reducing temperature

    By following this guide, you now have the knowledge to use DeepSeek's powerful models within the LangChain framework, giving you access to state-of-the-art language capabilities for your applications.

    ## Combining Ollama and DeepSeek in LangChain

    Having explored both Ollama and DeepSeek individually, let's examine how to combine these powerful tools in a single LangChain workflow. One of the most practical setups is running DeepSeek locally through Ollama while maintaining the flexibility to call DeepSeek's API when needed.

    ### Running DeepSeek Locally with Ollama

    Ollama makes it simple to run DeepSeek models on your local machine:

    ```bash
    # Pull the DeepSeek model to your local machine
    ollama pull deepseek:7b
    ```

    Once downloaded, you can access it through LangChain just like any other Ollama model:

    ```python
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage

    # Initialize the local DeepSeek model
    local_deepseek = ChatOllama(
        model="deepseek:7b",
        temperature=0.7,
        base_url="http://localhost:11434"
    )

    # Use it just like any other LangChain model
    response = local_deepseek.invoke([
        HumanMessage(content="Write a recursive function to calculate Fibonacci numbers.")
    ])
    print(response.content)
    ```

    ### Why Use Both?

    Combining Ollama's local DeepSeek deployment with the official DeepSeek API gives you several advantages:

    ```python
    from langchain_deepseek import ChatDeepSeek
    from langchain_ollama import ChatOllama

    # Initialize both local and API models
    local_model = ChatOllama(model="deepseek:7b")
    cloud_model = ChatDeepSeek(model="deepseek-chat")
    ```

    - **Complementary strengths**: Ollama offers speed and privacy for local inference, while DeepSeek's API provides access to their latest models with maximum accuracy and coding capabilities.

    - **Fallback architecture**: You can implement a fallback system that tries the local model first and defaults to the API if needed:

    ```python
    def get_response(query, use_local=True):
        try:
            if use_local:
                return local_model.invoke(query)
            else:
                return cloud_model.invoke(query)
        except Exception as e:
            print(f"Error with {'local' if use_local else 'cloud'} model: {e}")
            # Try the other model if the first one fails
            return cloud_model.invoke(query) if use_local else local_model.invoke(query)
    ```

    - **Cost optimization**: Use local inference for development and testing, then switch to API calls for production or when higher quality is required.

    ### Performance Tips

    To get the most out of your hybrid Ollama-DeepSeek setup:

    - **Prompt tuning per model**: Different models respond better to different prompt formats. Create model-specific prompt templates:

    ```python
    from langchain_core.prompts import ChatPromptTemplate

    # DeepSeek API model prompt template
    api_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are DeepSeek, an advanced AI assistant with strong reasoning abilities."),
        ("human", "{input}")
    ])

    # Local Ollama model prompt template
    local_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful coding assistant that gives concise answers."),
        ("human", "{input}")
    ])

    # Create model-specific chains
    api_chain = api_prompt | cloud_model
    local_chain = local_prompt | local_model
    ```

    - **Parallel execution tradeoffs**: You can run both models in parallel for critical tasks but be mindful of resource usage:

    ```python
    import asyncio

    async def get_both_responses(query):
        # Run both models in parallel
        local_task = local_model.ainvoke(query)
        cloud_task = cloud_model.ainvoke(query)

        # Wait for both to complete
        local_response, cloud_response = await asyncio.gather(local_task, cloud_task)

        # Compare or combine results
        return {
            "local": local_response.content,
            "cloud": cloud_response.content
        }
    ```

    - **Caching responses**: Implement caching to avoid redundant computations and reduce API costs:

    ```python
    from langchain.cache import InMemoryCache
    from langchain.globals import set_llm_cache

    # Set up caching
    set_llm_cache(InMemoryCache())

    # Now repeat queries will use cached responses
    # This works for both local and API models
    response1 = cloud_model.invoke("What is quantum computing?")
    response2 = cloud_model.invoke("What is quantum computing?")  # Uses cache
    ```

    By thoughtfully combining Ollama and DeepSeek, you can create a flexible system that balances speed, cost, privacy, and quality based on your specific requirements. This hybrid approach gives you the best of both worlds while maintaining the composability and structure that LangChain provides.

    ## Final Thoughts And Next Steps

    In this tutorial, we've covered how to use LangChain with both Ollama and DeepSeek, giving you the tools to build powerful AI applications. You can now run models locally with Ollama for privacy and speed, access DeepSeek's advanced models through their API, or combine both approaches in a flexible workflow that meets your specific needs.

    For your next steps, we recommend diving into the official documentation for [LangChain](https://www.langchain.com/), [Ollama](https://ollama.com), and [DeepSeek](https://www.deepseek.com/). Try building a simple project that chains these components together, such as a coding assistant that uses local models for rapid prototyping and API models for complex problems. As you gain comfort with these tools, you'll be able to create more advanced applications specific to your use cases.
    """
    )
    return


if __name__ == "__main__":
    app.run()
