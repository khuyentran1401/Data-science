import marimo

__generated_with = "0.13.7"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # LangChain with Ollama: A Step-by-Step Integration Tutorial
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Why Local AI Matters

        AI models are changing data science projects by automating feature engineering, summarizing datasets, generating reports, and even writing code to examine or clean data.

        However, using popular APIs like OpenAI or Anthropic can introduce serious privacy risks, especially when handling regulated data such as medical records, legal documents, or internal company knowledge. These services transmit user inputs to remote servers, making it difficult to guarantee confidentiality or data residency compliance.

        When data privacy is important, running models locally ensures full control. Nothing leaves your machine, so you manage all inputs, outputs, and processing securely.

        That's where [LangChain](https://www.langchain.com/) and [Ollama](https://ollama.com/) come in. LangChain provides the framework to build AI applications. Ollama lets you run open-source models locally. This guide shows you how to combine both tools to create privacy-preserving AI workflows that process sensitive data exclusively on your own machine.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Introduction to Ollama and LangChain

        Before diving into integration steps, let's understand both tools.

        ### What is Ollama?

        ![Ollama integration diagram showing how it connects with various AI libraries and frameworks](https://ollama.com/public/blog/libraries.svg)

        Ollama is an open-source tool that makes it easy to run large language models locally. It offers a simple CLI and REST API for downloading and interacting with popular models like Llama, Mistral, DeepSeek, and Gemma—no complex setup required.

        Since Ollama doesn't depend on external APIs, it is ideal for sensitive data or limited-connectivity environments.

        ### What is LangChain?

        LangChain is a framework for creating AI applications using language models. It offers a component-based structure that helps developers build AI workflows by connecting various operations.

        The framework helps with common AI development tasks:

        - **Prompt management**: Templates and tools for consistent model interactions
        - **Memory systems**: Maintaining context across conversations
        - **Output parsing**: Converting model responses into structured data
        - **Model integration**: Common interfaces for different AI models
        - **Chain composition**: Connecting multiple steps into complete workflows

        Data scientists use LangChain to move AI projects from research to production. Rather than writing custom code for model interactions, response handling, and error management, you can use LangChain's ready-made components to build applications.

        > Components are reusable code modules that perform specific functions like processing data, calling models, or parsing responses.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## LangChain + Ollama: Integration Tutorial

        Now that we understand the core technology, let's see how to integrate LangChain with Ollama to run models locally.

        ### Installation and Setup

        Getting started with local AI requires setting up both LangChain and Ollama, but they have different installation processes since LangChain is a Python library while Ollama runs as a system service.

        First, install the required packages:
        """
    )
    return


app._unparsable_cell(
    r"""
    !pip install langchain langchain-community langchain-ollama
    """,
    name="_",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Ollama needs to be installed separately since it's a standalone service that runs locally:

        - For macOS: Download from [ollama.com](https://ollama.com) - this installs both the CLI tool and service
        - For Linux: `curl -fsSL https://ollama.com/install.sh | sh` - this script sets up both the binary and system service
        - For Windows: Download Windows (Preview) from [ollama.com](https://ollama.com) - still in preview mode with some limitations

        Start the Ollama server:

        ```bash
        ollama serve
        ```

        The server will run in the background, handling model loading and inference requests.

        ### Pulling Models with Ollama

        With Ollama installed and running, you need to download models to your local machine before LangChain can use them. Unlike cloud APIs where models are immediately available, local deployment requires pulling model weights first.

        Before using any model with LangChain, you need to pull it to your local machine:

        ```bash
        ollama pull qwen3:0.6b
        ```

        When you run this command, Ollama:

        1. Downloads the model weights (often several GB in size)
        2. Optimizes the model for your specific hardware

        Once it is downloaded, you can serve the model with the following command:

        ```bash
        ollama run qwen3:0.6b
        ```

        The model size has a large impact on performance and resource requirements:

        - Smaller models (7B-8B) run well on most modern computers with 16GB+ RAM
        - Medium models (13B-34B) need more RAM or GPU acceleration
        - Large models (70B+) typically require a dedicated GPU with 24GB+ VRAM

        For a full list of models you can serve locally, check out [the Ollama model library](https://ollama.com/search). Before pulling a model and potentially waste your hardware resources, check out [the VRAM calculator](https://apxml.com/tools/vram-calculator) that tells you if you can run a specific model on your machine:

        ![VRAM Calculator showing memory requirements for different LLM models across various quantization levels](images/vram.png)

        ### Basic Chat Integration

        Once you have a model downloaded, use dedicated classes in LangChain to handle the communication between your Python code and the Ollama service:
        """
    )
    return


@app.cell
def _():
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama

    # Initialize the chat model with specific configurations
    chat_model = ChatOllama(
        model="qwen3:0.6b",
        temperature=0.5,
        base_url="http://localhost:11434",  # Can be changed for remote Ollama instances
    )

    # Create a conversation with system and user messages
    messages = [
        SystemMessage(
            content="You are a helpful coding assistant specialized in Python."
        ),
        HumanMessage(content="Write a recursive Fibonacci function with memoization."),
    ]

    # Invoke the model
    response = chat_model.invoke(messages)
    print(response.content)
    return chat_model, messages


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This snippet initializes a `ChatOllama` instance to interact with a local AI model, such as `qwen3:0.6b`. It configures model parameters like creativity (`temperature`) and the Ollama server URL. A conversation is then constructed with a system message to define the AI's role (e.g., a Python assistant) and a human message containing the user's specific query (e.g., requesting a Fibonacci function). Finally, the script sends this structured conversation to the model and prints the content of the response it receives.

        Under the hood, `ChatOllama`:

        1. Converts LangChain message objects into Ollama API format
        2. Makes HTTP POST requests to the `/api/chat` endpoint
        3. Processes streaming responses when activated
        4. Parses the response back into LangChain message objects

        The `ChatOllama` class also supports asynchronous operations, allowing data scientists to run multiple model calls in parallel—ideal for building responsive, non-blocking applications like dashboards or chat interfaces:
        """
    )
    return


@app.cell
async def _(chat_model, messages):
    async def generate_async():
        response = await chat_model.ainvoke(messages)
        return response.content

    # In async context
    result = await generate_async()

    print(result[:200])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Using Completion Models

        While chat models work well for conversational interactions, some data science tasks such as code generation, document completion, and creative writing often work better with traditional text completion.

        For such tasks, you can use the `OllamaLLM` class. These prompts are useful when you want the model to predict the continuation of a given input, such as writing code, completing documentation, or expanding a paragraph:
        """
    )
    return


@app.cell
def _():
    from langchain_ollama import OllamaLLM

    llm = OllamaLLM(model="qwen3:0.6b")
    text = "\nWrite a quick sort algorithm in Python with detailed comments:\n```python\ndef quicksort(\n"
    response_1 = llm.invoke(text)
    print(response_1[:500])
    return OllamaLLM, llm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The difference between `ChatOllama` and `OllamaLLM` classes:

        - `OllamaLLM` uses the `/api/generate` endpoint for text completion
        - `ChatOllama` uses the `/api/chat` endpoint for chat-style interactions
        - Completion is better for code continuation, creative writing, and single-turn prompts
        - Chat is better for multi-turn conversations and when using system prompts

        For streaming responses (showing tokens as they're generated):
        """
    )
    return


@app.cell
def _(llm):
    for chunk in llm.stream("Explain quantum computing in three sentences:"):
        print(chunk, end="", flush=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Use streaming responses to display output in real time, making interactive apps like chatbots feel faster and more responsive.

        ### Customizing Model Parameters

        Both completion and chat models use default settings that work reasonably well, but data science tasks often require specific behavior. Scientific analysis needs precise, factual responses while creative tasks benefit from more randomness and variety.

        Ollama offers fine-grained control over generation parameters:
        """
    )
    return


@app.cell
def _(OllamaLLM):
    llm_1 = OllamaLLM(
        model="qwen3:0.6b", temperature=0.7, stop=["```", "###"], repeat_penalty=1.1
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Details about these parameters:

        - `model`: Specifies the language model to use.
        - `temperature`: Controls randomness; lower = more focused, higher = more creative.
        - `stop`: Defines stop sequences that terminate generation early. Once one of these sequences is produced, the model stops generating further tokens.
        - `repeat_penalty`: Penalizes repeated tokens to reduce redundancy. Values greater than 1.0 discourage the model from repeating itself.

        Parameter recommendations:

        - For factual or technical responses: Lower `temperature` (0.1-0.3) and higher `repeat_penalty` (1.1-1.2)
        - For creative writing: Higher `temperature` (0.7-0.9)
        - For code generation: Medium `temperature` (0.3-0.6) with specific `stop` like "\`\`\`"

        The model behavior changes dramatically with these settings. For example:
        """
    )
    return


@app.cell
def _(OllamaLLM):
    # Scientific writing with precise output
    scientific_llm = OllamaLLM(model="qwen3:0.6b", temperature=0.1, repeat_penalty=1.2)

    # Creative storytelling
    creative_llm = OllamaLLM(model="qwen3:0.6b", temperature=0.9, repeat_penalty=1.0)

    # Code generation
    code_llm = OllamaLLM(model="codellama", temperature=0.3, stop=["```", "def "])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The `repeat_penalty` parameter works by tracking recently generated tokens and reducing their probability of appearing again. When set to 1.1, tokens that appeared in the last few generated words become 10% less likely to be selected. This helps prevent the model from getting stuck in repetitive loops while maintaining natural language flow.

        ### Creating LangChain Chains

        Customizing individual model parameters helps with specific tasks, but AI  workflows often involve multiple steps: data validation, prompt formatting, model inference, and output processing. Running these steps manually for each request becomes repetitive and error-prone. This is where LangChain truly shines - by allowing you to compose these components into chains that automate your entire workflow.

        LangChain's power comes from composing components into chains, which connect different operations in sequence to create end-to-end applications:
        """
    )
    return


@app.cell
def _():
    import json

    from langchain_core.prompts import PromptTemplate

    # Create a structured prompt template
    prompt = PromptTemplate.from_template(
        """
    You are an expert educator.
    Explain the following concept in simple terms that a beginner would understand.
    Make sure to provide:
    1. A clear definition
    2. A real-world analogy
    3. A practical example

    Concept: {concept}
    """
    )
    return PromptTemplate, json


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        First, we import the required LangChain components and create a prompt template. The `PromptTemplate.from_template()` method creates a reusable template with placeholder variables (like `{concept}`) that get filled in at runtime.
        """
    )
    return


@app.cell
def _(OllamaLLM, json):
    class JsonOutputParser:

        def parse(self, text):
            try:
                if "```json" in text and "```" in text.split("```json")[1]:
                    json_str = text.split("```json")[1].split("```")[0].strip()
                    return json.loads(json_str)
                return json.loads(text)
            except Exception as e:
                return {"raw_output": text}

    llm_2 = OllamaLLM(model="qwen3:0.6b")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Next, we define a custom output parser. This class attempts to extract JSON from the model's response, handling both code-block format and raw JSON. If parsing fails, it returns the original text wrapped in a dictionary.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Finally, we build the chain using LangChain's pipe operator (`|`). The `RunnablePassthrough()` passes input directly to the prompt template, which formats it and sends it to the LLM. The `StrOutputParser()` converts the response to a string. Here is the output:

        ```plaintext
        <think>
        Okay, so the user is asking for a simple explanation of recursive neural networks. Let me start by breaking down the concept. First, I need to define it clearly. Recursive neural networks... Hmm, I remember they\'re a type of neural network that can process data in multiple steps. Wait, maybe I should explain it as networks that can be broken down into smaller parts. Like, they can have multiple layers or multiple levels of processing.

        Now, the user wants a real-world analogy. Let me th...
        ```

        The chain architecture allows you to:

        1. Pre-process inputs before sending to the model
        2. Change model outputs into structured data
        3. Chain multiple models together
        4. Add memory and context management

        ### Working with Embeddings

        Embeddings change text into numerical vectors that capture semantic meaning, allowing computers to understand relationships between words and documents mathematically. Ollama supports specialized embedding models that excel at this conversion.

        First, let's set up the embedding model and understand what we're working with:

        """
    )
    return


@app.cell
def _():
    import numpy as np
    from langchain_ollama import OllamaEmbeddings

    # Initialize embeddings model with specific parameters
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",  # Specialized embedding model that is also supported by Ollama
        base_url="http://localhost:11434",
    )
    return embeddings, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The `nomic-embed-text` model is designed specifically for creating high-quality text embeddings. Unlike general language models that generate text, embedding models focus solely on converting text into meaningful vector representations.

        Now let's create an embedding for a sample query and examine its properties:

        """
    )
    return


@app.cell
def _(embeddings):
    # Create embeddings for a query
    query = "How do neural networks learn?"
    query_embedding = embeddings.embed_query(query)
    print(f"Embedding dimension: {len(query_embedding)}")
    return (query_embedding,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The 768-dimensional vector represents our query in mathematical space. Each dimension captures different semantic features - some might relate to technical concepts, others to question patterns, and so on. Words with similar meanings will have vectors that point in similar directions.

        Next, we'll create embeddings for multiple documents to demonstrate similarity matching:
        """
    )
    return


@app.cell
def _(embeddings):
    # Create embeddings for multiple documents
    documents = [
        "Neural networks learn through backpropagation",
        "Transformers use attention mechanisms",
        "LLMs are trained on text data",
    ]

    doc_embeddings = embeddings.embed_documents(documents)
    return doc_embeddings, documents


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The `embed_documents()` method processes multiple texts at once, which is more efficient than calling `embed_query()` repeatedly. This batch processing saves time when working with large document collections.

        To find which document best matches our query, we need to measure similarity between vectors. Cosine similarity is the standard approach:
        """
    )
    return


@app.cell
def _(doc_embeddings, documents, np, query_embedding):
    # Calculate similarity between vectors
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # Find most similar document to query
    similarities = [
        cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings
    ]
    most_similar_idx = np.argmax(similarities)

    print(f"Most similar document: {documents[most_similar_idx]}")
    print(f"Similarity score: {similarities[most_similar_idx]:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Cosine similarity returns values between -1 and 1, where 1 means identical meaning, 0 means unrelated, and -1 means opposite meanings. Our score of 0.847 indicates strong semantic similarity between the query about neural network learning and the document about backpropagation.

        These embeddings support several data science applications:

        1. **Semantic search**: Find documents by meaning rather than exact keyword matches
        2. **Document clustering**: Group related research papers, reports, or code documentation
        3. **Retrieval-Augmented Generation (RAG)**: Retrieve relevant context before generating responses
        4. **Anomaly detection**: Identify unusual or outlier documents in large collections
        5. **Content recommendation**: Suggest similar articles, datasets, or code examples

        When choosing embedding models for your projects, consider these factors:

        - **Dimension size**: Larger dimensions (1024+) capture more nuance but require more storage and computation
        - **Domain specialization**: Some models work better for scientific text, others for general content
        - **Processing speed**: Smaller models like `nomic-embed-text` balance quality with performance
        - **Language support**: Multilingual models handle multiple languages but may sacrifice quality for any single language

        The quality of your embeddings directly impacts downstream tasks like search relevance and clustering accuracy. Always test different models with your specific data to find the best fit.

        While semantic search with embeddings solves document retrieval, data scientists often need to go one step further. Finding relevant documents is useful, but answering specific questions about your research, datasets, or methodologies requires combining retrieval with natural language generation.

        ### Building a question-answering system for your data

        Data scientists work with extensive collections of research papers, project documentation, and dataset descriptions. When stakeholders ask questions like "What preprocessing steps were used in the customer churn analysis?" or "Which machine learning models performed best for fraud detection?", manual document search becomes time-consuming and error-prone.

        Standard language models can't answer these domain-specific questions because they lack access to your particular data and documentation. You need a system that searches your documents and generates accurate, source-backed answers.

        Retrieval-Augmented Generation (RAG) solves this problem by combining the semantic search capabilities we just built with text generation. RAG retrieves relevant information from your documents and uses it to answer questions with proper attribution.

        Here's how to build a RAG system using the embeddings and chat models we've already configured:
        """
    )
    return


@app.cell
def _(ChatOllama_1, OllamaEmbeddings_1):
    from langchain_core.documents import Document

    embeddings_1 = OllamaEmbeddings_1(model="nomic-embed-text")
    chat_model_1 = ChatOllama_1(model="qwen3:0.6b", temperature=0.3)
    documents_1 = [
        Document(
            page_content="Python is a high-level programming language known for its simplicity and readability."
        ),
        Document(
            page_content="Machine learning algorithms can automatically learn patterns from data without explicit programming."
        ),
        Document(
            page_content="Data preprocessing involves cleaning, changing, and organizing raw data for analysis."
        ),
        Document(
            page_content="Neural networks are computational models inspired by biological brain networks."
        ),
    ]
    doc_embeddings_1 = embeddings_1.embed_documents(
        [doc.page_content for doc in documents_1]
    )
    return chat_model_1, doc_embeddings_1, documents_1, embeddings_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This setup creates a searchable knowledge base from your documents. In production systems, these documents would contain sections from research papers, methodology descriptions, data analysis reports, or code documentation. The embeddings convert each document into vectors that support semantic search.

        """
    )
    return


@app.cell
def _(PromptTemplate, doc_embeddings_1, documents_1, embeddings_1, np):
    def similarity_search(query, top_k=2):
        """Find the most relevant documents for a query"""
        query_embedding = embeddings_1.embed_query(query)
        similarities = []
        for doc_emb in doc_embeddings_1:
            similarity = np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
            )
            similarities.append(similarity)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [documents_1[i] for i in top_indices]

    rag_prompt = PromptTemplate.from_template(
        "\nUse the following context to answer the question. If the answer isn't in the context, say so.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:\n"
    )
    return rag_prompt, similarity_search


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The `similarity_search` function finds documents most relevant to a question using the embeddings we created earlier. The prompt template structures how we present retrieved context to the language model, instructing it to base answers on the provided documents rather than general knowledge.
        """
    )
    return


@app.cell
def _(chat_model_1, rag_prompt, similarity_search):
    def answer_question(question):
        """Generate an answer using retrieved context"""
        relevant_docs = similarity_search(question, top_k=2)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        prompt_text = rag_prompt.format(context=context, question=question)
        response = chat_model_1.invoke([{"role": "user", "content": prompt_text}])
        return (response.content, relevant_docs)

    question = "What makes Python popular for data science?"
    answer, sources = answer_question(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Sources: {[doc.page_content[:50] + '...' for doc in sources]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        The complete RAG system retrieves relevant documents, presents them as context to the language model, and generates answers based on that specific information. This approach grounds responses in your actual documentation rather than the model's general training data.

        RAG systems address several common data science workflow challenges:

        - **Project handoffs**: New team members can query past work to understand methodologies and results
        - **Literature review**: Researchers can search large paper collections for relevant techniques and findings
        - **Data documentation**: Teams can build searchable knowledge bases about datasets, features, and processing steps
        - **Reproducibility**: Stakeholders can find detailed information about how analyses were conducted

        The RAG approach combines semantic search precision with natural language generation fluency. Instead of manually searching through documents or receiving generic answers from language models, you get accurate responses backed by specific sources from your knowledge base.

        This implementation provided a simple introduction to RAG systems, but production-level solutions require additional engineering work. Real-world RAG applications need features like metadata filtering, hybrid search (combining semantic and keyword search), chunking strategies, context window management, and evaluation frameworks to measure relevance and accuracy. Building a solid RAG system also involves addressing challenges like hallucinations, retrieval improvements, and handling dynamically changing knowledge bases.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Conclusion

        This tutorial demonstrated how to integrate LangChain with Ollama for local LLM execution. You learned to set up Ollama, download models, and use `ChatOllama` and `OllamaLLM` for various tasks. We also covered customizing model parameters, building LangChain chains, and working with embeddings. By running models locally, you maintain data privacy and control, which is suitable for many data science applications. For further learning, refer to the [LangChain](https://www.langchain.com/) and [Ollama](https://ollama.com) official documentation.

        """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
