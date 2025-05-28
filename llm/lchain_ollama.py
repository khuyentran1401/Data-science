# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langchain-community==0.3.24",
#     "langchain-core==0.3.61",
#     "langchain-ollama==0.3.3",
#     "marimo",
#     "numpy==2.2.6",
# ]
# ///

import marimo

__generated_with = "0.13.7"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# LangChain with Ollama: A Step-by-Step Integration Tutorial""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Installation and Setup

    ### Installation

    Ollama needs to be installed separately since it's a standalone service that runs locally:

    - For macOS: Download from [ollama.com](https://ollama.com) - this installs both the CLI tool and service
    - For Linux: `curl -fsSL https://ollama.com/install.sh | sh` - this script sets up both the binary and system service
    - For Windows: Download Windows (Preview) from [ollama.com](https://ollama.com) - still in preview mode with some limitations

    Start the Ollama server:

    ```bash
    ollama serve
    ```

    ### Pulling Models with Ollama

    Before using any model with LangChain, you need to pull it to your local machine:

    ```bash
    ollama pull qwen3:0.6b
    ```

    Once it is downloaded, you can serve the model with the following command:

    ```bash
    ollama run qwen3:0.6b
    ```


    ## Basic Chat Integration
    """
    )
    return


@app.cell
def _():
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama

    # Initialize the chat model with specific configurations
    chat_model = ChatOllama(
        model="qwen2.5:0.5b",
        temperature=0.3,  # Lower temperature for more deterministic outputs
        base_url="http://localhost:11434",
    )

    # Define a prompt for generating a basic function in a data science project
    messages = [
        SystemMessage(
            content="You are a data scientist who writes efficient Python code with short docstring."
        ),
        HumanMessage(
            content=(
                "Given a DataFrame with columns 'product' and 'sales', calculates the total sales for each product.")
        ),
    ]

    # Invoke the model and print the generated function
    response = chat_model.invoke(messages)
    print(response.content)
    return ChatOllama, chat_model, messages


@app.cell
async def _(chat_model, messages):
    async def generate_async():
        response = await chat_model.ainvoke(messages)
        return response.content

    # In async context
    result = await generate_async()

    print(result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Using Completion Models""")
    return


@app.cell
def _():
    from langchain_ollama import OllamaLLM

    llm = OllamaLLM(model="qwen2.5:0.5b")
    text = """
    Write a function that takes a DataFrame with columns 'product', 'year', and 'sales' and calculate the total sales for each product over the specified years.

    ```python
    def calculate_total_sales(df):
    """
    completion_response = llm.invoke(text)
    print(completion_response)
    return (OllamaLLM,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""For streaming responses (showing tokens as they're generated):""")
    return


@app.cell
def _():
    # for chunk in llm.stream(text):
    #     print(chunk, end="", flush=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Customizing Model Parameters

    Ollama offers fine-grained control over generation parameters:
    """
    )
    return


@app.cell
def _(OllamaLLM):
    llm_1 = OllamaLLM(
        model="qwen2.5:0.5b",
        temperature=0.7,
        repeat_penalty=1.1,
    )
    return


@app.cell
def _(OllamaLLM):
    # Scientific writing with precise output
    scientific_llm = OllamaLLM(model="qwen2.5:0.5b", temperature=0.1, repeat_penalty=1.2)

    scientific_prompt = "Summarize the key findings of a study on the impact of sleep on memory retention."
    scientific_response = scientific_llm.invoke(scientific_prompt)
    print(scientific_response)
    return


@app.cell
def _(OllamaLLM):
    # Creative storytelling
    creative_llm = OllamaLLM(model="qwen2.5:0.5b", temperature=0.9, repeat_penalty=1.0)

    creative_prompt = "Tell a short story about a robot who discovers the meaning of friendship."
    creative_response = creative_llm.invoke(creative_prompt)
    print(creative_response)
    return


@app.cell
def _(OllamaLLM):
    # Code generation
    code_llm = OllamaLLM(model="codellama:7b", temperature=0.3)

    code_prompt = "Write a Python function to calculate total sales of the `sale` column in a polars DataFrame."
    code_response = code_llm.invoke(code_prompt)
    print(code_response)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Creating LangChain Chains""")
    return


@app.cell
def _(OllamaLLM):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate

    model = OllamaLLM(model="codellama:7b")
    function_prompt = PromptTemplate.from_template(
        """
        Write a Python function using pandas that takes a DataFrame with columns '{date_col}', '{group_col}', and '{value_col}'.
        The function should return a new DataFrame that includes a {window}-day rolling average of {value_col} for each {group_col}.
        Only output the function code.
        """
    )

    # Build the chain
    code_chain = function_prompt | model | StrOutputParser()

    # Run the chain with specific variable values
    chain_response = code_chain.invoke({
        "date_col": "date",
        "group_col": "store_id",
        "value_col": "sales",
        "window": 7
    })

    print(chain_response)
    return PromptTemplate, StrOutputParser, code_chain


@app.cell
def _(OllamaLLM, PromptTemplate, StrOutputParser, code_chain):
    test_model = OllamaLLM(model="codellama:7b", temperature=0.3)

    test_prompt = PromptTemplate.from_template(
        """
        Given the following Python function:

        ```python
        {code}
        ```

        Write 1–2 simple unit tests for this function using pytest. Only include the test code.
        """
    )

    test_chain = (
        {"code": code_chain}
        | test_prompt
        | test_model
        | StrOutputParser()
    )

    # Invoke the test chain
    test_response = test_chain.invoke({
        "date_col": "date",
        "group_col": "store_id",
        "value_col": "sales",
        "window": 7
    })

    print(test_response)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Building a question-answering system for your data""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Working with Embeddings""")
    return


@app.cell
def _():
    import numpy as np
    from langchain_ollama import OllamaEmbeddings

    # Initialize embeddings model with specific parameters
    embedder = OllamaEmbeddings(
        model="nomic-embed-text",  # Specialized embedding model that is also supported by Ollama
    )
    return embedder, np


@app.cell
def _(embedder):
    # Create embeddings for a query
    example_query = "How do neural networks learn?"
    example_query_embedding = embedder.embed_query(example_query)
    print(f"Embedding dimension: {len(example_query_embedding)}")
    return


@app.cell
def _(embedder):
    documents = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning algorithms can automatically learn patterns from data without explicit programming.",
        "Data preprocessing involves cleaning, changing, and organizing raw data for analysis.",
        "Neural networks are computational models inspired by biological brain networks.",
    ]

    doc_embeddings = embedder.embed_documents(documents)

    print(f"Generated {len(doc_embeddings)} embeddings for the input documents.")
    return (documents,)


@app.cell
def _(np):
    # Calculate similarity between vectors
    def compute_cosine_similarity(query_vec, document_vec):
        """Compute cosine similarity between two vectors."""
        return np.dot(query_vec, document_vec) / (
            np.linalg.norm(query_vec) * np.linalg.norm(document_vec)
        )

    def get_most_similar_indices(similarities, num_documents):
        """Return indices of top `num_documents` highest similarity scores."""
        return np.argsort(similarities)[-num_documents:][::-1]


    def similarity_search(query, documents, embedder, num_documents=2):
        """Return the most relevant documents for the given query."""
        query_embedding = embedder.embed_query(query)
        document_embeddings = embedder.embed_documents(documents)
        similarities = [
            compute_cosine_similarity(query_embedding, doc_embedding)
            for doc_embedding in document_embeddings
        ]
        top_indices = get_most_similar_indices(similarities, num_documents)
        return [documents[i] for i in top_indices]
    return (similarity_search,)


@app.cell
def _(documents, embedder, similarity_search):
    question = "What makes Python popular for data science?"

    relevant_docs = similarity_search(query=question, documents=documents, embedder=embedder)

    print(f"Top {len(relevant_docs)} relevant documents retrieved:\n")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc}\n")
    return question, relevant_docs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Building a RAG Pipeline""")
    return


@app.cell
def _(PromptTemplate):
    rag_prompt = PromptTemplate.from_template("""
        Use the following context to answer the question. If the answer isn't in the context, say so.
        Context:
        {context}

        Question: {question}

        Answer:
    """)
    return (rag_prompt,)


@app.cell
def _(relevant_docs):
    context = "\n".join(relevant_docs)
    return (context,)


@app.cell
def _(ChatOllama, StrOutputParser, context, question, rag_prompt):
    rag_chat_model = ChatOllama(model="qwen2.5:0.5b", temperature=0.3)

    rag_chain = rag_prompt | rag_chat_model | StrOutputParser()

    # Run the chain with specific variable values
    rag_response = rag_chain.invoke({"context": context, "question": question})

    print(rag_response)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
