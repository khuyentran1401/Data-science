import marimo

__generated_with = "0.13.7"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Build Smarter Data Science Workflows with DeepSeek and LangChain

        ## Introduction

        Data science workflows require both technical precision and clear communication. Many data science tasks need more than text generation—they require step-by-step problem solving. For example:

        - Debugging requires tracing logic, spotting data issues, and interpreting metrics.
        - Explaining features means sequencing steps clearly so non-technical stakeholders understand.

        Most language models have trouble with data science reasoning tasks. They often miss statistical concepts or create unreliable solutions. DeepSeek models focus on reasoning and coding, solving analytical problems systematically while producing dependable code for data applications. LangChain connects these models to production workflows, handling prompts, responses, and output parsing.

        This guide combines DeepSeek's reasoning abilities with LangChain's framework. You'll learn API setup, chain building, response streaming, and data extraction. We also cover local Ollama deployment for projects with privacy requirements.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Introduction to DeepSeek and LangChain

        Before diving into integration steps, let's understand both tools and why they work well together.

        ### What is DeepSeek?

        ![DeepSeek AI logo and visualization showing advanced language model capabilities for reasoning and code generation](https://platform.theverge.com/wp-content/uploads/sites/2/chorus/uploads/chorus_asset/file/25848982/STKB320_DEEPSEEK_AI_CVIRGINIA_A.jpg?quality=90&strip=all&crop=0,0,100,100)

        DeepSeek offers open-source language models that excel at reasoning and coding tasks. They can be accessed via an API or run locally, making them suitable for working with sensitive data that can't leave your system.

        #### Installation and Setup For DeepSeek

        DeepSeek is primarily accessed through its API. You'll need to create a [DeepSeek account](https://platform.deepseek.com), generate an API key, and add a balance. Once you have your API key, set it as an environment variable in a `.env` file:

        ```bash
        # Create .env file if it doesn't exist
        # Add the following line to your .env file:
        DEEPSEEK_API_KEY="sk-your-api-key-here"
        ```

        Make sure to load the environment variable in your script:

        """
    )
    return


@app.cell
def _():
    from dotenv import load_dotenv  # pip install python-dotenv

    load_dotenv()  # Load your API key
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### What is LangChain?

        LangChain is a framework for creating AI applications using language models.

        Rather than writing custom code for model interactions, response handling, and error management, you can use LangChain's ready-made components to build applications.

        >

        #### Installation and Setup For LangChain

        To get started with LangChain and DeepSeek integration, install:

        ```python
        pip install langchain langchain-deepseek python-dotenv
        ```

        In this installation:

        - langchain provides the core framework
        - langchain-deepseek enables DeepSeek model integration
        - python-dotenv helps manage environment variables for API keys.

        ### Why combine DeepSeek with LangChain?

        DeepSeek's models excel at reasoning-heavy tasks, but integrating them into larger workflows can be challenging without proper tooling. LangChain provides the infrastructure to build production-ready applications around DeepSeek's capabilities. You can create chains that validate inputs, format prompts properly, parse structured outputs, and handle errors gracefully. This combination gives you both powerful reasoning models and the tools to deploy them reliably.

        ## LangChain + DeepSeek: Integration Tutorial

        Now that we understand both tools, let's build a complete integration that puts DeepSeek's reasoning capabilities to work in your data science projects.

        ### Using DeepSeek Chat Models

        DeepSeek offers two main model families through their API:

        - **DeepSeek-V3** (`model="deepseek-chat"`) - general purpose model with tool calling and structured output
        - **DeepSeek-R1** (`model="deepseek-reasoner"`) - focused on reasoning tasks

        After setting up your environment, you'll need to select a model for your tasks. Here's how to use the chat model with LangChain:
        """
    )
    return


@app.cell
def _():
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_deepseek import ChatDeepSeek

    # Initialize the chat model
    llm = ChatDeepSeek(
        model="deepseek-chat",  # Can also use "deepseek-reasoner"
        temperature=0,  # 0 for more deterministic responses
        max_tokens=None,  # None means model default
        timeout=None,  # API request timeout
        max_retries=2,  # Retry failed requests
    )

    # Create a conversation with system and user messages
    messages = [
        SystemMessage(
            content="You are a helpful assistant that translates English to French."
        ),
        HumanMessage(content="Translate this sentence: I love programming."),
    ]

    # Generate a response
    response = llm.invoke(messages)
    print(response.content)
    return ChatDeepSeek, llm, messages


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        You can also use asynchronous operations for handling multiple requests without blocking:
        """
    )
    return


@app.cell
def _(llm, messages):
    async def generate_async():
        response = await llm.ainvoke(messages)
        return response.content

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Async operations let you process multiple DeepSeek requests in parallel, which helps when running batch operations . This approach reduces total waiting time compared to sequential processing, making it valuable for pipelines that need to scale.

        ### Building Chains with DeepSeek

        Individual model calls work for simple tasks, but data science workflows usually need multiple steps: input validation, prompt formatting, model inference, and output processing. Manual management of these steps becomes tedious and error-prone in larger projects.

        LangChain lets you connect components into chains. Here's how to build a translation chain with DeepSeek:
        """
    )
    return


@app.cell
def _(llm):
    from langchain_core.prompts import ChatPromptTemplate

    # Ensure llm is initialized as shown previously
    # llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

    # Create a structured prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that translates {input_language} to {output_language}.",
            ),
            ("human", "{input}"),
        ]
    )

    # Build the chain
    chain = prompt | llm

    # Execute the chain
    result = chain.invoke(
        {
            "input_language": "English",
            "output_language": "German",
            "input": "I love programming.",
        }
    )

    print(result.content)
    return ChatPromptTemplate, prompt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's break down this code:

        1. `ChatPromptTemplate.from_messages()` creates a template with placeholders in curly braces (`{input_language}`). These placeholders get replaced with actual values when the chain runs.
        2. The prompt template contains tuples for different message types - `("system", "...")` for system instructions and `("human", "...")` for user messages.
        3. The pipe operator (`|`) connects components. When you write `prompt | llm`, it means "send the output of prompt to llm as input." This creates a processing pipeline.
        4. `chain.invoke()` runs the chain with a dictionary of values that replace the placeholders. The dictionary keys match the placeholder names in the template.
        5. Behind the scenes, LangChain formats the messages, sends them to DeepSeek, and returns the model's response.

        This approach lets you build complex workflows while keeping your code clean and maintainable.

        ### Streaming Responses

        When working with complex data analysis or long-form explanations, users often wait several seconds for complete responses. This creates a poor experience in interactive applications like data exploration notebooks or real-time dashboards.

        For this reason, you might want to stream tokens as they're generated:
        """
    )
    return


@app.cell
def _(llm, prompt):
    from langchain_core.output_parsers import StrOutputParser

    # Ensure prompt and llm are initialized as shown previously
    # prompt = ChatPromptTemplate.from_messages([...])
    # llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

    streamed_chain = prompt | llm | StrOutputParser()

    for chunk in streamed_chain.stream(
        {
            "input_language": "English",
            "output_language": "Italian",
            "input": "Machine learning is changing the world.",
        }
    ):
        print(chunk, end="", flush=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In the streaming example above, several key elements work together:

        1. The `StrOutputParser()` converts the model's message output into plain text strings.
        2. The chain uses the `stream()` method instead of `invoke()` to receive content in chunks.
        3. The for-loop processes each text fragment as it arrives:
           - `print(chunk, end="")` displays the text without adding newlines between chunks
           - `flush=True` ensures text appears immediately without buffering

        When you run this code, you'll see words appear progressively rather than waiting for the entire translation to complete.

        ### Comparing `deepseek-chat` and `deepseek-reasoner` for specific tasks

        Different data science tasks benefit from different model approaches. While both DeepSeek models can handle general queries, choosing the right model for your specific use case can improve results and reduce costs.

        The `deepseek-chat` model works well for most data science tasks that need quick, direct responses. The `deepseek-reasoner` model excels when you need detailed step-by-step analysis or complex problem-solving that requires showing the reasoning process.

        Here, we are comparing both models on the same task:
        """
    )
    return


@app.cell
def _(ChatDeepSeek, HumanMessage_1):
    chat_model = ChatDeepSeek(model="deepseek-chat", temperature=0)
    reasoner_model = ChatDeepSeek(model="deepseek-reasoner", temperature=0)
    test_query = [
        HumanMessage_1(
            content="\nI have a dataset with 10,000 customer records. My logistic regression model achieves 85% accuracy, but precision for the positive class is only 60%. \nWhat steps should I take to improve this?\n"
        )
    ]
    print("=== DeepSeek-Chat Response ===")
    chat_response = chat_model.invoke(test_query)
    print(chat_response.content)
    print("\n=== DeepSeek-Reasoner Response ===")
    reasoner_response = reasoner_model.invoke(test_query)
    print(reasoner_response.content)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The reasoner model will typically provide more detailed explanations with explicit reasoning steps, while the chat model gives concise, actionable answers. Choose based on whether you need transparency in the thinking process or quick practical solutions to less challenging tasks.

        ### Structured Output

        Data science applications often need to extract specific information from model responses and feed it into other systems. Free-text responses make this difficult because parsing natural language output reliably requires additional processing and error handling.

        When you need structured data instead of free text, you can use DeepSeek's structured output capability with LangChain:
        """
    )
    return


@app.cell
def _(ChatPromptTemplate, llm):
    from typing import List

    from langchain_core.pydantic_v1 import BaseModel, Field

    # Ensure llm is initialized: llm = ChatDeepSeek(model="deepseek-chat")

    # Define the output schema
    class MovieReview(BaseModel):
        title: str = Field(description="The title of the movie")
        year: int = Field(description="The year the movie was released")
        rating: float = Field(description="Rating from 0-10")
        pros: List[str] = Field(description="List of positive aspects")
        cons: List[str] = Field(description="List of negative aspects")

    # Bind the Pydantic model to the LLM for structured output
    structured_llm = llm.with_structured_output(MovieReview)

    # Create a chain
    prompt_structured = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a movie critic. Provide your output in the requested structured format.",
            ),
            ("human", "Write a review for {movie_title}."),
        ]
    )

    chain_structured = prompt_structured | structured_llm

    # Get structured output
    review = chain_structured.invoke({"movie_title": "The Matrix"})
    print(f"Title: {review.title}")
    print(f"Year: {review.year}")
    print(f"Rating: {review.rating}/10")
    print("Pros:")
    for pro in review.pros:
        print(f"- {pro}")
    print("Cons:")
    for con in review.cons:
        print(f"- {con}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This example defines a `MovieReview` Pydantic model and uses `with_structured_output` to instruct the DeepSeek model to return data in this format. This is very useful for data extraction and building predictable AI workflows.

        ## Combining Ollama and DeepSeek in LangChain

        While DeepSeek's cloud API provides powerful reasoning capabilities, some data science projects involve sensitive data that cannot leave your infrastructure. Regulatory requirements, proprietary datasets, or security policies may prevent you from using external APIs entirely.

        While this tutorial focuses on the DeepSeek API, it's worth noting that DeepSeek models can also be run locally using Ollama. This offers a way to keep data private. For detailed instructions on setting up Ollama with LangChain, please refer to our [LangChain + Ollama tutorial](placeholder-link-to-ollama-article.md).

        ### Running DeepSeek Locally with Ollama

        Setting up local AI inference involves downloading large model files and configuring your system properly. Ollama simplifies this process by handling model management and serving automatically:
        """
    )
    return


app._unparsable_cell(
    r"""
    # Pull the DeepSeek model to your local machine (if you have Ollama installed)
    ollama pull deepseek:7b # Example, other versions may be available
    """,
    name="_",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Once downloaded, you can access it through LangChain just like any other Ollama model:
        """
    )
    return


@app.cell
def _(HumanMessage_1):
    from langchain_ollama import ChatOllama

    try:
        local_deepseek_ollama = ChatOllama(
            model="deepseek:7b", temperature=0.7, base_url="http://localhost:11434"
        )
        response_local = local_deepseek_ollama.invoke(
            [
                HumanMessage_1(
                    content="Write a recursive function to calculate Fibonacci numbers."
                )
            ]
        )
        print("Response from local DeepSeek (via Ollama):")
        print(response_local.content)
    except Exception as e:
        print(f"Could not connect to local Ollama or run DeepSeek model: {e}")
        print(
            "Ensure Ollama is installed, running, and the deepseek:7b model is pulled."
        )
    return (ChatOllama,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The only difference is the base URl, which will be a localhost link to the running Ollama instance, usually at port 11434.

        ### Switching Between Local (Ollama-hosted) and Cloud Versions of DeepSeek

        Production data science applications often need to balance multiple constraints: some tasks require the latest models available through APIs, while others must process sensitive data locally. Maintaining separate codebases for different deployment modes creates maintenance overhead and increases complexity.

        You can implement code that works with both local (Ollama-hosted) and cloud versions of DeepSeek in your applications:
        """
    )
    return


@app.cell
def _(ChatDeepSeek, ChatOllama, HumanMessage_1):
    cloud_deepseek_model = ChatDeepSeek(model="deepseek-chat", temperature=0)
    try:
        local_deepseek_ollama_model = ChatOllama(
            model="deepseek:7b", base_url="http://localhost:11434", temperature=0.7
        )
        ollama_available = True
    except Exception:
        ollama_available = False
        print("Local Ollama DeepSeek model not available. Will use API only.")
    query_message = [
        HumanMessage_1(content="Explain the concept of zero-shot learning.")
    ]
    if ollama_available:
        print("\n--- Attempting Local DeepSeek (via Ollama) ---")
        try:
            response_1 = local_deepseek_ollama_model.invoke(query_message)
            print(response_1.content)
        except Exception as e:
            print(f"Error with local Ollama model: {e}. Falling back to cloud API.")
            response_1 = cloud_deepseek_model.invoke(query_message)
            print("\n--- Response from Cloud DeepSeek API ---")
            print(response_1.content)
    else:
        print("\n--- Using Cloud DeepSeek API ---")
        response_1 = cloud_deepseek_model.invoke(query_message)
        print(response_1.content)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        This approach allows you to use local inference for speed and privacy, falling back to the API for potentially more powerful models or when local setup is unavailable.

        ### Performance and cost considerations for DeepSeek API vs. local Ollama

        When building data science applications, choosing between API-based and local deployment involves trade-offs between performance, cost, privacy, and maintenance complexity.

        **API Deployment (DeepSeek Cloud):**

        **Best For:** Quick prototyping, access to cutting-edge models, minimal infrastructure setup.

        *Pros:*

        - No hardware requirements or model management
        - Access to latest model versions immediately
        - Scales automatically with usage
        - No local storage needs (models can be 7GB+)

        *Cons:*

        - API costs accumulate with usage (typically $0.001-0.002 per 1K tokens)
        - Requires internet connectivity for all requests
        - Data leaves your infrastructure (potential privacy concerns)
        - Rate limits may apply during peak usage

        **Local Deployment (Ollama):**

        **Best For:** Privacy-critical use cases, stable performance, and cost-efficient production.

        *Pros:*

        - Complete data privacy and control
        - No ongoing API costs after initial setup
        - Works offline once models are downloaded
        - Predictable performance without network latency

        *Cons:*

        - Requires significant hardware resources (16GB+ RAM recommended)
        - Model downloads can be several GB in size
        - Manual updates and maintenance required
        - Initial setup complexity

        **Recommendation for data science teams:**

        - **Start with API** for prototyping and experimentation
        - **Move to local deployment** for production applications with sensitive data or high-volume processing
        - **Use hybrid approach** with local models for routine tasks and API for complex reasoning that benefits from larger models

        ## Conclusion

        This tutorial provided a comprehensive guide to integrating LangChain with DeepSeek. You've learned how to use DeepSeek's API for chat, chaining, streaming, and structured output. We also briefly covered running DeepSeek locally via Ollama, offering a path to privacy-enhanced AI. By combining LangChain's framework with DeepSeek's powerful models, you are well-equipped to build sophisticated data science applications. For further details, consult the [LangChain](https://www.langchain.com/) and [DeepSeek](https://www.deepseek.com/) official documentation.

        """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
