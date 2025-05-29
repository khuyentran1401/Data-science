import marimo

__generated_with = "0.13.7"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Getting started with Diffbot LLM

        ```yaml
        pip install openai python dotenv
        touch .env  # Create a .env file
        echo "DIFFBOT_API_TOKEN=your-token-here" >> .env  # Add your token to a .env file
        ```
        ### Your first query with citations

        """
    )
    return


@app.cell
def _():
    import os

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()

    client = OpenAI(
        base_url="https://llm.diffbot.com/rag/v1",
        api_key=os.getenv("DIFFBOT_API_TOKEN"),
    )

    def query_diffbot(query_text, model="diffbot-small-xl", temperature=0.5):
        completion = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": query_text}],
        )
        return completion

    completion = query_diffbot("What is GraphRAG?")
    return completion, load_dotenv, os, query_diffbot


@app.cell
def _(completion):
    print(completion.choices[0].message.content[:1000])
    return


@app.cell
def _(query_diffbot):
    completion_1 = query_diffbot("What is the weather in Tokyo?")
    print(completion_1.choices[0].message.content)
    return


@app.cell
def _(query_diffbot):
    completion_2 = query_diffbot(
        "Find me the information on the upcoming hackathon organized by HuggingFace"
    )
    print(completion_2.choices[0].message.content)
    return


@app.cell
def _(query_diffbot):
    completion_3 = query_diffbot("Find the square root of 12394890235")
    print(completion_3.choices[0].message.content)
    return


@app.cell
def _(query_diffbot):
    _image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Black_hole_-_Messier_87_crop_max_res.jpg/960px-Black_hole_-_Messier_87_crop_max_res.jpg"
    completion_4 = query_diffbot(f"Describe this image to me: {_image_url}")
    print(completion_4.choices[0].message.content)
    return


@app.cell
def _(query_diffbot):
    _image_url = "https://codecut.ai/wp-content/uploads/2025/01/Banner-updated-colors-fixed-maybe-1.png"
    completion_5 = query_diffbot(f"Describe this image to me: {_image_url}")
    print(completion_5.choices[0].message.content)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Self-hosting for privacy
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If your use-case involves high-stakes sensitive information like financial or medical databases, you can get all the benefits of the Serverless API locally by running a couple of Docker commands:

        For the 8B model, much smaller in disk size:

        ```bash
        docker run --runtime nvidia --gpus all -p 8001:8001 --ipc=host -e VLLM_OPTIONS="--model diffbot/Llama-3.1-Diffbot-Small-2412 --served-model-name diffbot-small --enable-prefix-caching"  docker.io/diffbot/diffbot-llm-inference:latest
        ```

        For the larger 70B model with full capabilities:

        ```bash
        docker run --runtime nvidia --gpus all -p 8001:8001 --ipc=host -e VLLM_OPTIONS="--model diffbot/Llama-3.3-Diffbot-Small-XL-2412 --served-model-name diffbot-small-xl --enable-prefix-caching --quantization fp8 --tensor-parallel-size 2"  docker.io/diffbot/diffbot-llm-inference:latest
        ```

        Once the application starts up successfully and you see a message like the following:

        ```plaintext
        INFO:  Application startup complete.
        INFO:  Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
        ```

        You can run all the examples above by replacing the base URL with the endpoint `http://localhost:8001/rag/v1`.

        However, do note that these models require high-end GPUs like A100 and H100s to run at full precision. If you don't have the right hardware, consider using [RunPod.io](https://runpod.io) which cost:

        - $5.98/hr for dual H100 GPU setup (total 160 GB VRAM)
        - $1.89/hr for a single A100 GPU setup (80 GB VRAM)

        ![](images/runpod-instances.png)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Building real-world applications with Diffbot and LangChain

        ### LangChain + Diffbot basics

        ```bash
        pip install langchain langchain-openai python-dotenv
        ```
        """
    )
    return


@app.cell
def _(load_dotenv, os):
    from langchain_openai import ChatOpenAI

    load_dotenv()

    llm = ChatOpenAI(
        model="diffbot-small-xl",
        temperature=0,
        max_tokens=None,
        timeout=None,
        base_url="https://llm.diffbot.com/rag/v1",
        api_key=os.getenv("DIFFBOT_API_TOKEN"),
    )
    return ChatOpenAI, llm


@app.cell
def _(llm):
    messages = [
        ("system", "You are a helpful assistant that translates English to French."),
        ("human", "I love programming."),
    ]

    ai_msg = llm.invoke(messages)
    print(ai_msg.content)  # "J'aime le programmation."
    return


@app.cell
def _(llm):
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that translates {input_language} to {output_language}.",
            ),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm
    _result = chain.invoke(
        {
            "input_language": "English",
            "output_language": "German",
            "input": "I love programming.",
        }
    )
    return (ChatPromptTemplate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Building a RAG application with Diffbot and LangChain
        """
    )
    return


@app.cell
def _(ChatOpenAI, ChatPromptTemplate):
    import json
    from typing import Dict, List

    from langchain_core.output_parsers import StrOutputParser

    class ResearchAssistant:

        def __init__(self, diffbot_api_key: str):
            self.llm = ChatOpenAI(
                model="diffbot-small-xl",
                temperature=0.3,
                base_url="https://llm.diffbot.com/rag/v1",
                api_key=diffbot_api_key,
            )
            self.setup_chains()

        def setup_chains(self):
            self.topic_extraction_prompt = ChatPromptTemplate.from_template(
                "\n        Analyze the following document and extract 3-5 main topics or entities that would benefit \n        from current information. Return as a JSON list of topics.\n        \n        Document: {document}\n        \n        Topics (JSON format):\n        "
            )
            self.research_prompt = ChatPromptTemplate.from_template(
                "\n        Provide comprehensive, current information about: {topic}\n        \n        Context from document: {context}\n        \n        Include:\n        1. Current status and recent developments\n        2. Key statistics or data points  \n        3. Recent news or updates\n        4. Relevant industry trends\n        \n        Ensure all facts are cited with sources.\n        "
            )
            self.report_prompt = ChatPromptTemplate.from_template(
                "\n        Create a comprehensive research report based on the document analysis and current research.\n        \n        Original Document Summary: {document_summary}\n        \n        Research Findings: {research_findings}\n        \n        Generate a well-structured report that:\n        1. Summarizes the original document's main points\n        2. Provides current context for each major topic\n        3. Identifies any outdated information in the document\n        4. Suggests areas for further investigation\n        \n        Include proper citations throughout.\n        "
            )

        def extract_topics(self, document: str) -> List[str]:
            """Extract main topics from the document for research."""
            chain = self.topic_extraction_prompt | self.llm | StrOutputParser()
            try:
                _result = chain.invoke({"document": document})
                topics = json.loads(_result.strip())
                return topics if isinstance(topics, list) else []
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error extracting topics: {e}")
                return []

        def research_topic(self, topic: str, context: str) -> str:
            """Research current information about a specific topic."""
            chain = self.research_prompt | self.llm | StrOutputParser()
            return chain.invoke({"topic": topic, "context": context})

        def generate_report(self, document: str, research_findings: List[Dict]) -> str:
            """Generate comprehensive report with current information."""
            summary_prompt = ChatPromptTemplate.from_template(
                "Provide a concise summary of this document: {document}"
            )
            summary_chain = summary_prompt | self.llm | StrOutputParser()
            document_summary = summary_chain.invoke({"document": document})
            findings_text = "\n\n".join(
                [
                    f"**{finding['topic']}:**\n{finding['research']}"
                    for finding in research_findings
                ]
            )
            report_chain = self.report_prompt | self.llm | StrOutputParser()
            return report_chain.invoke(
                {
                    "document_summary": document_summary,
                    "research_findings": findings_text,
                }
            )

        def analyze_document(self, document: str) -> Dict:
            """Complete document analysis with current research."""
            print("Extracting topics from document...")
            topics = self.extract_topics(document)
            if not topics:
                return {"error": "Could not extract topics from document"}
            print(f"Researching {len(topics)} topics...")
            research_findings = []
            for topic in topics:
                print(f"  - Researching: {topic}")
                research = self.research_topic(topic, document)
                research_findings.append({"topic": topic, "research": research})
            print("Generating comprehensive report...")
            final_report = self.generate_report(document, research_findings)
            return {
                "topics": topics,
                "research_findings": research_findings,
                "final_report": final_report,
                "status": "completed",
            }

    return (ResearchAssistant,)


@app.cell
def _(ResearchAssistant, os):
    assistant = ResearchAssistant(os.getenv("DIFFBOT_API_TOKEN"))
    sample_document = "\nArtificial Intelligence has made significant progress in natural language processing. \nCompanies like OpenAI and Google have released powerful language models. \nThe field of machine learning continues to evolve with new architectures and techniques.\nInvestment in AI startups reached $25 billion in 2023.\n"
    _result = assistant.analyze_document(sample_document)
    print(_result["final_report"])
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
