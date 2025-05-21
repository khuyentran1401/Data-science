# Contribution Guidelines

## Table of Contents

- [CodeCut Mission](#codecut-mission)
- [Your Responsibility as a Writer](#your-responsibility-as-a-writer)
- [Writing Checklist](#writing-checklist)
- [Write Article Draft](#write-article-draft)
- [Write Code](#write-code)

## CodeCut Mission

CodeCut exists to help data scientists stay productive and up-to-date by delivering short, focused, and practical code examples that showcase modern tools in action. Our mission is to make complex concepts simple and actionable—not through lengthy explanations, but by showing how things work with clarity and precision.

We strive to:

- Help readers quickly understand what a tool does
- Show how it fits into real-world data science workflows
- Provide just enough to empower readers to try it on their own

We want CodeCut to feel like the resource you wish you had when exploring a new library: clean, concise, and immediately useful.

## Your Responsibility as a Writer

As a writer for CodeCut, your role is to:

- Break down complex tools and workflows into clear, digestible pieces
- Focus on practical value over theoretical depth
- Maintain a tone that is approachable, confident, and helpful
- Write only about topics you are genuinely interested in
- Enjoy the writing process—we want this to be fun for you, too

## Writing Checklist

### Writing Style Checklist

- [ ] Use action verbs instead of passive voice
- [ ] Limit paragraphs to 2-4 sentences
- [ ] For every major code block, provide a clear explanation of what it does and why it matters.
- [ ] Structure content for quick scanning with clear headings and bullet points

### Data Science-Focused Writing Checklist

- [ ] Write for data scientists comfortable with Python but unfamiliar with this specific tool or library.
- [ ] Use examples that align with common data science workflows or problems
- [ ] Highlight **only** the features that matter to a data science audience

### Structure Checklist

- [ ] Start with a real, practical data science problem
- [ ] Explain how each tool solves the problem
- [ ] Use diagrams or charts to explain complex ideas, when appropriate.
- [ ] Define new concepts and terminology
- [ ] Only include the essential setup steps needed to run the examples. For anything beyond that, link to the official documentation.

## Write Article Draft

1. Create your blog post in [HackMD](https://hackmd.io)
2. Follow [these instructions](https://hackmd.io/c/tutorials/%2F%40docs%2Finvite-others-to-a-private-note-en) to share your draft with khuyentran@codecut.ai for review

## Write Code

### Environment Setup

#### Install uv

[uv](https://github.com/astral.sh/uv) is a fast Python package installer and resolver.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

#### Install Dependencies

```bash
# Install dependencies from pyproject.toml
uv sync
```

#### Install Pre-commit Hooks

We use pre-commit to ensure code quality and consistency.

```bash
# Install pre-commit hooks
uv run pre-commit install
```

### Working with Marimo Notebooks

#### Creating a New Notebook

Create a new notebook using marimo:

```bash
uv run marimo edit notebook.py --sandbox
```

#### Publishing Notebooks

To export your marimo notebooks to HTML locally:

1. Make sure the `export_notebook.sh` script is executable:

   ```bash
   chmod +x export_notebook.sh
   ```

2. Run the script with your notebook name:

   ```bash
   # For notebooks in the root directory
   ./export_notebook.sh notebook_name

   # For notebooks in subdirectories
   ./export_notebook.sh path/to/notebook_name
   ```

   For example:

   ```bash
   ./export_notebook.sh data_science_tools/polars_vs_pandas
   ./export_notebook.sh llm/pydantic_ai_examples
   ```

The exported HTML files will be automatically deployed to GitHub Pages through the GitHub Actions workflow.

### Pull Request Process

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Submit a pull request with a clear description of changes

