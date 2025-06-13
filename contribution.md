<file name=1 path=/Users/khuyentran/Data-science/contribution.md># Contribution Guidelines

## Table of Contents

- [CodeCut Mission](#codecut-mission)
- [Your Responsibility as a Writer](#your-responsibility-as-a-writer)
- [Writing Checklist](#writing-checklist)
- [Write Article Draft](#write-article-draft)
- [Write Code](#write-code)

## CodeCut Mission

CodeCut exists to help data scientists stay productive and up-to-date by delivering short, focused, and practical code examples that showcase modern tools in action.

We strive to:

- Help readers quickly understand what a tool does
- Show how it fits into real-world data science workflows
- Provide just enough to empower readers to try it on their own

## Your Responsibility as a Writer

As a writer for CodeCut, your role is to:

- Break down complex tools and workflows into clear, digestible pieces
- Focus on practical value over theoretical depth
- Maintain a tone that is approachable, confident, and helpful
- Show rather than tell - use code snippets, visuals, or graphs to demonstrate your points

## How to Write a Good Article

Good technical articles are:

- Easy to skim
- Broadly helpful
- Clear and concise

Follow the tips highlighted in [How to Write Good Technical Articles](how_to_write_good_articles.md) to write a good article.

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
   ```bash
   # Click the "Fork" button on the repository's GitHub page
   # Then clone your forked repository
   git clone https://github.com/YOUR-USERNAME/REPOSITORY-NAME.git
   cd REPOSITORY-NAME
   ```

2. Create a new branch for your feature
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Make your changes
   ```bash
   # Make your code changes
   # Add your changes to staging
   git add .
   # Commit your changes
   git commit -m "Description of your changes"
   ```

4. Pull the latest changes
   ```bash
   # Pull the latest changes
   git pull origin main
   ```

5. Submit a pull request
   ```bash
   # Push your changes to your fork
   git push origin feature/your-feature-name
   # Then go to GitHub and click "Create Pull Request"
   ```