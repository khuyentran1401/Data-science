import re
from pathlib import Path


def get_notebook_title(html_path):
    """Extract title from HTML file."""
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Look for title in the HTML content
            title_match = re.search(r"<title>(.*?)</title>", content)
            if title_match:
                return title_match.group(1)
    except Exception as e:
        print(f"Error reading {html_path}: {e}")
    return Path(html_path).stem.replace("_", " ").title()

def generate_index():
    # Paths
    public_dir = Path("public")
    index_path = public_dir / "index.html"

    # Get all HTML files from subdirectories
    html_files = sorted(public_dir.glob("*/*.html"))

    # HTML template
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CodeCut Blog</title>
    <link href="https://fonts.googleapis.com/css2?family=Comfortaa:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Comfortaa', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            color: white;
            background-color: #2F2D2E;
        }}
        h1 {{
            color: white;
            border-bottom: 2px solid white;
            padding-bottom: 0.5rem;
        }}
        .notebook-list {{
            list-style: none;
            padding: 0;
        }}
        .notebook-item {{
            margin-bottom: 2rem;
            padding: 1rem;
            border: 2px solid white;
            border-radius: 8px;
            transition: transform 0.2s;
        }}
        .notebook-item:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(255, 255, 255, 0.1);
        }}
        .notebook-title {{
            font-size: 1.5rem;
            margin: 0 0 0.5rem 0;
            color: white;
        }}
        .notebook-description {{
            color: #CCCCCC;
            margin-bottom: 1rem;
        }}
        .notebook-link {{
            display: inline-block;
            padding: 0.5rem 1rem;
            background-color: #72BEFA;
            color: #2F2D2E;
            text-decoration: none;
            border-radius: 4px;
            transition: all 0.2s;
            border: 2px solid white;
            font-weight: 600;
        }}
        .notebook-link:hover {{
            background-color: #5AA8E8;
            border: 2px solid white;
            transform: translateY(-2px);
        }}
    </style>
</head>
<body>
    <h1>CodeCut Marimo Notebook</h1>
    <ul class="notebook-list">
{notebook_items}
    </ul>
</body>
</html>"""

    # Generate notebook items
    notebook_items = []
    for html_file in html_files:
        if html_file.name == "index.html":  # Skip index.html itself
            continue
        title = get_notebook_title(html_file)
        relative_path = html_file.relative_to(public_dir)
        notebook_items.append(f"""        <li class="notebook-item">
            <h2 class="notebook-title">{title}</h2>
            <a href="{relative_path}" class="notebook-link">View the notebook</a>
        </li>""")

    # Generate final HTML
    final_html = html_template.format(notebook_items="\n".join(notebook_items))

    # Write to index.html
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    print(f"Generated index.html with {len(notebook_items)} notebooks")

if __name__ == "__main__":
    generate_index()
