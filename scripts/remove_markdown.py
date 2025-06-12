#!/usr/bin/env python3
import re
import sys

import nbformat as nbf


def retain_headings(md_source):
    """Keep only heading lines (start with #) and blank lines."""
    lines = md_source.splitlines()
    kept = [ln for ln in lines if re.match(r"^\s*#+\s+", ln) or not ln.strip()]
    return "\n".join(kept).strip() + "\n" if kept else ""

def clean_notebook(in_nb_path, out_nb_path):
    ntbk = nbf.read(in_nb_path, as_version=nbf.NO_CONVERT)
    new_cells = []

    for cell in ntbk.cells:
        if cell.cell_type == "markdown":
            cleaned_md = retain_headings(cell.source)
            if cleaned_md:
                cell.source = cleaned_md
                new_cells.append(cell)
        elif cell.cell_type == "code":
            new_cells.append(cell)  # keep code cells untouched
        # skip raw or other cell types

    ntbk.cells = new_cells
    nbf.write(ntbk, out_nb_path, version=nbf.NO_CONVERT)
    print(f"✅ Cleaned notebook saved to {out_nb_path}")

def main():
    if len(sys.argv) != 3:
        print("Usage: remove_markdown.py INPUT.ipynb OUTPUT.ipynb")
        sys.exit(1)
    clean_notebook(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()
