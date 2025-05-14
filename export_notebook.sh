#!/bin/bash

# Check if a notebook name was provided
if [ -z "$1" ]; then
    echo "Error: Please provide a notebook name"
    echo "Usage: ./export_notebook.sh notebook_name"
    exit 1
fi

# Get the notebook name without .py extension if provided
notebook_name=${1%.py}

# Get the directory path
dir_path=$(dirname "$notebook_name")
notebook_base=$(basename "$notebook_name")

# Create build directory and subdirectories if they don't exist
mkdir -p "build/$dir_path"

# Export the notebook to HTML
echo "Exporting $notebook_name.py to HTML..."
uv run marimo export html "$notebook_name.py" -o "build/$notebook_name.html" --sandbox

# Check if the export was successful
if [ $? -eq 0 ]; then
    echo "Successfully exported $notebook_name.py to build/$notebook_name.html"
else
    echo "Error: Failed to export notebook"
    exit 1
fi 