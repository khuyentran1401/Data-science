# Define variables
NOTEBOOK ?= notebook.py  # Default value, can be overridden
OUTPUT_DIR = marimo_notebooks
OUTPUT_FILE = $(OUTPUT_DIR)/$(notdir $(NOTEBOOK:.py=.html))

# Create the output directory if it doesn't exist
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Export the Marimo notebook to Jupyter Notebook format
html: $(OUTPUT_DIR)
	marimo export html $(NOTEBOOK) --output $(OUTPUT_FILE)

# Phony targets
.PHONY: html