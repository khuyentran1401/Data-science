# Setting Up Automatic Publishing of Marimo Notebooks to GitHub Pages

## What Has Been Completed

1. Created a GitHub Actions workflow file at `.github/workflows/publish-marimo.yml`
2. Committed the workflow file to the local repository

## How This Works

The workflow we've set up will:

- Trigger automatically whenever Python files are pushed to the main/master branch
- Allow manual triggering through the GitHub Actions interface
- Find all Python files in your repository that contain "import marimo" (excluding the venv directory)
- Export each found notebook to HTML format using marimo's export feature
- Create a nice index page that links to all the exported notebooks
- Deploy everything to the gh-pages branch, which will serve as the source for GitHub Pages

## Next Steps

1. **Push the changes to GitHub**:
   ```bash
   git push origin master
   ```

2. **Configure GitHub Pages in your repository settings**:
   - Go to your GitHub repository at https://github.com/khuyentran1401/Data-science
   - Navigate to Settings → Pages
   - Under "Source", select "Deploy from a branch"
   - Select the "gh-pages" branch and the "/ (root)" folder
   - Click "Save"

3. **Wait for the workflow to run**:
   - After pushing your changes, the workflow will run automatically
   - You can monitor its progress in the "Actions" tab of your repository
   - Once completed successfully, it will create the gh-pages branch

4. **Access Your Published Notebooks**:
   - Your notebooks will be available at: https://khuyentran1401.github.io/Data-science/
   - The index page will show links to all of your published notebooks
   - Your polars_vs_pandas.py notebook will be published at: https://khuyentran1401.github.io/Data-science/data_science_tools/polars_vs_pandas.html

## Adding More Marimo Notebooks

To add more marimo notebooks to be published:

1. Create or edit a Python file that uses marimo (imports the marimo package)
2. Commit and push the file to your GitHub repository
3. The workflow will automatically detect and publish the new notebook

## Troubleshooting

If your notebooks aren't being published:

1. Ensure they import marimo (the workflow searches for "import marimo" in the file)
2. Check the GitHub Actions logs for any errors
3. Make sure GitHub Pages is properly configured to serve from the gh-pages branch
4. Verify that your repository has the necessary permissions for GitHub Actions

## Dependencies

The workflow installs the following Python packages:
- marimo
- pandas
- numpy
- polars

If your notebooks require additional dependencies, you'll need to add them to the "Install dependencies" step in the workflow file.

