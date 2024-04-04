# ML in Practice Part 1: AM Radio

This repo contains the code to generate this **[Blog Post](https://www.ccostes.com/posts/ml/am_explainer)**.
<a href="https://www.ccostes.com/posts/ml/am_explainer"><img src="static/blogpost.jpg" /></a>
The notebook source is stored as a `.py` file using jupytext for better version-control-ability, with the resulting `.ipynb` file stored for convenience.

## Build
Create a python environment and install the requisite dependencies:
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Create `.ipynb` Notebook
To generate the notebook file with cell outputs run `jupytext --execute --to notebook am.py` (leave off `--execute` to skip running all of the cells). 

If you've modified the `.py` file and want to update the `.ipynb` notebook, you can use the `--update` flag to prevent overwriting any existing cell outputs `jupytext --update --to notebook am.py`.

If you want to update the .py file with changes from the notebook, use `jupytext --to py:percent am.ipynb`

### Generate Markdown
To generate a markdown file for the blog post, run `jupytext --execute --to markdown am.py --output ./index.md` (this will take a min or two).