# ML Workspace
Repo of machine learning related code and articles.

Jupytext is used to version-control jupyter notebooks as .py files with the percent
formatting. `.ipynb` files are not versioned, and can be generated with the 
`jupytext --to notebook --execute <file>.py` command.

Publishable content for each subdirectory is stored in a `public` folder which is versioned
to enable publishing without rebuilding the source notebooks.

## Building
1. Install python deps
```
python -m vevnv env
source env/bin/activate
pip install -r requirements.txt
```
2. Build notebooks, eg. for am_explainer
```
cd am_explainer
jupytext --to notebook --execute am.py
jupyter nbconvert --to markdown --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags=hide-input --output-dir='./public' am.ipynb
```