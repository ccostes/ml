# ML Workspace
Repo of machine learning related code and articles.

Jupytext is used to version-control jupyter notebooks as .py files with the percent
formatting. `.ipynb` files are not versioned, and can be generated with the 
`jupytext --to notebook --execute <file>.py` command.

Publishable content for each subdirectory is stored in a `public` folder which is versioned
to enable publishing without rebuilding the source notebooks.

## Building
Install python deps in a virtual environment:
```
python -m vevnv env
source env/bin/activate
pip install -r requirements.txt
```

To build the markdown files to publish on the blog, use jupytext eg.: `jupytext --execute --to markdown am.py --output ./index.md`
