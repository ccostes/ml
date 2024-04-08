# AM Radio Demodulation Model

Code used to train the transformer-based nerual network to demodulate and de-noise AM radio described in this [blog post](https://ccostes.com/posts/2024-04-07-sdr_ml/).

The main notebook is included as a [jupyter .ipynb](am_transformer.ipynb) file for convenience, with the associated [jupytext .py](am_transformer.py) file used for better version history.

To update the `.py` file with changes from the notebook, run `jupytext --to py:percent .\am_transformer.ipynb`.

## Setup
Create a python environment and install the requisite dependencies:
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```