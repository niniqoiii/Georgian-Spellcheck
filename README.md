# Georgian Spellchecker Project

This project implements Assignment 2: a character-level Georgian spellchecker using a recurrent encoder-decoder GRU model.

## Online dataset

The Georgian word list is loaded from this Google Drive file:

https://drive.google.com/file/d/1NnELSMHpI9ru6RgzGVT6wiIiTAjWKgj_/view?usp=sharing

The same file is cached locally at:

```text
data/georgian_words.txt
```

If the Google Drive dataset changes, rerun `data_and_training.ipynb`. The notebook will try to download the latest version from Drive, save it into `data/georgian_words.txt`, generate new synthetic errors, and retrain the model.

## Files

```text
data_and_training.ipynb        # data loading, typo generation, model training
inference.ipynb                # correct_word demo and 20+ examples
spellchecker.py                # model, training, inference helper code
data/georgian_words.txt        # cached Georgian word dataset
model/georgian_spellchecker.pt # trained model checkpoint
requirements.txt
```

## How to run

Install requirements:

```bash
pip install -r requirements.txt
```

Then run:

1. `data_and_training.ipynb`
2. `inference.ipynb`

The required function is available in the inference notebook:

```python
def correct_word(word: str, model_path: str) -> str:
    """
    Takes a potentially misspelled Georgian word and returns the corrected version.
    """
```

## Important note

`georgian_words.txt` must contain only correctly spelled Georgian words, one word per line. The code creates misspelled training examples automatically.
