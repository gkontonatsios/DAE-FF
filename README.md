# DAE-FF

A Neural Network-based Feature Extraction Method to Facilitate Citation Screening for Systematic Reviews.

## Installation

This code was tested with Python 3.6 on Ubuntu 18.04 and Windows 10.

```
% pip install -r requirements.txt
```

If you plan to run the code on a **GPU** install packages using [conda](https://anaconda.org/anaconda/tensorflow-gpu):

```bash
% conda install -c anaconda tensorflow-gpu==1.13.1 scikit-learn pandas nltk 
% pip install console-progressbar keras==2.0.9
```


You need to also download data for `nltk`:

```python
import nltk

nltk.download(['punkt', 'stopwords', 'averaged_perceptron_tagger'])
```
