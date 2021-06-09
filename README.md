# DAE-FF
A Neural Network-based Feature Extraction Method to Facilitate Citation Screening for Systematic Reviews. 


## Installation

This code was tested with Python 3.6 on Ubuntu 18.04 and Windows 10.

```
% pip install -r requirements.txt
```

You need to also download data for `nltk`:

```python
import nltk
nltk.download(['punkt', 'stopwords', 'averaged_perceptron_tagger'])
```

