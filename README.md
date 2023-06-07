![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

# banking_notes_svm

Using Machine Learning (Scikit Learn) to building data analysis, training and evaluating a model Bank Note Authentication.

## Getting Started
**Dataset** ```/src/banking_notes.csv```

Data were extracted from images that were taken from genuine and forged banknote-like specimens. For digitization, an industrial camera usually used for print inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extract features from images.

[UCI - Banknote Authentication](https://archive.ics.uci.edu/ml/datasets/banknote+authentication#)

- Owner of database: **Volker Lohweg** (University of Applied Sciences, Ostwestfalen-Lippe, volker.lohweg '@' hs-owl.de)
- Donor of database: **Helene Dörksen** (University of Applied Sciences, Ostwestfalen-Lippe, helene.doerksen '@' hs-owl.de)

**Date received:** August, 2012

### Attribute Information

  1. Variance of Wavelet Transformed image (continuous)
  2. Skewness of Wavelet Transformed image (continuous)
  3. Curtosis of Wavelet Transformed image (continuous)
  4. Entropy of image (continuous)
  5. Class (integer)

### Dependencies

- Python3
- Pip ```sudo apt install python3-pip```

### Clone this repository

```git clone https://github.com/sunr00t/banking_notes_svm```

### Install requirements

[Scikit Learn](https://scikit-learn.org/stable/install.html)
[Pandas](https://pandas.pydata.org/docs/getting_started/install.html)

```pip install -r requirements.txt```

### Running

```python ./main.py```
