# banking_notes_svm

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-%157F1F.svg?style=for-the-badge&logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-%23316192.svg?style=for-the-badge&logo=docker&logoColor=white)

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

- Docker
- Python3
- Pip
- Flask
- Flask_Cors
- Joblib
- Pandas
- Scikit Learn
- Matplotlib
- Seaborn

### Clone this repository

```git clone https://github.com/sunr00t/banking_notes_svm```

### Running

1. ```docker build -t banking_notes .```
2. ```docker run -p 5000:5000 banking_notes```
3. Access webservice on ```http://localhost:5000```

### Routes

1. ```/``` ```http://localhost:5000/```
2. ```/validate``` ```http://localhost:5000/validade```

### Endpoint explain: 

```http://localhost:5000 [GET]```

```
2023-06-16 00:43:14.414213
````

```http://localhost:5000/validade [POST]```

```json
{
 "variance": -2.6864,
 "curtosis": -0.097265,
 "skewness": 0.61663,
 "entropy": 0.061192
}
```

```RESPONSE```

```json
{
 "classification": "1"
}
```
