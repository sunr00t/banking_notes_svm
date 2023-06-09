# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# Importando os dados e renomeando os cabecalhos
data = pd.read_csv("./src/banking_notes.csv", header=0, names=['variance', 'skewness', 'curtosis', 'entropy', 'classification'])
print(data.head())
# %%
# Gráfico de dispersão
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
xs = data[data.columns[0]].tolist()
ys = data[data.columns[1]].tolist()
zs = data[data.columns[2]].tolist()
gs = data[data.columns[3]].tolist()
ax.scatter(xs, ys, zs, gs, alpha=0.3)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

#indexação dos valores
attributes = data[data.columns[0:4]]
classification = data['classification']

# %%
# Treinando utilizando 30% dos dados

train, test, train_labels, test_labels = train_test_split(attributes,classification, test_size=0.30, random_state=28)
# %%
# Iniciando Classificação de Gaussian Naive Bayes
gnb = GaussianNB()

# %%
# Classificando o modelo
model = gnb.fit(train, train_labels)
# %%
# Previsões
preds = gnb.predict(test)

# Avaliando a eficacia
accuracy = accuracy_score(test_labels, preds)
print("NaiveBayes Score: -> ", accuracy)

# Iniciando o Classificador SVC
sv = SVC()

# Treinando o Classificador
sv_model = sv.fit(train, train_labels)

# Fazendo Previsoes
sv_preds = sv.predict(test)

# Avaliando a Eficacia
sv_accuracy = accuracy_score(test_labels, sv_preds)

print("SVM Score -> ", sv_accuracy)
