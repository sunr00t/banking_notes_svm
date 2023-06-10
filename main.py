# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Importando os dados e renomeando os cabecalhos
data = pd.read_csv("./src/banking_notes.csv", header=0, names=['variance', 'skewness', 'curtosis', 'entropy', 'classification'])
print(data.head())
data.shape

# %%
# Gráfico de Correlação
sns.heatmap(data.corr(), annot=True, cmap='rocket_r',cbar=True,linewidths=0.2)
plt.show()

classification = data['classification']
print(data['classification'].value_counts())
# %%
# Gráfico dos Atributos
ax = sns.countplot(x = classification, data = data)
sns.displot(data["variance"], height = 3, aspect = 1.5)
plt.xlabel("variance")
sns.displot(data["curtosis"], height = 3, aspect = 1.5)
plt.xlabel("curtosis")
sns.displot(data["skewness"], height = 3, aspect = 1.5)
plt.xlabel("skewness")
sns.displot(data["entropy"], height = 3, aspect = 1.5)
plt.xlabel("entropy")
plt.show()
# %%
# Treinando os dados
attributes = data[data.columns[0:4]] #Indexação dos Atributos
# Treinando com 30% dos dados
# 1ª Seq x_train = treino ~ x_test = testes
# 2ª Seq y_train = treino ~ y_test = testes
x_train, x_test, y_train, y_test = train_test_split(attributes,classification, test_size=0.30, random_state=28)
gnb = GaussianNB() # Funcao de classificação de Gaussian Naive Bayes
model = gnb.fit(x_train, y_train)

# %%
# Score dos Modelos
preds = gnb.predict(x_test)
accuracy = accuracy_score(y_test, preds) # Avaliando a eficácia dos testes
print("NaiveBayes Score: -> ", accuracy)

sv = SVC() # Função de classificação de SVC
sv_model = sv.fit(x_train, y_train) # Treinando o modelo
sv_preds = sv.predict(x_test) # Fazendo Previsoes
sv_accuracy = accuracy_score(y_test, sv_preds) # Avaliando a Eficacia
print("SVM Score -> ", sv_accuracy)

#coeficiente de determinação 
print ("Score Modelo Treinado -> ", sv_model.score(x_train, y_train)) 
print ("Score do Modelo de Testes -> ", sv_model.score(x_test, y_test))
