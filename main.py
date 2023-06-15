# %%
# Importando os dados
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("./src/banking_notes.csv", header=0, names=['variance', 'skewness', 'curtosis', 'entropy', 'classification'])
classification = dataset['classification']
# %%
# Gráficos de Correlação
def show_plots():
  sns.heatmap(dataset.corr(), annot=True, cmap='rocket_r',cbar=True,linewidths=0.2)
  plt.show()
  sns.pairplot(dataset, hue='classification')
  plt.show()
  print(dataset['classification'].value_counts())
  
  sns.countplot(x = classification, data = dataset)
  sns.displot(dataset["variance"], height = 3, aspect = 1.5)
  plt.xlabel("variance")
  sns.displot(dataset["curtosis"], height = 3, aspect = 1.5)
  plt.xlabel("curtosis")
  sns.displot(dataset["skewness"], height = 3, aspect = 1.5)
  plt.xlabel("skewness")
  sns.displot(dataset["entropy"], height = 3, aspect = 1.5)
  plt.xlabel("entropy")
  plt.show()
# %%
# Treinando os dados
attributes = dataset[dataset.columns[0:4]] #Indexação dos Atributos
# Treinando com 30% dos dados
# 1ª Seq x_train = treino ~ x_test = testes
# 2ª Seq y_train = treino ~ y_test = testes
x_train, x_test, y_train, y_test = train_test_split(attributes,classification, test_size=0.30, random_state=28)

# Funcao de classificação de Gaussian Naive Bayes
gnb = GaussianNB() 

# Função de classificação de SVC
sv = SVC() 
model = gnb.fit(x_train, y_train)

# %%
# Treinando o modelo
sv_model = sv.fit(x_train, y_train) 
# Fazendo Previsoes
sv_preds = sv.predict(x_test)

# Score dos Modelos
def show_scores():
  y_pred = gnb.predict(x_test)
  # Avaliando a eficácia dos testes
  accuracy = accuracy_score(y_test, y_pred) 
  print("NaiveBayes Score: -> ", accuracy)
  # Avaliando a Eficacia
  sv_accuracy = accuracy_score(y_test, sv_preds) 
  print("SVM Score -> ", sv_accuracy)
  # Coeficiente de determinação 
  print ("Score Modelo Treinado -> ", sv_model.score(x_train, y_train)) 
  print ("Score do Modelo de Testes -> ", sv_model.score(x_test, y_test))

# Simulando User Entries
def user_entries(variance, curtosis, skewness, entropy ):
  user_entry = [[variance, curtosis, skewness, entropy]]
  result = sv_model.predict(user_entry)
  return result[0]