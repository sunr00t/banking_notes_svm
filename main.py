# %%
# Importando os dados
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from flask import Flask

app = Flask(__name__)
@app.route("/")
def home():
  return "Hello Flask"
  
if __name__ == "__main__":
  app.run(debug=True)
  
data = pd.read_csv("./src/banking_notes.csv", header=0, names=['variance', 'skewness', 'curtosis', 'entropy', 'classification'])
classification = data['classification']
# %%
# Gráficos de Correlação
def show_plots():
  sns.heatmap(data.corr(), annot=True, cmap='rocket_r',cbar=True,linewidths=0.2)
  plt.show()
  sns.pairplot(data, hue='classification')
  plt.show()
  print(data['classification'].value_counts())
  
  sns.countplot(x = classification, data = data)
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
  if result == 1:
    response = print('Nota Verdadeira')
  elif result == 0:
    response = print('Nota Falsa')
  else:
    response = print('Informe os valores')
  return response

home()