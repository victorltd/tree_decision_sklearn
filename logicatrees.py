import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn.tree import export_graphviz

banco= pd.read_csv('logica.csv')

previsores = banco.iloc[:,0:4].values
classe= banco.iloc[:,4].values


#fazer a transformacao de valores categoricos em numeros
labelencoder= LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])

previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])
"""
previsores[:,6] = labelencoder.fit_transform(previsores[:,6])
previsores[:,8] = labelencoder.fit_transform(previsores[:,8])
previsores[:,9] = labelencoder.fit_transform(previsores[:,9])
previsores[:,11] = labelencoder.fit_transform(previsores[:,11])
previsores[:,13] = labelencoder.fit_transform(previsores[:,13])
previsores[:,14] = labelencoder.fit_transform(previsores[:,14])
previsores[:,16] = labelencoder.fit_transform(previsores[:,16])
previsores[:,18] = labelencoder.fit_transform(previsores[:,18])
previsores[:,19] = labelencoder.fit_transform(previsores[:,19])
"""

x_treinamento, x_teste, y_treinamento, y_teste= train_test_split(previsores,
classe, test_size=0.2, random_state=1)

arvore = DecisionTreeClassifier()
arvore.fit(x_treinamento, y_treinamento)

export_graphviz(arvore, out_file= 'tree.dot')

previsoes= arvore.predict(x_teste)

confusao= confusion_matrix(y_teste, previsoes)
taxa_acertos= accuracy_score(y_teste, previsoes)
taxa_erro= 1-taxa_acertos
