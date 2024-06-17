# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 09:27:11 2024

@author: chau
"""

import pandas as pd
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib_inline
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

data_target13 = pd.read_csv('data_target13.csv',sep=',',encoding="latin-1")
data_target13.head()

data_target13.info()

# On renomme les variables

dictionnaire ={'masse':'Poids (kg)',
               'C02': 'CO2',
               'dimension': 'Dimension (mm)',
               'cylindrÃ©e':'Cylindrée (cm3)',
               'puissance':'Puissance (ch)',
               'autonomie Ã©lectrique': 'Autonomie electrique (km)',
               'fuel mode' : 'fuel_mode',
               'fuel type': 'fuel_type'}

data_target13 = data_target13.rename(dictionnaire, axis = 1)

data_target13.head() 



colonnes_selectionnees = ['Poids (kg)','Dimension (mm)','Cylindrée (cm3)','Puissance (ch)','Autonomie electrique (km)']


#On partage le dataset en variables explicatives (masse, dimension, cylindrée, puissance, autonomie éléctrique, fuel mode,
#fuel type) et en variable cible :CO2

feats = data_target13.drop('CO2', axis = 1)
target = data_target13['CO2'] # Variable cible
feats.info()

# Séparation du données en jeu d'entrainement et de test de sorte que le jeu de test contienne 20% des données du dataset:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.20, random_state = 42)

# Séparation des colonnes catégorielles des colonnes numériques et création des dataframe correspondants 
# sur les jeux d'entrainement et de test

num = ['Poids (kg)','Dimension (mm)','Cylindrée (cm3)','Puissance (ch)','Autonomie electrique (km)'] # num= liste des variables explicatives numériques
num_train = X_train[num]
num_test = X_test[num]

cat = ['fuel_mode','fuel_type'] # cat = liste des variables explicatives catégorielles
cat_train = X_train[cat]
cat_test = X_test[cat]

# Encodage des variables catégorielles

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder() # creation de 'encoder' un objet OneHotEncoder

# fit and transform des colonnes de cat_train columns / création d'un nouveau dataframe à partir d'un numpy array
one_cat_train_array = encoder.fit_transform(cat_train).toarray() 
one_cat_train_df = pd.DataFrame(one_cat_train_array, columns=encoder.get_feature_names_out())
one_cat_train_df.head()

# fit and transform des colonnes de cat_test columns / création d'un nouveau dataframe à partir d'un numpy array
one_cat_test_array = encoder.transform(cat_test).toarray() 
one_cat_test_df = pd.DataFrame(one_cat_test_array, columns=encoder.get_feature_names_out())
one_cat_test_df.head(10)

# Standardisation des colonne numériques

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
num_train_scaled_array = sc.fit_transform(num_train)
num_train_scaled_df = pd.DataFrame(num_train_scaled_array, columns=sc.get_feature_names_out())
num_test_scaled_array = sc.transform(num_test)
num_test_scaled_df = pd.DataFrame(num_test_scaled_array, columns=sc.get_feature_names_out())

# Reconstitution des jeux d'entraînement et de test après encodage des variables catégorielles 
# en concaténant num_train avec cat_train et num_test avec cat_test.

X_train_new = pd.concat([num_train_scaled_df, one_cat_train_df.set_index(num_train_scaled_df.index)], axis=1)
X_test_new = pd.concat([num_test_scaled_df, one_cat_test_df.set_index(num_test_scaled_df.index)], axis=1)
X_test_new.head(10)

#Affichage des dimensions des jeux reconstitués.
print("Train Set:", X_train_new.shape) # Pour afficher les dimensions du train set recontitué
print("Test Set:", X_test_new.shape) # Pour afficher les dimensions du test set recontitué


# Entrainement sur le modèle de Random Forest

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest.fit(X_train_new, y_train)

# Joblib permet d'entraîner le modèle rapidement sans attendre dans streamlit:
import joblib
joblib.dump(random_forest,"model_rf")

#Accuracy de modèle Random Forest
y_pred_reg = random_forest.predict(X_test_new)

print('Coefficient de détermination du modèle sur train:', random_forest.score(X_train_new, y_train))
print('Coefficient de détermination du modèle sur test:', random_forest.score(X_test_new, y_test))


import matplotlib.pyplot as plt

fig = plt.figure(figsize = (10,10))
pred_test = random_forest.predict(X_test_new)
plt.scatter(pred_test, y_test, c='green')

plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), color = 'red')
plt.xlabel("prediction")
plt.ylabel("vrai valeur")
plt.title('Random Forest pour la prédiction emission de CO2 des véhicules')

plt.show()


#Randomforest regressor et résidus:
from sklearn.ensemble import RandomForestRegressor 
regressor_random_forest = RandomForestRegressor(random_state=42) 
  
regressor_random_forest.fit(X_train_new, y_train)

import joblib
joblib.dump(regressor_random_forest,"model_RF")

#Accuracy de Random Forest:
print(regressor_random_forest.score(X_train_new,y_train))
print(regressor_random_forest.score(X_test_new,y_test))

regressor_random_forest = RandomForestRegressor(random_state=42) 
  
regressor_random_forest.fit(X_train_new, y_train)
y_pred_random_forest = regressor_random_forest.predict(X_test_new)
y_pred_random_forest_train = regressor_random_forest.predict(X_train_new)


#Résidus random forest:
#mae_random_forest_train = mean_absolute_error(y_train,y_pred_random_forest_train)
#mse_random_forest_train = mean_squared_error(y_train,y_pred_random_forest_train,squared=True)
#rmse_random_forest_train = mean_squared_error(y_train,y_pred_random_forest_train,squared=False)

#mae_random_forest_test = mean_absolute_error(y_test,y_pred_random_forest)
#mse_random_forest_test = mean_squared_error(y_test,y_pred_random_forest,squared=True)
#rmse_random_forest_test = mean_squared_error(y_test,y_pred_random_forest,squared=False)
 
#data = {'MAE train': [mae_decision_tree_train, mae_random_forest_train],
#'MAE test': [mae_decision_tree_test, mae_random_forest_test],
# 'MSE train': [mse_decision_tree_train,mse_random_forest_train],
#'MSE test': [mse_decision_tree_test,mse_random_forest_test],
#'RMSE train': [rmse_decision_tree_train, rmse_random_forest_train],
#'RMSE test': [rmse_decision_tree_test, rmse_random_forest_test]}
#df = pd.DataFrame(data, index = ['Decision Tree', 'Random Forest '])
#df.head()

#Un premier moyen devaluer la pertinence du modèle est de le comparer à des modèles Benchmark ou des modèles naifs: 
 
#from sklearn.metrics import r2_score
#y_pred=np.ones(len(y_test))*y_train.mean()
#print('r2:',format(round(r2_score(y_test,y_pred),2)))
#print('MAE naif:',format(round(mean_absolute_error(y_test,y_pred),2)))
#print('RMSE naif:',format(round(np.sqrt(mean_squared_error(y_test,y_pred)),2)))


#Afficher un graphique de dispersion qui compare les valeurs prédites aux valeurs réelles pour le modèle Random Forest:
#g=sns.relplot(x=y_test,y=y_pred_random_forest)
#g.ax.axline(xy1=(20,20), xy2=(100,100), color='b', dashes=(5,2));

#Analysons plus près les résidus:
#result=pd.DataFrame()
#y_pred=y_pred_random_forest
#result['y_test']=y_test
#result['y_pred']=y_pred
#result['residus']=result.y_test - result.y_pred
#quantiles=result.residus.quantile([0.1,0.25,0.75,0.9])
#print(quantiles)

#Graphiques des résidus:
#sns.relplot(data=result,x='y_test', y='residus',alpha=0.5, height=8, aspect=10/8)
#plt.plot([0,result.y_test.max()],[0,0], 'r-.,')
#plt.plot([0,result.y_test.max()],[quantiles[0.10],quantiles[0.10]],'y--',label="80% des résidus présents dans cet intervalle")
#plt.plot([0,result.y_test.max()],[quantiles[0.90],quantiles[0.90]],'y--')
#plt.plot([0,result.y_test.max()],[quantiles[0.25],quantiles[0.25]],'y--',label="50% des résidus présents dans cet intervalle")
#plt.plot([0,result.y_test.max()],[quantiles[0.75],quantiles[0.75]],'y--')
#plt.xlim(0,result.y_test.max()+10)
#plt.xlabel('y_test')
#plt.ylabel('test résidus')
#plt.title('Résidus')
#plt.legend()
#plt.show();


#Résidus en valeur absolue:
#residus=abs(y_test - y_pred_random_forest)
#residus.name='Résidus abs'


#Description des résidus sont très peu significatifs
#residus.describe()


#Pourcentage d'observation ayant une erreur supérieur à 5
#len(residus[residus>5])/len(residus)

#L'ECDF Plot permet d'afficher la fonction de répartition Empirique Cumulative d'une variable
#sns.ecdfplot(residus)
#plt.plot([0,60],[0.7,0.7], 'y--')
#plt.show();







   