# PROGETTO STATISTICA a.a 2023/2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import zipfile

# FASE 1 - CARICARE IL DATASET

#data = pd.read_csv('smoking_driking_dataset_Ver01.csv')

url="https://www.kaggle.com/api/v1/datasets/download/sooyoungher/smoking-drinking-dataset"
r = requests.get(url)  


with open('dataset.zip', 'wb') as f:
    f.write(r.content)

with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

data = pd.read_csv('smoking_driking_dataset_Ver01.csv')

#-----------------------------------------------------------------------------------------------------------------------

# FASE 2 - PRE-PROCESSING

# Intervallo di righe da rimuovere
start_index = 30000
end_index = 991346

# Rimuovere le righe dall'indice start_index a end_index
data = data.drop(data.index[start_index:end_index])

# Rimuovere le colonne non necessarie
data = data.drop(columns=['sight_left', 'sight_right', 'hear_left', 'hear_right'])

# Verifica valori NaN
data = data.dropna(axis=0)

# Sostituisco possibili valori della variabile target
data['DRK_YN'].replace("Y", 1, inplace=True)
data['DRK_YN'].replace("N", 0, inplace=True)

#-----------------------------------------------------------------------------------------------------------------------

# FASE 3 - EXPLORATORY DATA ANALYSIS (EDA)

# Dimesione del dataset
print(data.shape)

# Informazioni sulle colonne del dataset
print(data.info())

# Media del livello di colesterolo
dataChole = data["tot_chole"].mean()
print(f"Media livelli di colesterolo: {dataChole}\n")

# Ispezione del dataset
meanAge = data["age"].mean()
medianAge = data["age"].median()
print(f"Età media: {meanAge}")
print(f"Mediana età: {medianAge}")

meanWeight = data["weight"].mean()
medianWeight = data["weight"].median()
print(f"Peso medio: {meanWeight}")
print(f"Mediana peso: {medianWeight}")

meanHeight = data["height"].mean()
medianHeight = data["height"].median()
print(f"Altezza media: {meanHeight}")
print(f"Mediana altezza: {medianHeight}")


# PLOTTING DEI GRAFICI

# Grafico a barre dello stato di fumo
smoking_counts = data['SMK_stat_type_cd'].value_counts()
plt.bar(smoking_counts.index, smoking_counts.values)
plt.title('Stato di fumo')
plt.show()

# Grafico a barre del genere
gender_counts = data['sex'].value_counts()
plt.bar(gender_counts.index, gender_counts.values)
plt.title('Genere')
plt.show()

# Istogramma del livello di colesterolo medio
sns.histplot(data["tot_chole"], bins=80, color='red', edgecolor='black', kde=True)
plt.title('Istogramma del livello di colesterolo medio')
plt.xlim(0, 400)
plt.show()

data['SMK_stat_type_cd'] = data['SMK_stat_type_cd'].astype(int) 

# Correlazione tra persone che fumano e che bevono
sns.countplot(data=data, x='SMK_stat_type_cd', hue='DRK_YN')
plt.title('Distribuzione del Livello di Fumo e Consumo di Alcool')
plt.xlabel('Livello di Fumo (1: Non fumatore, 2: Ex-fumatore, 3: Fumatore)')
plt.ylabel('Numero')
plt.legend(title='Beve (1: Sì, 0: No)')
plt.show()

print("------------------------------------------------------------------------------------------------------")

# Creiamo dataset con i valori numerici
data_num = data[['age','height','weight','waistline','SBP','DBP','BLDS','tot_chole','HDL_chole','LDL_chole',
                 'triglyceride','hemoglobin','urine_protein', 'serum_creatinine',
                 'SGOT_AST', 'SGOT_ALT', 'gamma_GTP', 'SMK_stat_type_cd']]

# Stampa principali dati statici sul dataset numerico
print(data_num.describe())
print(data_num.info())

# Matrice di correlazione
Corr = data_num.corr()
print(Corr)

# Visualizzazione della matrice di correlazione
plt.matshow(data_num.corr(), vmin=-1, vmax=1)
plt.xticks(np.arange(0, data_num.shape[1]), data_num.columns, rotation=45)
plt.yticks(np.arange(0, data_num.shape[1]), data_num.columns)
plt.title("Matrice di correlazione")
plt.colorbar()
plt.show()

print("--------------------------------------------------------------------------------------------------------")

# FASE 4 - SPLITTING

# Selezione delle feature e della variabile target
data_final = data[['age','height','weight','waistline','SBP','DBP','BLDS','tot_chole','HDL_chole','LDL_chole',
                 'triglyceride','hemoglobin','urine_protein', 'serum_creatinine',
                 'SGOT_AST', 'SGOT_ALT', 'gamma_GTP', 'SMK_stat_type_cd','DRK_YN']]

from sklearn import model_selection, svm

np.random.seed(35)

# Fase di splitting del training set dal test set
data_train, data_test = model_selection.train_test_split(data_final, train_size=0.78)

# Fase di splitting del training set dal validation set
data_train, data_val = model_selection.train_test_split(data_train, train_size=0.73)

#-----------------------------------------------------------------------------------------------------------------------

# FASE 5 - REGRESSIONE

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

# Stampo indici di correlazione delle due coppie di attributi scelti
indiceCorr = data['SBP'].corr(data['DBP'])
indiceCorr2 = data['tot_chole'].corr(data['LDL_chole'])
print(f"Indice di correlazione tra SBP e DBP: {indiceCorr}")
print(f"Indice di correlazione tra colesterolo e LDL: {indiceCorr2}")

# Regressione lineare tra SBP e DBP
X = data_train["SBP"].values.reshape(-1, 1)
y = data_train["DBP"].values.reshape(-1,1)

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.xlabel('SBP')
plt.ylabel('DBP')
plt.title('Regressione lineare')
plt.show()

# Coefficiente r^2
r_squared = r2_score(y_pred, y)
print(f"R^2: {r_squared}")

# Analisi di normalità dei residui
residuals = y_pred - y

# Calcolo MSE
mse = np.mean(residuals**2)
print(f"MSE: {mse}")

# QQ-plot dei residui
plt.figure(figsize=(6, 6))
sm.qqplot(residuals, line='s')
plt.title('QQ Plot dei Residui')
plt.xlabel('Quantili teorici')
plt.ylabel('Quantili dei residui')
plt.grid(True)
plt.show()

# Regressione lineare tra tot_chole e LDL_chole
X = data_train["tot_chole"].values.reshape(-1,1)
y = data_train["LDL_chole"].values.reshape(-1,1)

reg = LinearRegression()
reg.fit(X, y)

y_pred = reg.predict(X)

plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.xlabel('Livello di colesterolo')
plt.ylabel('LIvello di LDL')
plt.title('Regressione lineare')
plt.xlim(0, 400)
plt.ylim(0, 300)
plt.show()

# Coefficiente r^2
r_squared = r2_score(y_pred, y)
print(f"R^2: {r_squared}")

# Analisi di normalità dei residui
residuals = y_pred - y

# Calcolo MSE
mse = np.mean(residuals**2)
print(f"MSE: {mse}")

# QQ-plot dei residui
plt.figure(figsize=(6, 6))
sm.qqplot(residuals, line='s')
plt.title('QQ Plot dei Residui')
plt.xlabel('Quantili teorici')
plt.ylabel('Quantili dei residui')
plt.grid(True)
plt.show()

# Elimino outlier
residuals = residuals[residuals > -3000]

# QQ-plot dei residui senza outlier
plt.figure(figsize=(6, 6))
sm.qqplot(residuals, line='s')
plt.title('QQ Plot dei Residui')
plt.xlabel('Quantili teorici')
plt.ylabel('Quantili dei residui')
plt.grid(True)
plt.show()

#-----------------------------------------------------------------------------------------------------------------------

# FASE 6 - ADDESTRAMENTO DEL MODELLO

# Selezione delle feature e della variabile target per il training
X = data_train[['age','height','weight','waistline','SBP','DBP','BLDS','tot_chole','HDL_chole','LDL_chole',
                 'triglyceride','hemoglobin','urine_protein', 'serum_creatinine',
                 'SGOT_AST', 'SGOT_ALT', 'gamma_GTP', 'SMK_stat_type_cd']]

y = data_train[['DRK_YN']]

# Selezione delle feature e della variabile target per la validazione
X_val = data_val[['age','height','weight','waistline','SBP','DBP','BLDS','tot_chole','HDL_chole','LDL_chole',
                 'triglyceride','hemoglobin','urine_protein', 'serum_creatinine',
                 'SGOT_AST', 'SGOT_ALT', 'gamma_GTP', 'SMK_stat_type_cd']]

y_val = data_val[['DRK_YN']]

#.............................................................................................
# LogisticRegression non converge
"""
modelReg = LogisticRegression()
modelReg.fit(X, y['DRK_YN'])

y_pred = modelReg.predict(X_val)
print(y_pred)

# Misurare errore di misclassificazione
ME = np.sum(y_pred != y_val["DRK_YN"])
print(f"ME: {ME}")

MR = ME / len(y_pred)
print(f"MR: {MR}")

Acc = 1-MR
print(f"Acc: {Acc}")
"""
#............................................................................................

# Addestramento del modello SVM con kernel rbf
modelRBF = svm.SVC(kernel="rbf", gamma=1)
modelRBF.fit(X, y['DRK_YN'])

y_pred = modelRBF.predict(X_val)
print(y_pred)

# Misurare errore di misclassificazione
ME = np.sum(y_pred != y_val["DRK_YN"])
print(f"ME: {ME}")

# Tasso di errore di classificazione
MR = ME / len(y_pred)
print(f"MR: {MR}")

# Accuratezza
Acc = 1-MR
print(f"Acc: {Acc}")

#............................................................................................

# Addestramento del modello SVM con kernel linear
modelLinear = svm.SVC(kernel="linear")
modelLinear.fit(X, y['DRK_YN'])

y_pred = modelLinear.predict(X_val)
print(y_pred)

# Misurare errore di misclassificazione
ME = np.sum(y_pred != y_val["DRK_YN"])
print(f"ME: {ME}")

# Tasso di errore di classificazione
MR = ME / len(y_pred)
print(f"MR: {MR}")

# Accuratezza
Acc = 1-MR
print(f"Acc: {Acc}")

#-----------------------------------------------------------------------------------------------------------------------

# FASE 7 - HYPERPARAMETER TUNING

# Lista per memorizzare l'accuratezza per diversi gradi del polinomio
listaAcc = []

# Prova con diversi gradi del polinomio
for d in range(1, 6):

    modelSVC = svm.SVC(C=2, kernel="poly", degree=d)

    modelSVC.fit(X, y["DRK_YN"])

    X_val = data_val[['age','height','weight','waistline','SBP','DBP','BLDS','tot_chole','HDL_chole','LDL_chole',
                 'triglyceride','hemoglobin','urine_protein', 'serum_creatinine',
                 'SGOT_AST', 'SGOT_ALT', 'gamma_GTP', 'SMK_stat_type_cd']]

    y_val = data_val[['DRK_YN']]

    y_pred = modelSVC.predict(X_val)
    print(y_pred)

    ME = np.sum(y_pred != y_val["DRK_YN"])

    MR = ME / len(y_pred)

    Acc = 1 - MR
    listaAcc.append(Acc)

# Visualizzazione dell'accuratezza per diversi gradi del polinomio
plt.plot(range(1, 6), listaAcc, marker='o')
plt.title('Accuratezza vs Grado del Polinomio')
plt.xlabel('Grado del Polinomio')
plt.ylabel('Accuratezza')
plt.xticks(range(1, 6))
plt.show()

#-----------------------------------------------------------------------------------------------------------------------

# FASE 8 - VALUTAZIONE DELLE PERFORMANCE

from sklearn.metrics import (accuracy_score, confusion_matrix)

# Selezione delle feature e della variabile target per il test set
X_test = data_test[['age','height','weight','waistline','SBP','DBP','BLDS','tot_chole','HDL_chole','LDL_chole',
                   'triglyceride','hemoglobin','urine_protein', 'serum_creatinine',
                   'SGOT_AST', 'SGOT_ALT', 'gamma_GTP', 'SMK_stat_type_cd']]

y_test = data_test[['DRK_YN']]

modelTest = svm.SVC(C=2, kernel="poly", degree=5)

modelTest.fit(X, y["DRK_YN"])

# Predizione sul test set
y_pred = modelTest.predict(X_test)

# Calcolo delle metriche di valutazione
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuratezza: {accuracy}")

print("Matrice di confusione:")
print(conf_matrix)

#-----------------------------------------------------------------------------------------------------------------------

# FASE 9 - STUDIO STATISTICO SUI RISULTATI DELLA VALUTAZIONE

from scipy import stats

k = 10
accuracies = []

# Ripetizione del training e della valutazione k volte
for i in range(k):

    # Split dei dati
    data_train, data_test = model_selection.train_test_split(data_final, train_size=0.78)
    data_train, data_val = model_selection.train_test_split(data_train, train_size=0.73)

    # Addestro il modello sul train set
    X_train = data_train[['age','height','weight','waistline','SBP','DBP','BLDS','tot_chole','HDL_chole','LDL_chole',
                          'triglyceride','hemoglobin','urine_protein', 'serum_creatinine',
                          'SGOT_AST', 'SGOT_ALT', 'gamma_GTP', 'SMK_stat_type_cd']]
    y_train = data_train[['DRK_YN']]

    modelSVC = svm.SVC(C=2, kernel="poly", degree=5)
    modelSVC.fit(X_train, y_train['DRK_YN'])

    # Provo il modello sul test set
    X_test = data_test[['age','height','weight','waistline','SBP','DBP','BLDS','tot_chole','HDL_chole','LDL_chole',
                      'triglyceride','hemoglobin','urine_protein', 'serum_creatinine',
                      'SGOT_AST', 'SGOT_ALT', 'gamma_GTP', 'SMK_stat_type_cd']]
    y_test = data_test[['DRK_YN']]
    y_pred = modelSVC.predict(X_test)

    ME = np.sum(y_pred != y_test["DRK_YN"])
    MR = ME / len(y_pred)
    Acc = 1 - MR
    accuracies.append(Acc)

# Analisi statistica descrittiva
accuracies = np.array(accuracies)
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)

print(f"Accuratezza media: {mean_acc}")
print(f"Deviazione Standard: {std_acc}")

# Visualizzare i risultati con un istogramma
plt.hist(accuracies, bins=10, edgecolor='black')
plt.title('Istogramma accuratezza')
plt.xlabel('Accuratezza')
plt.ylabel('Frequenza')
plt.show()

# Boxplot delle accuratezze
plt.figure(figsize=(8, 6))
plt.boxplot(accuracies)
plt.title('Boxplot delle accuratezze')
plt.ylabel('Accuratezza')
plt.show()

# Calcolo intervallo di confidenza
k = len(data)
conf_interval = stats.t.interval(0.95, df=k-1, loc=mean_acc, scale=std_acc/np.sqrt(k))
print(f"Intervallo di confidenza 95% : {conf_interval}")
