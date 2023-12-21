'''
Importación de librerías
'''
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


'''
Datos
'''
path = "./PenguinData"
penguins_size = path + os.sep + "penguins_size.csv"

df = pd.read_csv(penguins_size)
display(df)

# Eliminación de las filas que contengan al menos un valor "N. A."
df = df.dropna()

# Creación de un codificador que me permita asignar números a variables "categóricas". 2º)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])
df['island'] = le.fit_transform(df['island'])
df['sex'] = le.fit_transform(df['sex'])
display(df)

y = df.species.values.astype(int)
caract_cols = ['island', 'culmen_length_mm','culmen_depth_mm','flipper_length_mm','body_mass_g','sex']
X_all = df[caract_cols].values
print(X_all.shape)

'''
List of datasets and their names included in the experimental study
'''
datasets = [(X_all,y)]  #y es la etiqueta   datasets es el vector de caracteristica
dataset_names = ["Data All"]
print(y.shape)

'''
Definición del espacio de búsqueda para la optimización de los parámetros de SVM
'''
C_range = np.logspace(-2,10,10)  # se define el rango de C
gamma_range = np.logspace(-9,3,10)  # se define el rango de gamma 
param_grid_svm = dict(gamma=gamma_range, C=C_range)
nested_cv = 5
grid_svm = GridSearchCV(SVC(), param_grid=param_grid_svm, cv=nested_cv) 


# Aquí se muestra el rango de valores a considerar
C_range,gamma_range


'''
Definición del espacio de búsqueda para MLP
'''
alpha_range = np.logspace(-5, -1, 5)
hidden_layer_sizes_range=[(50,),(100,),(200,),(500,),(1000,)]

param_grid_mlp = dict(alpha=alpha_range, hidden_layer_sizes=hidden_layer_sizes_range)


grid_mlp = GridSearchCV(MLPClassifier(max_iter=1000,
                                      early_stopping=True), param_grid=param_grid_mlp, cv=nested_cv)


'''
Conjunto de clasificadores usados, así como sus nombres.
'''
cls_names = ["SVM","MLP"]

classifiers = [
    make_pipeline(StandardScaler(), grid_svm),
    make_pipeline(StandardScaler(), grid_mlp)]


# Método que ejecuta los clasificacodres y devuelve las etiquetas predichas correspondientes.
def predictions(model,X_train,y_train,X_test,y_test):    
    
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    return y_test, y_pred


def predictions_model(X_train,y_train,X_test, y_test,model):
    print('\t'+str(model)[:20], end=' - ')
    y_test,preds = predictions(model,X_train,y_train,X_test,y_test)
    print('OK')
        
    return y_test,preds


from sklearn.model_selection import train_test_split

def run_all_save(filename, train_size): # le paso por parametro los porcentajes de entrenamiento
    all_preds = {}

    for dataset,dataset_name in zip(datasets, dataset_names):
        print(dataset_name)
        X,y = dataset
        
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=train_size)
        

        for model,cls_name in zip(classifiers,cls_names):
            print(cls_name)
            y_test,preds = predictions_model(X_train,y_train,X_test,y_test,model)
            all_preds[(dataset_name,cls_name)]=(y_test,preds)
    
    
    all_preds["cls_names"]=cls_names
    all_preds["dataset_names"]=dataset_names

    
    with open(filename, 'wb') as fp:
         pickle.dump(all_preds, fp)    



'''
All the predictions are going to be saved in a Python dictionary for 
further analysis.
'''
filename = 'PrediccionesPenguin.obj'


# Función que debe evaluar los resultados de clasificación.
def evalua(y_test, y_pred):
    # Inicializa un contador que se encarga de almacenar las predicciones correctas (aciertos)
    aciertos = 0

    # Itera a través de los elementos de y_test y y_pred para comparar los resultados
    for i in range(len(y_test)):
        # Comprueba si la predicción es correcta
        if y_test[i] == y_pred[i]:
            aciertos += 1

    # Calcula la precisión dividiendo las predicciones correctas entre el total de elementos en y_test
    precision = aciertos / len(y_test)
    return precision # retorna la tasa de acierto (precision)


def conf_mat_df(cm,labels):
    return (pd.DataFrame(cm,index=labels, columns=labels)
          .rename_axis("actual")
          .rename_axis("predicted", axis=1))


def get_results(filename):
    with open(filename, 'rb') as fp:
        all_preds = pickle.load(fp)

    cls_names = all_preds.pop("cls_names")
    dataset_names = all_preds.pop("dataset_names")

    data_cls_pairs = list(all_preds.keys())
    data_cls_pairs.sort()

    results = {}


    acc_df = pd.DataFrame(index=dataset_names, columns=cls_names)

    ## A DataFrame is created to store the accuracy in each clase
    for dataset in dataset_names:
        results[(dataset,"acc")] = pd.DataFrame(columns=cls_names)


    for dataset_name,cls_name in data_cls_pairs:

        #print(dataset_name,cls_name)
        y_true, y_pred = all_preds[(dataset_name,cls_name)]
        labels = list(np.unique(y_true))

        acc = evalua(y_true, y_pred)
        # Fill accuracy dataframe
        acc_df.at[dataset_name,cls_name]=acc

        # Get conf_mat
        cm = confusion_matrix(y_true, y_pred)
        cm_df = conf_mat_df(cm,labels)
        results[(dataset_name,cls_name,"cm")] = cm_df
        
        # Get classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        results[(dataset_name,cls_name,"report")] = report_df

        # Acc per class
        cm_dig = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_dig = cm_dig.diagonal()

        dfi = results[(dataset_name,"acc")]
        dfi[cls_name]=pd.Series(cm_dig,labels)    
        results[(dataset_name,"acc")]=dfi.copy()


    results["Acc"] = acc_df
    return results 
  
results = get_results(filename)


# Realizar pruebas para 50 %
run_all_save(filename, train_size=0.5)

# Obtener resultados
results_1 = get_results(filename)
df_total_1 = results_1["Acc"].astype(float)
df_conf_1 = results_1[("Data All", "SVM", "cm")].astype(float)
df_report_1 = results_1[("Data All", "SVM", "report")].astype(float)

# Realizar pruebas para 60 %
run_all_save(filename, train_size=0.6)

# Obtener resultados
results_2 = get_results(filename)
df_total_2 = results_2["Acc"].astype(float)
df_conf_2 = results_2[("Data All", "SVM", "cm")].astype(float)
df_report_2 = results_2[("Data All", "SVM", "report")].astype(float)

# Realizar pruebas para 70 %
run_all_save(filename, train_size=0.7)

# Obtener resultados
results_3 = get_results(filename)
df_total_3 = results_3["Acc"].astype(float)
df_conf_3 = results_3[("Data All", "SVM", "cm")].astype(float)
df_report_3 = results_3[("Data All", "SVM", "report")].astype(float)

# Realizar pruebas para 80%
run_all_save(filename, train_size=0.8)

# Obtener resultados
results_4 = get_results(filename)
df_total_4 = results_4["Acc"].astype(float)
df_conf_4 = results_4[("Data All", "SVM", "cm")].astype(float)
df_report_4 = results_4[("Data All", "SVM", "report")].astype(float)


df_total = results["Acc"].astype(float)
df_conf = results[("Data All","SVM","cm")].astype(float)
df_report = results[("Data All","SVM","report")].astype(float)


df_total
df_conf
df_report.round(4)[["precision","recall","f1-score"]]


# 50% - 50%
df_total_1
df_conf_1
df_report_1.round(4)[["precision","recall","f1-score"]]

# 60% - 40%
df_total_2
df_conf_2
df_report_2.round(4)[["precision","recall","f1-score"]]

# 70% - 30%
df_total_3
df_conf_3
df_report_3.round(4)[["precision","recall","f1-score"]]

# 80% - 20%
df_total_4
df_conf_4
df_report_4.round(4)[["precision","recall","f1-score"]]


# Concatenamos todos los df_total
df_total_combined = pd.concat([df_total_1, df_total_2, df_total_3, df_total_4], axis=0)
df_total_combined.index = ["50%-50%", "60%-40%", "70%-30%", "80%-20%"]
df_total_combined.columns = ["SVM", "MLP"]

df_total_combined # imprimo una tabla con la combinación de las tasas de acierto de todos los porcentajes de entrenamiento

# Obtener los porcentajes de partición y las tasas de acierto para SVM y MLP
train_sizes = [0.5, 0.6, 0.7, 0.8]
svm_acc_values = df_total_combined['SVM'].values
mlp_acc_values = df_total_combined['MLP'].values

# Crear una figura para colocar el gráfico
fig, ax = plt.subplots(figsize=(5, 4))

# Graficar la tasa de acierto para SVM
ax.plot(train_sizes, svm_acc_values, marker='o', color='blue', label='SVM')

# Graficar la tasa de acierto para MLP
ax.plot(train_sizes, mlp_acc_values, marker='s', color='green', label='MLP')

# Configurar el diseño del gráfico
ax.set_xlabel('Porcentaje de Partición del Conjunto de Entrenamiento')
ax.set_ylabel('Tasa de Acierto')
ax.set_title('Tasas de Acierto para SVM y MLP')
ax.legend()

# Cambiar el color de fondo
ax.set_facecolor('lavender')

# Mostrar el gráfico
plt.show()


# Definir los porcentajes de partición
train_sizes = [0.5, 0.6, 0.7, 0.8]

# Número de ejecuciones
num_executions = 10

# Lista para almacenar los resultados del promedio del método para SVM y MLP
results_average_svm = []
results_average_mlp = []

# Iterar sobre los diferentes porcentajes de partición
for train_size in train_sizes:
    acc_svm = []
    acc_mlp = []

    # Realizar el experimento 10 veces
    for _ in range(num_executions):
        # Ejecutar 10 veces cada método para SVM
        run_all_save(filename, train_size=train_size)
        results_svm = get_results(filename)
        accuracy_svm = results_svm["Acc"].astype(float).loc["Data All", "SVM"]
        acc_svm.append(accuracy_svm)

        # Ejecutar 10 veces cada método para MLP
        run_all_save(filename, train_size=train_size)
        results_mlp = get_results(filename)
        accuracy_mlp = results_mlp["Acc"].astype(float).loc["Data All", "MLP"]
        acc_mlp.append(accuracy_mlp)

    # Calcular el promedio y desviación estándar para SVM
    mean_average_svm = np.mean(acc_svm)
    std_average_svm = np.std(acc_svm)

    # Almacenar el promedio y desviación estándar en una lista para SVM
    results_average_svm.append((mean_average_svm, std_average_svm))

    # Calcular el promedio y desviación estándar para MLP
    mean_average_mlp = np.mean(acc_mlp)
    std_average_mlp = np.std(acc_mlp)

    # Almacenar el promedio y desviación estándar en una lista para MLP
    results_average_mlp.append((mean_average_mlp, std_average_mlp))

# Desempaquetar los resultados para la visualización para SVM
means_average_svm, stds_average_svm = zip(*results_average_svm)

# Desempaquetar los resultados para la visualización para MLP
means_average_mlp, stds_average_mlp = zip(*results_average_mlp)

# Crear gráfico con barras de error (desviación estándar) para SVM
plt.figure(figsize=(5, 3))
plt.errorbar(train_sizes, means_average_svm, yerr=stds_average_svm, label='SVM', marker='o', linestyle='-', color='purple', capsize=5)

# Crear gráfico con barras de error (desviación estándar) para MLP
plt.errorbar(train_sizes, means_average_mlp, yerr=stds_average_mlp, label='MLP', marker='s', linestyle='-', color='orange', capsize=5)

plt.title('Tasa de Acierto Promedio y Desviación Estándar en función del porcentaje de partición')
plt.xlabel('Porcentaje de partición del conjunto de entrenamiento')
plt.ylabel('Tasa de Acierto')
plt.legend()
plt.show()
