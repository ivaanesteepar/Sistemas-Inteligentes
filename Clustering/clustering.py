import math
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import pandas as pd

from skimage import io
from sklearn.cluster import KMeans
from collections import Counter
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

# *** K-MEANS ***

# Lista de nombres de archivos de imágenes
image_files = ['Colorful.jpg', 'Keyboard.jpg', 'PoolBar.jpg']

# Lista de números de clusters correspondientes a cada imagen
num_clusters_list = [3, 5, 9]

# Nuevos valores para iteraciones y umbrales
new_max_iter = 50
new_tol = 1e-3

# Bucle externo para procesar cada imagen
for image_file in image_files:
    # Cargar la imagen
    img_arr = img.imread(image_file)
    (h, w, c) = img_arr.shape 
    img2D = img_arr.reshape(h * w, c) 
    
    # Bucle interno para probar con diferentes números de clusters
    fig, axs = plt.subplots(1, len(num_clusters_list), figsize=(16, 4))
    fig.suptitle(f'Image: {image_file}')
    
    for i, num_clusters in enumerate(num_clusters_list):
        # Aplicar K-Means con el número de clusters específico para cada imagen y los nuevos valores
        kmeans_model = KMeans(n_clusters=num_clusters, max_iter=new_max_iter, tol=new_tol)
        cluster_labels = kmeans_model.fit_predict(img2D)
        labels_count = Counter(cluster_labels) 
        rgb_cols = kmeans_model.cluster_centers_.round(0).astype(int) 
        img_quant = np.reshape(rgb_cols[cluster_labels], (h, w, c))
        
        # Mostrar las imágenes cuantizadas
        axs[i].imshow(img_quant) 
        axs[i].set_title(f'{num_clusters} clusters')

    plt.show()


# *** MODIFICACIÓN DEL NÚMERO DE ITERACIONES Y UMBRALES ***

import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

# Lista de nombres de archivos de imágenes
image_files = ['Colorful.jpg', 'Keyboard.jpg', 'PoolBar.jpg']

# Número de clusters deseado
num_clusters = 5

# Lista de valores de iteraciones
iterations_list = [50, 80, 100]

# Bucle externo para procesar cada imagen
for image_file in image_files:
    # Cargar la imagen
    img_arr = img.imread(image_file)
    (h, w, c) = img_arr.shape
    img2D = img_arr.reshape(h * w, c)

    # Bucle interno para probar con diferentes valores de iteraciones
    fig, axs = plt.subplots(1, len(iterations_list), figsize=(16, 4))
    fig.suptitle(f'Image: {image_file}')

    for i, max_iter in enumerate(iterations_list):
        # Calcular el índice de tolerancia correspondiente
        tol_index = i

        # Asignar el valor de tolerancia
        tol = 10**(-3 - tol_index)

        # Aplicar K-Means con el número de clusters específico y los valores de iteraciones y umbral
        kmeans_model = KMeans(n_clusters=num_clusters, max_iter=max_iter, tol=tol)
        cluster_labels = kmeans_model.fit_predict(img2D)
        labels_count = Counter(cluster_labels)
        rgb_cols = kmeans_model.cluster_centers_.round(0).astype(int)
        img_quant = np.reshape(rgb_cols[cluster_labels], (h, w, c))

        # Mostrar las imágenes cuantizadas
        axs[i].imshow(img_quant)
        axs[i].set_title(f'Iteracciones: {max_iter}, Umbrales: {tol}')

    plt.show()


# ***************************************************************************************************

# *** FUZZY C-MEANS ***

import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
from skfuzzy import cmeans  # Instala esta biblioteca mediante el comando: conda install -c conda-forge scikit-fuzzy

# Se cargan las imagenes
imagen_a = io.imread('Colorful.jpg')
imagen_b = io.imread('Keyboard.jpg')
imagen_c = io.imread('PoolBar.jpg')

# Lista de imágenes y sus nombres
imagenes = {'Colorful': imagen_a, 'Keyboard': imagen_b, 'PoolBar': imagen_c}

# Número de clusters
num_clusters_lista = [3, 5, 9]

# Configurar el algoritmo
m_parametro = 1.5  # Parámetro que determina la difusividad
max_iteraciones = 50  # Iteraciones
error_umbral = 1e-3  # Umbral

# Configurar el tamaño de la cuadrícula de subgráficos
figura, ejes = plt.subplots(len(imagenes), len(num_clusters_lista), figsize=(20, 8))

for j, (nombre_imagen, imagen) in enumerate(imagenes.items()):
    # Aplanar la imagen y normalizarla
    imagen_plana = imagen.reshape((-1, 3)).astype(float) / 255.0
    
    # Iterar sobre cada número de clusters
    for i, c in enumerate(num_clusters_lista):
        centros_imagen, u_imagen, _, _, _, _, _ = cmeans(imagen_plana.T, c, m_parametro, error_umbral, max_iteraciones, seed=0)
                
        # Obtener las etiquetas de cluster para cada píxel
        etiquetas = np.argmax(u_imagen, axis=0)
                
        # Reconstruir la imagen con los centros de cluster
        imagen_agrupada = np.zeros_like(imagen_plana)
        for l in range(c):
            puntos_cluster = imagen_plana[etiquetas == l]
            imagen_agrupada[etiquetas == l] = np.mean(puntos_cluster, axis=0)

        # Revertir la normalización y remodelar la imagen
        imagen_agrupada = (imagen_agrupada * 255.0).reshape(imagen.shape)
                
        # Visualizar las imágenes segmentadas
        ejes[j, i].imshow(imagen_agrupada.astype(np.uint8))
        ejes[j, i].set_title(f'{nombre_imagen}\nClusters: {c}', fontsize=16)
        ejes[j, i].axis('off')

plt.tight_layout()
plt.show()


# *** MODIFICACIÓN DEL NÚMERO DE ITERACIONES, UMBRALES Y DIFUSIVIDAD ***

import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
from skfuzzy import cmeans

# Lista de nombres de archivos de imágenes
image_files = ['Colorful.jpg', 'Keyboard.jpg', 'PoolBar.jpg']

# Número de clusters deseado
num_clusters = 5

# Lista de valores de iteraciones y m (difusividad) para cada imagen
iterations_and_m_values = [(50, 2), (80, 4), (100, 7)]

# Bucle externo para procesar cada imagen
for image_file in image_files:
    # Cargar la imagen
    imagen = io.imread(image_file)
    (h, w, c) = imagen.shape
    imagen_plana = imagen.reshape(h * w, c)

    # Bucle interno para probar con diferentes valores de iteraciones y m
    fig, axs = plt.subplots(1, len(iterations_and_m_values), figsize=(16, 4))
    fig.suptitle(f'Image: {image_file}')

    for i, (max_iter, m_parametro) in enumerate(iterations_and_m_values):
        # Calcular el índice de tolerancia correspondiente
        tol_index = i

        # Asignar el valor de tolerancia (umbrales)
        tol = 10 ** (-3 - tol_index)

        # Aplicar c-means con el número de clusters específico y los valores de las iteraciones, umbrales y m
        centros_imagen, u_imagen, _, _, _, _, _ = cmeans(
            imagen_plana.T, num_clusters, m_parametro, tol, max_iter, seed=0
        )

        # Obtener las etiquetas de cluster para cada píxel
        etiquetas = np.argmax(u_imagen, axis=0)

        # Reconstruir la imagen con los centros de cluster
        imagen_agrupada = np.zeros_like(imagen_plana)
        for l in range(num_clusters):
            puntos_cluster = imagen_plana[etiquetas == l]
            imagen_agrupada[etiquetas == l] = np.mean(puntos_cluster, axis=0)

        # Revertir la normalización y remodelar la imagen
        imagen_agrupada = (imagen_agrupada * 255.0).reshape(imagen.shape)

        # Mostrar las imágenes agrupadas
        axs[i].imshow(imagen_agrupada.astype(np.uint8))
        axs[i].set_title(f'Iteraciones: {max_iter}, Umbrales: {tol}, M: {m_parametro}')
        axs[i].axis('off')

    plt.show()

# ***************************************************************************************************

# *** MIXTURA DE GAUSSIANAS ***

import numpy as np
from skimage import io
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def apply_gaussian_mixture_and_visualize(image, num_components_list, title):
    flattened_image = image.reshape((-1, 3)).astype(float) / 255.0
    max_iterations = 50
    convergence_tolerance = 1e-3

    fig, axes = plt.subplots(1, len(num_components_list), figsize=(15, 5))

    for i, n_components in enumerate(num_components_list):
        gmm = GaussianMixture(n_components=n_components, max_iter=max_iterations, tol=convergence_tolerance, random_state=0)
        gmm.fit(flattened_image)

        labels = gmm.predict(flattened_image)
        clustered_image = gmm.means_[labels].reshape(image.shape)

        axes[i].imshow(clustered_image)
        axes[i].set_title(f'Clusters: {n_components}', fontsize=16)
        axes[i].axis('off')

    plt.suptitle(title, fontsize=20)
    plt.show()

image_path_colorful = 'Colorful.jpg'
image_path_keyboard = 'Keyboard.jpg'
image_path_poolbar = 'PoolBar.jpg'

image_colorful = io.imread(image_path_colorful)
image_keyboard = io.imread(image_path_keyboard)
image_poolbar = io.imread(image_path_poolbar)

num_components_list = [3, 5, 9]

apply_gaussian_mixture_and_visualize(image_colorful, num_components_list, 'Colorful')
apply_gaussian_mixture_and_visualize(image_keyboard, num_components_list, 'Keyboard')
apply_gaussian_mixture_and_visualize(image_poolbar, num_components_list, 'PoolBar')



# *** MODIFICACIÓN DEL NÚMERO DE ITERACIONES Y UMBRALES ***

import numpy as np
from skimage import io
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def apply_gaussian_mixture_and_visualize(image, max_iterations_list, convergence_tolerance_list, title):
    # Preprocesamiento: convertir la imagen en una matriz 2D normalizada
    flattened_image = image.reshape((-1, 3)).astype(float) / 255.0

    # Crear subgráficos para visualizar resultados
    fig, axes = plt.subplots(1, len(max_iterations_list), figsize=(15, 5))

    # Iterar sobre las listas de iteraciones y umbrales
    for i, (max_iterations, convergence_tolerance) in enumerate(zip(max_iterations_list, convergence_tolerance_list)):
        # Crear un modelo de mezcla gaussiana con parámetros dados
        gmm = GaussianMixture(n_components=5, max_iter=max_iterations, tol=convergence_tolerance, random_state=0)
        gmm.fit(flattened_image)

        # Predecir etiquetas y reconstruir la imagen
        labels = gmm.predict(flattened_image)
        clustered_image = gmm.means_[labels].reshape(image.shape)

        # Visualizar la imagen clustered en el subgráfico actual
        axes[i].imshow(clustered_image)
        axes[i].set_title(f'Iteraciones: {max_iterations}, Umbrales: {convergence_tolerance}', fontsize=16)
        axes[i].axis('off')
        
        # Dar información a cada imagen
        print(f'Image: {title}, Iteraciones: {max_iterations}, Umbrales: {convergence_tolerance}')

    plt.suptitle(title, fontsize=20)
    plt.show()

# Rutas de las imágenes
image_path_colorful = 'Colorful.jpg'
image_path_keyboard = 'Keyboard.jpg'
image_path_poolbar = 'PoolBar.jpg'

# Cargar las imagenes
image_colorful = io.imread(image_path_colorful)
image_keyboard = io.imread(image_path_keyboard)
image_poolbar = io.imread(image_path_poolbar)

# Listas de iteraciones y umbrales de convergencia para cada imagen
max_iterations_list_50 = [50, 80, 100]
convergence_tolerance_list_50 = [1e-3, 1e-4, 1e-6]

# Aplicar la mezcla gaussiana y visualizar resultados para cada imagen
apply_gaussian_mixture_and_visualize(image_colorful, max_iterations_list_50, convergence_tolerance_list_50, 'Colorful')
apply_gaussian_mixture_and_visualize(image_keyboard, max_iterations_list_50, convergence_tolerance_list_50, 'Keyboard')
apply_gaussian_mixture_and_visualize(image_poolbar, max_iterations_list_50, convergence_tolerance_list_50, 'PoolBar')
