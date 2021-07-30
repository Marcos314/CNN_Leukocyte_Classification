# -*- coding: utf-8 -*-
"""cnn_blood_v2.ipynb
Rede convolucional para identificação de células de tecido sanguíneo apartir de imagens de microscópio
Inspirado (quase totalmente copiado) de https://www.kaggle.com/kbrans/cnn-91-6-acc-with-new-train-val-test-splits
"""

# Bibliotecas Python necessárias
import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import keras
import tensorflow as tf

from tqdm import tqdm 

from sklearn.utils import shuffle
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.applications.vgg16 import VGG16 
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential, Model 
from keras.applications import DenseNet201
from keras.initializers import he_normal
from keras.layers import Lambda, SeparableConv2D, BatchNormalization, Dropout, MaxPooling2D, Input, Dense, Conv2D, Activation, Flatten 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


print(os.getcwd())
"""Normalização e Input dos dados"""
# Função recebe o path relativo onde se encontra imagens e retorna-as em 2 arrays: images e labels
def loadDataset(train_path, valid_path, test_path):
    datasets = [train_path, valid_path, test_path]
    images = []
    labels = []
    image_size = (150,150)
    # iterar nos datasets de treino e validação
    for dataset in datasets:
        print(dataset)
        # iterar nas subpastas dos datasets
        for folder in os.listdir(dataset):
            if   folder in ['eosinofilo']: label = 0
            elif folder in ['linfocito']:  label = 1
            elif folder in ['monocito']:   label = 2
            elif folder in ['neutrofilo']: label = 3
            # iterar em cada imagem dos datasets
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                # pegar caminho de cada imagem
                img_path = os.path.join(os.path.join(dataset, folder), file)
                # abrir e redimensionar cada imagem
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, image_size)
                # adicionar a imagem e seu label correspondente  à saída
                images.append(image)
                labels.append(label)
    # Criar arrays com as saídas (imagens e labels)
    images = np.array(images, dtype = 'float32')
    labels = np.array(labels, dtype = 'int32')
    return images, labels

# # Combinar todas as imagens originais em um único dataset
images, labels = loadDataset('/array/marcos314/TCC_2020/dataset/train/','/array/marcos314/TCC_2020/dataset/valid/','/array/marcos314/TCC_2020/dataset/test/')

# Embaralhar os dados e separar novos conjuntos de  treinamento (80%), validação (10%) e teste (10%)
images, labels = shuffle(images, labels, random_state=10)
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size = 0.2)
test_images, val_images, test_labels, val_labels = train_test_split(test_images, test_labels, test_size = 0.5)

n_train = train_labels.shape[0]
n_val = val_labels.shape[0]
n_test = test_labels.shape[0]

print("Quantidade de imagens para treinamento: {}".format(n_train))
print("Quantidade de imagens para validação: {}".format(n_val))
print("Quantidade de imagens para teste: {}".format(n_test))

print("Formato das imagens de treinamento: {}".format(train_images.shape))
print("Labels das imagens de treinamento: {}".format(train_labels.shape))
print("Formato das imagens de validação: {}".format(val_images.shape))
print("Labels das imagens de validação: {}".format(val_labels.shape))
print("Formato das imagens de teste: {}".format(test_images.shape))
print("Labels das imagens de teste: {}".format(test_labels.shape))

_, train_counts = np.unique(train_labels, return_counts = True)
_, val_counts = np.unique(val_labels, return_counts = True)
_, test_counts = np.unique(test_labels, return_counts = True)

"""Normalização dos dados"""

train_images = train_images / 255.0 
val_images = val_images / 255.0
test_images = test_images / 255.0

# """**Construção do modelo**"""


def buildModel():
    model = Sequential()

    '''Xception Model '''
    # Primeira camada concolucional
    model.add(Conv2D(32 , (3,3), padding = 'same', activation = 'relu', strides=(2, 2), input_shape = (150,150,3)))
    model.add(Conv2D(64 , (3,3), padding = 'same', activation = 'relu'))

    # Segunda camada concolucional
    model.add(SeparableConv2D(128, (3,3), activation = 'relu', padding = 'same'))
    model.add(SeparableConv2D(128, (3,3), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2), strides=(2, 2)))

    # Terceira camada concolucional
    model.add(SeparableConv2D(256, (3,3), activation = 'relu', padding = 'same'))
    model.add(SeparableConv2D(256, (3,3), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2), strides=(2, 2)))

    # Quarta camada concolucional
    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))
    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # Quinta camada concolucional
    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))
    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))
    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))

    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))
    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))
    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))

    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))
    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))
    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))

    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))
    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))
    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))

    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))
    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))
    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))

    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))
    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))
    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))

    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))
    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))
    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))

    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))
    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))
    model.add(SeparableConv2D(728, (3,3), activation = 'relu', padding = 'same'))

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.2))

    # Camada (Fully Connected ) FC para classificar as features aprendidas
    model.add(Flatten())
    model.add(Dense(units = 512 , activation = 'relu'))
    model.add(Dropout(0.7))
    model.add(Dense(units = 128 , activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 64 , activation = 'relu'))
    model.add(Dropout(0.3))

    # Camada de saída
    model.add(Dense(units = 4 , activation = 'softmax'))

    # Compilação
    model.compile(optimizer = "adam" , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])
    
    # Callback
    checkpoint = ModelCheckpoint(filepath='blood_model_check_point_epochs_10.hdf5', save_best_only=True, save_weights_only=False)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=3, verbose = 1, mode='min', restore_best_weights = True)
    learning_rate_reduction = ReduceLROnPlateau(
        monitor = 'val_accuracy', 
        patience = 2, 
        verbose = 1, 
        factor = 0.3, 
        min_lr = 0.000001)

    # Treinamento
    history = model.fit(
        train_images, 
        train_labels, 
        batch_size = 32, 
        epochs = 10, 
        validation_data=(val_images, val_labels), 
        callbacks=[learning_rate_reduction])

    model.save('BloodModel_29_JUL_eXception_completa.h5')

    return model


# Avaliação dp desempenh da CNN
def plotAccucaria(history):
    epochs = [i for i in range(10)]
    fig , ax = plt.subplots(1,2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    fig.set_size_inches(20,10)
    ax[0].plot(epochs , train_acc , 'go-' , label = 'Acurácia de treinamento')
    ax[0].plot(epochs , val_acc , 'ro-' , label = 'Acurária de validação')
    ax[0].set_title('Acurácia de treinamento e validação')
    ax[0].legend()
    ax[0].set_xlabel("Épocas")
    ax[0].set_ylabel("Acurácia")

    ax[1].plot(epochs , train_loss , 'g-o' , label = 'Perda no treinamento')
    ax[1].plot(epochs , val_loss , 'r-o' , label = 'Perda na validação')
    ax[1].set_title('Perdas no treinamento e validação')
    ax[1].legend()
    ax[1].set_xlabel("Épocas")
    ax[1].set_ylabel("Perdas")
    plt.savefig('../pictures/train_valid_acc.png', dpi=300, figsize=(9,5))
    

# Usando a função plotAcuracia
# plotAccucaria(history)

print('\n========= Resultados do Treinamento da CNN\n')
# Resultados do treinamento da CNN
results = buildModel().evaluate(test_images, test_labels)
print("Perda do modelo = ", results[0])
print("Acurácia do modelo = ", results[1]*100, "%")





# # Salvar o modelo
# model.save('BloodModel_29_JUL_eXception_completa.h5')

# from sklearn.metrics import classification_report

# predictions = model.predict(test_images)
# predictions = np.argmax(predictions,axis=1)
# predictions[:15]

# print(classification_report(
#     test_labels, 
#     predictions, 
#     target_names = ['eosinófilo (Class 0)', 'linfócito (Class 1)', 'monócito (Class 2)', 'neutrófilo (Class 3)']))

# cm = confusion_matrix(test_labels, predictions)
# cm = pd.DataFrame(cm, index = ['0', '1', '2', '3'], columns = ['0', '1', '2', '3'])
# cm

# def plot_confusion_matrix (cm):
#     plt.figure(figsize = (10,10))
#     sns.heatmap(
#         cm, 
#         cmap = 'Blues', 
#         linecolor = 'black', 
#         linewidth = 1, 
#         annot = True, 
#         fmt = '', 
#         xticklabels = class_names, 
#         yticklabels = class_names)
    
# plot_confusion_matrix(cm)
