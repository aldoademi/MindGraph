import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Carica l'immagine
img = cv2.imread('/home/aldoademi/Desktop/targets/economist_daily_chart_103.png')

# Ridimensiona l'immagine a una dimensione fissa
img = cv2.resize(img, (224, 224))

# Normalizza l'immagine
img = img.astype('float32') / 255.0

# Crea un modello di rete neurale convoluzionale per calcolare la mappa di salienza
model = keras.applications.VGG16(include_top=False, weights=None, input_shape=(224, 224, 3))
x = model.output
x = keras.layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(x)
salient_map_model = keras.models.Model(model.input, x)

# Calcola la mappa di salienza dell'immagine
saliency_map = salient_map_model.predict(np.expand_dims(img, axis=0))[0, :, :, 0]

# Applica una soglia alla mappa di salienza per ottenere una maschera binaria
_, saliency_mask = cv2.threshold(saliency_map, 0.5, 1, cv2.THRESH_BINARY)

# Calcola il rapporto tra il numero di pixel salienti nella maschera e il numero totale di pixel dell'immagine
memorability = np.sum(saliency_mask) / (saliency_mask.shape[0] * saliency_mask.shape[1])

# Stampa il grado di memorabilità
print("Grado di memorabilità dell'immagine: ", memorability)
