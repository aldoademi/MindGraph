import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Funzione per selezionare l'immagine
def select_image():
    # Apre la finestra di dialogo per selezionare un file
    path = filedialog.askopenfilename()
    if len(path) > 0:
        # Carica l'immagine
        img = cv2.imread(path)
        
        # Converti l'immagine da BGR a RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
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
        
        # Aggiorna l'interfaccia grafica con il grado di memorabilità calcolato
        result_label.config(text=f"Grado di memorabilità dell'immagine: {memorability:.2f}")
        
        # Mostra l'immagine nella finestra
        img = (img * 255).astype('uint8')
        img_tk = ImageTk.PhotoImage(Image.fromarray(img))
        img_label.config(image=img_tk)
        img_label.image = img_tk
        

# Crea l'interfaccia grafica
root = tk.Tk()
root.title("Calcolo grado di memorabilità")
root.geometry("600x400")

# Aggiunge un pulsante per selezionare l'immagine
select_button = tk.Button(root, text="Seleziona immagine", command=select_image)
select_button.pack(pady=20)

# Aggiunge un'etichetta per mostrare l'immagine
img_label = tk.Label(root)
img_label.pack(pady=20)

# Aggiunge una label per mostrare il risultato
result_label = tk.Label(root, text="")
result_label.pack()

# Avvia l'interfaccia grafica
root.mainloop()
