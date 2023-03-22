import cv2
import tkinter as tk
from tkinter import filedialog, messagebox

def select_image():
    # Apre una finestra di dialogo per selezionare un'immagine
    file_path = filedialog.askopenfilename()
    if file_path:
        # Carica l'immagine selezionata
        img = cv2.imread(file_path)

        # Converte l'immagine in formato CV_32FC1 e in scala di grigi
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float32') / 255
        img_scaled = img * 255
        img_scaled = img_scaled.astype('uint8')

        # Calcola la mappa dei salienti dell'immagine con il metodo "Spectral Residual"
        saliency_map = cv2.saliency.StaticSaliencySpectralResidual_create().computeSaliency(img)

        # Applica una soglia alla mappa dei salienti per ottenere una maschera binaria
        _, saliency_mask = cv2.threshold(img_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Calcola il rapporto tra il numero di pixel bianchi nella maschera e il numero totale di pixel dell'immagine
        memorability = cv2.countNonZero(saliency_mask) / (img.shape[0] * img.shape[1])

        # Mostra l'immagine selezionata
        cv2.imshow('Immagine selezionata', cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB))

        # Stampa il grado di memorabilità
        messagebox.showinfo("Grado di memorabilità", f"Grado di memorabilità dell'immagine: {memorability}")
        cv2.waitKey(0)

# Crea una finestra Tkinter
root = tk.Tk()

# Aggiunge un bottone per selezionare un'immagine
select_button = tk.Button(root, text='Seleziona un\'immagine', command=select_image)
select_button.pack()

# Avvia il loop principale della finestra
root.mainloop()
