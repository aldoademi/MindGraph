import cv2

# Carica l'immagine
img = cv2.imread('/home/aldoademi/Desktop/targets/economist_daily_chart_103.png')

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

# Stampa il grado di memorabilità
print("Grado di memorabilità dell'immagine: ", memorability)
