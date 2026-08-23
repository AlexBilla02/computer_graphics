"""
io_utils.py
-----------
Funzioni per leggere e scrivere immagini su disco.

Usiamo Pillow (PIL) SOLO come lettore/scrittore di file immagine (jpg, png...),
esattamente come faremmo con una libreria per leggere un file di testo:
non e' una libreria di computer vision, non fa nessun calcolo per noi.
Tutta la matematica del paper la scriveremo noi con NumPy.

Convenzione usata in tutto il progetto:
- un'immagine a colori e' un array NumPy float64 di shape (H, W, 3),
  con valori compresi tra 0.0 e 1.0 (invece che 0-255, e' piu' comodo
  per i calcoli che faremo piu' avanti).
- una maschera (per rappresentare Omega) e' un array NumPy booleano
  di shape (H, W): True = pixel dentro Omega, False = pixel fuori.
"""

import numpy as np
from PIL import Image


def load_image(path):
    """
    Carica un'immagine a colori da disco.

    Ritorna:
        array NumPy float64, shape (H, W, 3), valori in [0, 1].
    """
    img = Image.open(path).convert("RGB")
    array = np.asarray(img, dtype=np.float64) / 255.0
    return array


def save_image(path, array):
    """
    Salva un'immagine a colori su disco.

    Argomenti:
        path: percorso del file di destinazione (l'estensione, es. .png,
              decide il formato).
        array: array NumPy shape (H, W, 3), valori attesi in [0, 1].
               Se durante i calcoli qualche valore fosse leggermente
               fuori da questo intervallo (puo' succedere, la matematica
               non garantisce automaticamente il range), lo tronchiamo
               con np.clip prima di salvare.
    """
    clipped = np.clip(array, 0.0, 1.0)
    array_uint8 = (clipped * 255.0).round().astype(np.uint8)
    Image.fromarray(array_uint8, mode="RGB").save(path)


def save_mask(path, mask):
    """Salva una maschera booleana come PNG bianco/nero.

    Bianco indica i pixel dentro Omega e nero quelli esterni, la stessa
    convenzione usata da :func:`load_mask`. Il file risultante e' quindi
    riutilizzabile sia dalla CLI sia dai test YAML.
    """
    if mask.ndim != 2:
        raise ValueError("La maschera deve avere shape (H, W).")
    pixels = np.where(mask, 255, 0).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(path)


def load_mask(path, threshold=0.5):
    """
    Carica una maschera da un'immagine in scala di grigi.
    Convenzione: bianco = dentro Omega, nero = fuori da Omega.

    Argomenti:
        threshold: soglia (tra 0 e 1) sopra la quale un pixel e'
                   considerato "dentro" Omega. Utile se l'immagine
                   della maschera non e' perfettamente bianco/nero
                   (es. e' stata salvata come JPG con compressione).

    Ritorna:
        array NumPy booleano, shape (H, W).
    """
    img = Image.open(path).convert("L")  # "L" = un solo canale, scala di grigi
    array = np.asarray(img, dtype=np.float64) / 255.0
    return array > threshold
