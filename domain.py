"""
domain.py
---------
Gestisce il dominio Omega (la regione dell'immagine che vogliamo
ricalcolare) e il suo bordo discreto ∂Omega, seguendo esattamente la
notazione della Sezione 2 del paper "Poisson Image Editing".

Richiamo dal paper (versione discreta):
    - Omega e' un insieme di pixel (qui: una maschera booleana).
    - Per ogni pixel p, Np e' l'insieme dei suoi vicini "a 4 connessioni"
      (sopra, sotto, sinistra, destra) che appartengono all'immagine S.
    - Il bordo ∂Omega e' l'insieme dei pixel FUORI da Omega che hanno
      almeno un vicino DENTRO Omega.
"""

import numpy as np


def rectangular_mask(shape, top, left, height, width):
    """
    Crea una maschera Omega rettangolare. Utile per i primi test, prima
    di avere un vero strumento di selezione (lo faremo con la GUI, piu'
    avanti).

    Argomenti:
        shape: (H, W), dimensioni dell'immagine su cui costruire la maschera.
        top, left: coordinate (riga, colonna) dell'angolo in alto a sinistra.
        height, width: altezza e larghezza del rettangolo.
    """
    mask = np.zeros(shape, dtype=bool)
    mask[top:top + height, left:left + width] = True
    return mask


def elliptical_mask(shape, center, axes):
    """
    Crea una maschera Omega ellittica (o circolare, se i due semiassi
    sono uguali). Utile anch'essa per i test, e per selezioni "morbide"
    tipiche del seamless cloning (il paper stesso nota che una selezione
    grossolana va benissimo).

    Argomenti:
        center: (riga, colonna) del centro dell'ellisse.
        axes: (semiasse verticale, semiasse orizzontale), in pixel.
    """
    H, W = shape
    cy, cx = center
    ry, rx = axes
    # np.ogrid crea due griglie "sottili" (una colonna, una riga) che
    # numpy poi espande automaticamente (broadcasting) al confronto sotto,
    # evitando di dover costruire due griglie H x W complete in memoria.
    y_indices, x_indices = np.ogrid[:H, :W]
    normalized = ((y_indices - cy) / ry) ** 2 + ((x_indices - cx) / rx) ** 2
    return normalized <= 1.0


def compute_boundary(mask):
    """
    Calcola il bordo discreto ∂Omega: i pixel FUORI da Omega che toccano
    (con connettivita' a 4) almeno un pixel DENTRO Omega.

    Invece di scrivere un doppio ciclo "for y in range(H): for x in
    range(W): ..." (corretto ma lentissimo in Python puro), sfruttiamo
    NumPy: confrontiamo l'intera maschera con se stessa "spostata" di un
    pixel in ciascuna delle 4 direzioni. E' lo stesso identico
    ragionamento, solo vettorizzato.

    Ritorna:
        array booleano, stessa shape di mask.
    """
    outside = ~mask

    touches_omega = np.zeros_like(mask, dtype=bool)

    # touches_omega[y, x] diventa True se il vicino in quella direzione
    # e' dentro Omega. Ogni riga sotto controlla una sola direzione.
    touches_omega[:, :-1] |= mask[:, 1:]   # vicino a destra e' in Omega?
    touches_omega[:, 1:] |= mask[:, :-1]   # vicino a sinistra e' in Omega?
    touches_omega[:-1, :] |= mask[1:, :]   # vicino sotto e' in Omega?
    touches_omega[1:, :] |= mask[:-1, :]   # vicino sopra e' in Omega?

    return outside & touches_omega


def num_pixels(mask):
    """Conta quanti pixel True (dentro Omega) ci sono nella maschera."""
    return int(mask.sum())
