"""
clone.py
--------
Modulo "ad alto livello": mette insieme domain.py, guidance.py e solver.py
per realizzare il seamless cloning (Sezione 3 del paper) — incollare
un'immagine sorgente dentro una destinazione, senza cuciture visibili.

E' il modulo che userai direttamente (o che usera' la futura interfaccia
grafica) senza doverti preoccupare dei dettagli del sistema lineare.
"""

import numpy as np

import guidance as guidance_module
import solver


def place_source_in_canvas(source, source_mask, canvas_shape, offset):
    """
    Allinea la sorgente e la sua maschera Omega alle coordinate della
    destinazione. Serve perche' il paper richiede che sorgente e
    destinazione vivano nello stesso sistema di riferimento S: il
    gradiente si calcola confrontando pixel VICINI, quindi non ha senso
    lavorare in due sistemi di coordinate diversi.

    Immagina di prendere un canvas vuoto, grande esattamente come la
    destinazione, e di incollarci sopra la sorgente (e la sua maschera)
    nel punto indicato da 'offset'. Il resto del canvas resta vuoto (a
    0): non verra' mai letto, perche' e' fuori da Omega.

    Argomenti:
        source: array (Hs, Ws, 3), l'immagine sorgente originale.
        source_mask: array booleano (Hs, Ws), Omega definita NELLE
                     coordinate della sorgente (es. un cerchio disegnato
                     grossolanamente attorno all'oggetto da clonare).
        canvas_shape: (H, W), le dimensioni dell'immagine di destinazione.
        offset: (top, left) — la riga e la colonna della destinazione in
                cui deve finire l'angolo in alto a sinistra della sorgente.

    Ritorna:
        aligned_source: array (H, W, 3), come 'source' ma posizionata
                        dentro un canvas grande quanto la destinazione.
        aligned_mask: array booleano (H, W), Omega nelle coordinate della
                      destinazione.
    """
    H, W = canvas_shape
    Hs, Ws, C = source.shape
    top, left = offset

    aligned_source = np.zeros((H, W, C), dtype=source.dtype)
    aligned_mask = np.zeros((H, W), dtype=bool)

    # Se l'offset porta la sorgente parzialmente (o del tutto) fuori dai
    # bordi della destinazione, la ritagliamo invece di lasciar fallire
    # il programma con un errore di indice fuori range.
    src_top = max(0, -top)
    src_left = max(0, -left)
    dst_top = max(0, top)
    dst_left = max(0, left)

    height = min(Hs - src_top, H - dst_top)
    width = min(Ws - src_left, W - dst_left)

    if height <= 0 or width <= 0:
        raise ValueError(
            "Con questo offset, la sorgente cade interamente fuori dalla destinazione."
        )

    aligned_source[dst_top:dst_top + height, dst_left:dst_left + width] = (
        source[src_top:src_top + height, src_left:src_left + width]
    )
    aligned_mask[dst_top:dst_top + height, dst_left:dst_left + width] = (
        source_mask[src_top:src_top + height, src_left:src_left + width]
    )

    return aligned_source, aligned_mask


def seamless_clone(source, source_mask, destination, offset,
                    guidance_fn=guidance_module.import_gradient_guidance):
    """
    Esegue il seamless cloning completo, dalla sorgente "grezza" al
    risultato finale.

    Argomenti:
        source: array (Hs, Ws, 3), l'immagine sorgente.
        source_mask: array booleano (Hs, Ws), Omega nelle coordinate della
                     sorgente.
        destination: array (H, W, 3), l'immagine di sfondo.
        offset: (top, left), dove posizionare la sorgente nella destinazione.
        guidance_fn: quale strategia di campo guida usare. Default:
                     guidance.import_gradient_guidance (il seamless
                     cloning "base" della Sezione 3). Questo e' il
                     parametro che rende il progetto modulare: domani,
                     per usare "mixing gradients", bastera' passare
                     guidance.mixing_gradient_guidance qui, senza
                     cambiare nient'altro in questa funzione.

    Ritorna:
        result: array (H, W, 3), l'immagine finale, valori in [0, 1].
    """
    # Questi controlli sono intenzionalmente qui, all'ingresso pubblico
    # dell'algoritmo: sia la GUI sia eventuali nuove funzionalita' future
    # ricevono errori leggibili prima di costruire il sistema sparso.
    if source.ndim != 3 or destination.ndim != 3:
        raise ValueError("Sorgente e destinazione devono essere immagini (H, W, C).")
    if source.shape[2] != destination.shape[2]:
        raise ValueError("Sorgente e destinazione devono avere lo stesso numero di canali.")
    if source_mask.shape != source.shape[:2]:
        raise ValueError("La maschera deve avere le stesse dimensioni della sorgente.")
    if not np.any(source_mask):
        raise ValueError("La maschera e' vuota: seleziona almeno un pixel della sorgente.")

    canvas_shape = destination.shape[:2]
    aligned_source, mask = place_source_in_canvas(
        source, source_mask, canvas_shape, offset
    )

    # Il campo guida si calcola sul canvas allineato: cosi' i suoi vicini
    # (N/S/E/W) sono gia' nello stesso sistema di coordinate della
    # destinazione, come richiesto dal paper.
    v = guidance_fn(aligned_source, destination, mask)

    # Nel cloning RGB classico il risultato finale e' un'immagine ordinaria;
    # qui il clipping e' appropriato. Altre operazioni (es. illuminazione in
    # logaritmo) chiamano invece direttamente solver.solve senza clip.
    return np.clip(solver.solve(mask, v, destination), 0.0, 1.0)


def naive_paste(source, source_mask, destination, offset):
    """
    NON fa parte dell'algoritmo del paper: e' un semplice "copia-incolla"
    dei pixel della sorgente dentro la maschera, senza nessuna correzione.
    La teniamo qui solo come termine di paragone nella demo, per vedere a
    occhio la differenza rispetto al seamless cloning.
    """
    canvas_shape = destination.shape[:2]
    aligned_source, mask = place_source_in_canvas(
        source, source_mask, canvas_shape, offset
    )
    result = destination.copy()
    result[mask] = aligned_source[mask]
    return result
