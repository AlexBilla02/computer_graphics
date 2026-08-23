"""
guidance.py
-----------
Qui vivono le diverse strategie per calcolare il campo guida v (Sezione 3
del paper "Poisson Image Editing"). E' il modulo pensato apposta per
essere "estendibile": ogni funzione di guida ha la STESSA firma,

    guidance_fn(source, destination, mask) -> dict {'N','S','E','W': array}

cioe', per ogni pixel, quattro numeri: il valore v_pq per ciascuno dei
suoi (fino a 4) vicini possibili, Nord/Sud/Est/Ovest.

Perche' proprio questo formato? Perche' e' esattamente quello che serve
nell'equazione discreta del paper (Sezione 2, eq. 7):

    |Np| f_p - sum_{q in Np, q in Omega} f_q
        = sum_{q in Np, q in dOmega} f*_q  +  sum_{q in Np} v_pq

Il solver (solver.py) usera' queste 4 mappe senza sapere COME sono state
calcolate: gli basta riceverle in questo formato. Quindi, per aggiungere
domani una nuova strategia (es. "mixing gradients", Sezione 3), bastera'
scrivere una nuova funzione qui con la stessa firma — nessun'altra parte
del codice (solver.py, clone.py) dovra' cambiare.

Convenzione sui segni (facile da sbagliare, quindi la scriviamo esplicita):
    v_pq = valore_guida(p) - valore_guida(q)

Quindi, per il pixel p, la mappa 'E' contiene v_{p, vicino_est}, la mappa
'W' contiene v_{p, vicino_ovest}, ecc. Nota che il valore in 'E' per un
pixel e' l'opposto del valore in 'W' per il suo vicino a destra: e' la
stessa identica differenza, guardata dai due lati opposti — lo stesso
principio gia' usato in domain.py per calcolare il bordo.
"""

import numpy as np


def _directional_differences(channel):
    """
    Funzione di supporto interna (il trattino basso iniziale e' una
    convenzione Python per dire "non usarla da fuori questo file"):
    dato UN SOLO canale colore (shape (H, W)), calcola le 4 mappe di
    differenza N/S/E/W definite sopra.

    Esempio per la mappa 'E' (Est): per ogni pixel (y, x) che ha un vicino
    a destra (cioe' x non e' l'ultima colonna),
        E[y, x] = channel[y, x] - channel[y, x + 1]

    Per i pixel sull'ultimo bordo dell'immagine (senza vicino in quella
    direzione) lasciamo il valore a 0: quella direzione non verra' mai
    consultata dal solver per quei pixel, perche' il vicino non esiste.
    """
    H, W = channel.shape

    E = np.zeros((H, W), dtype=np.float64)
    Ovest = np.zeros((H, W), dtype=np.float64)
    S = np.zeros((H, W), dtype=np.float64)
    N = np.zeros((H, W), dtype=np.float64)

    E[:, :-1] = channel[:, :-1] - channel[:, 1:]
    Ovest[:, 1:] = channel[:, 1:] - channel[:, :-1]
    S[:-1, :] = channel[:-1, :] - channel[1:, :]
    N[1:, :] = channel[1:, :] - channel[:-1, :]

    return {"N": N, "S": S, "E": E, "W": Ovest}


def _color_directional_differences(image):
    """
    Funzione di supporto condivisa: applica _directional_differences a
    un'immagine A COLORI (H, W, C), canale per canale, e restituisce le
    4 mappe N/S/E/W complete (H, W, C). E' la parte di calcolo che serve
    identica sia per "import gradient" sia per "mixing gradients" (e per
    qualunque altra strategia futura basata su differenze tra pixel).
    """
    H, W, C = image.shape
    result = {direction: np.zeros((H, W, C), dtype=np.float64)
              for direction in "NSEW"}

    for c in range(C):
        diffs = _directional_differences(image[:, :, c])
        for direction in "NSEW":
            result[direction][:, :, c] = diffs[direction]

    return result


def import_gradient_guidance(source, destination, mask):
    """
    Strategia piu' semplice: il campo guida v e' semplicemente il
    gradiente della sorgente, v = grad(g) (Sezione 3, "Seamless cloning").

    Il risultato dentro Omega avra' la stessa "struttura fine" (lo stesso
    Laplaciano) della sorgente, con i valori assoluti che si adattano
    gradualmente per combaciare con la destinazione esattamente sul bordo.

    'destination' e 'mask' non vengono usati in questa particolare
    strategia, ma li riceviamo comunque per rispettare la firma comune:
    cosi' clone.py puo' chiamare QUALSIASI funzione di guidance nello
    stesso identico modo, che li usi o no.

    Argomenti:
        source: array (H, W, 3), gia' allineato alle coordinate della
                destinazione (fatto da clone.place_source_in_canvas).

    Ritorna:
        dict con chiavi 'N','S','E','W', ciascuna di shape (H, W, 3).
    """
    return _color_directional_differences(source)


def mixing_gradient_guidance(source, destination, mask):
    """
    Strategia "mixing gradients" (Sezione 3 del paper): invece di prendere
    SEMPRE il gradiente della sorgente, per ogni pixel e ogni direzione si
    prende il gradiente PIU' FORTE (in valore assoluto) tra sorgente e
    destinazione.

    A cosa serve: se incolli un oggetto con dei "buchi" (es. una
    ringhiera, un pizzo, del testo) o parzialmente trasparente, con
    import_gradient_guidance perderesti completamente cio' che c'e' sotto
    nella destinazione, perche' verrebbe sovrascritto da un'interpolazione
    liscia della sola sorgente. Con il mixing, dove la sorgente non ha
    nulla di significativo da dire (gradiente quasi zero, es. dentro un
    buco), vince invece il gradiente della destinazione, che quindi
    "traspare" naturalmente nel risultato finale.

    Nota concettuale: qui il campo v non e' piu' necessariamente il
    gradiente di un'unica immagine (puo' "saltare" dall'una all'altra
    pixel per pixel) — ma questo non e' un problema: l'equazione di
    Poisson si puo' risolvere con QUALSIASI campo vettoriale v come
    termine guida, non solo con gradienti "puri".

    Argomenti:
        source: array (H, W, 3), gia' allineato alle coordinate della
                destinazione.
        destination: array (H, W, 3), l'immagine di sfondo.

    Ritorna:
        dict con chiavi 'N','S','E','W', ciascuna di shape (H, W, 3).
    """
    source_diffs = _color_directional_differences(source)
    destination_diffs = _color_directional_differences(destination)

    result = {}
    for direction in "NSEW":
        source_value = source_diffs[direction]
        destination_value = destination_diffs[direction]

        # Il paper confronta l'intensita' del gradiente COLORATO, non ogni
        # componente RGB separatamente: scegliamo quindi una sola sorgente
        # (g oppure f*) per tutto il vettore RGB del bordo p-q.  Se si
        # confrontassero i canali separatamente, un singolo bordo potrebbe
        # diventare artificialmente, per esempio, "rosso dalla sorgente e
        # blu dalla destinazione", cosa che non corrisponde alla eq. (12).
        source_strength = np.linalg.norm(source_value, axis=2)
        destination_strength = np.linalg.norm(destination_value, axis=2)
        use_source = source_strength >= destination_strength
        # ``[..., None]`` estende la scelta (H, W) ai tre canali RGB.
        result[direction] = np.where(use_source[..., None], source_value, destination_value)

    return result


def texture_flattening_guidance(source, destination, mask, edge_threshold=0.10):
    """Campo guida per il *texture flattening* (paper, eq. 14--15).

    L'immagine da modificare e' ``destination``; ``source`` e' presente
    soltanto per rispettare la firma comune delle guidance functions. Per
    ogni arco discreto p-q conserviamo il gradiente originale solo se la
    sua norma RGB supera ``edge_threshold``; negli altri punti v_pq = 0.
    Questo e' il "sparse sieve" M del paper: le texture fini vengono
    eliminate, mentre i bordi abbastanza forti restano ancorati.

    ``edge_threshold`` e' espresso nella scala di colori [0, 1]. Il paper
    non prescrive un edge detector specifico: questa soglia sulla norma
    del gradiente e' un edge detector semplice, deterministico e privo di
    dipendenze esterne. Una soglia maggiore conserva meno dettagli.
    """
    if edge_threshold < 0:
        raise ValueError("edge_threshold deve essere maggiore o uguale a zero.")

    differences = _color_directional_differences(destination)
    result = {}
    for direction, value in differences.items():
        is_edge = np.linalg.norm(value, axis=2) >= edge_threshold
        # Eq. (15): gradiente originale sul bordo, zero altrove.
        result[direction] = np.where(is_edge[..., None], value, 0.0)
    return result


def local_illumination_guidance(source, destination, mask, alpha_scale=0.2,
                                beta=0.2, epsilon=1e-12):
    """Campo guida per le *local illumination changes* (paper, eq. 16).

    ``destination`` deve essere gia' nel dominio logaritmico. La funzione
    applica letteralmente v = alpha * beta * |grad f*|**(-beta) * grad f*.
    Come indicato nel paper, alpha e' ``alpha_scale`` (0.2 di default)
    moltiplicato per la norma media del gradiente nella selezione Omega.
    Il piccolo ``epsilon`` evita la divisione per zero nelle zone piatte.
    """
    if alpha_scale < 0:
        raise ValueError("alpha_scale deve essere maggiore o uguale a zero.")
    if beta < 0:
        raise ValueError("beta deve essere maggiore o uguale a zero.")

    differences = _color_directional_differences(destination)
    valid_strengths = []
    for direction, value in differences.items():
        # Raccogliamo solo i gradienti che partono da Omega, coerentemente
        # con la definizione del campo guida nel dominio selezionato.
        valid_strengths.append(np.linalg.norm(value, axis=2)[mask])
    average_norm = np.concatenate(valid_strengths).mean() if np.any(mask) else 0.0
    alpha = alpha_scale * average_norm

    result = {}
    for direction, value in differences.items():
        norm = np.linalg.norm(value, axis=2)
        multiplier = alpha * beta * np.maximum(norm, epsilon) ** (-beta)
        result[direction] = multiplier[..., None] * value
    return result


# Registro unico delle modalita' disponibili.  CLI e GUI lo importano invece
# di mantenere ciascuna una propria copia delle funzioni supportate.
# Per introdurre un nuovo campo guida basta quindi implementare la funzione
# qui sopra e aggiungerla a questo dizionario con un nome stabile.
GUIDANCE_STRATEGIES = {
    "import": import_gradient_guidance,
    "mixing": mixing_gradient_guidance,
}

# Le etichette sono separate dai nomi tecnici: il nome ``import`` resta
# adatto alla CLI/API, mentre la GUI puo' mostrare un testo descrittivo.
GUIDANCE_DISPLAY_NAMES = {
    "import": "Import gradients (seamless cloning)",
    "mixing": "Mixing gradients",
}
