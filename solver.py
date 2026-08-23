"""
solver.py
---------
Costruisce e risolve il sistema lineare descritto nella Sezione 2 del
paper (l'equazione discreta di Poisson), dati:
    - una maschera Omega (domain.py)
    - un campo guida v, nel formato N/S/E/W (guidance.py)
    - un'immagine "nota" da cui prendere i valori sul bordo dOmega
      (la destinazione)

Richiamo dell'equazione per ogni pixel p dentro Omega (connettivita' a 4):

    |Np| f_p - sum_{q in Np, q in Omega} f_q
        = sum_{q in Np, q in dOmega} f*_q  +  sum_{q in Np} v_pq

Dove |Np| e' il numero di vicini di p che appartengono all'immagine S
(di solito 4, meno se p e' vicinissimo al bordo dell'immagine stessa,
non al bordo di Omega).

Punto chiave per capire il design: la matrice A del sistema dipende SOLO
dalla maschera Omega (dice quali pixel sono incogniti e come sono
collegati ai loro vicini) — non dipende ne' dal campo guida ne' dal
colore. Per questo la costruiamo una volta sola e la riusiamo per tutti
e 3 i canali colore (R, G, B): cambia solo il termine noto b.
"""

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg


# Le 4 direzioni con il relativo spostamento (dy, dx) per trovare il
# pixel vicino corrispondente. Le chiavi devono corrispondere esattamente
# a quelle usate in guidance.py ('N','S','E','W').
_NEIGHBOR_OFFSETS = {
    "N": (-1, 0),
    "S": (1, 0),
    "E": (0, 1),
    "W": (0, -1),
}


def _build_index_map(mask):
    """
    Assegna a ogni pixel dentro Omega un indice intero univoco (0, 1, 2...):
    lo useremo come numero di riga/colonna nella matrice sparsa A. I pixel
    FUORI da Omega non hanno bisogno di un indice, perche' il loro valore
    e' gia' noto (viene dalla destinazione) — non sono incognite del
    sistema.

    Ritorna:
        index_map: array (H, W) di interi; -1 dove il pixel non e' in Omega.
        omega_coords: array (num_unknowns, 2), le coordinate (riga, colonna)
                      di ogni pixel incognito, nello stesso ordine dei loro
                      indici (0, 1, 2, ...).
    """
    H, W = mask.shape
    index_map = -np.ones((H, W), dtype=np.int64)
    omega_coords = np.argwhere(mask)  # lista ordinata di (riga, colonna) in Omega
    index_map[mask] = np.arange(len(omega_coords))
    return index_map, omega_coords


def build_system(mask, guidance, destination):
    """
    Costruisce la matrice sparsa A e, per ogni canale colore, il vettore
    termine noto b, secondo l'equazione discreta di Poisson riportata sopra.

    Perche' una matrice SPARSA e non una normale (piena)? Perche' se Omega
    ha, diciamo, 10.000 pixel, una matrice piena sarebbe 10.000 x 10.000 =
    100 milioni di numeri, quasi tutti zero (ogni equazione coinvolge al
    massimo 5 pixel: se' stesso + 4 vicini). La versione sparsa memorizza
    SOLO i valori diversi da zero, risparmiando memoria e tempo.

    Argomenti:
        mask: array booleano (H, W), Omega.
        guidance: dict 'N'/'S'/'E'/'W' -> array (H, W, C), come restituito
                  dalle funzioni in guidance.py.
        destination: array (H, W, C), da cui prendere i valori noti f*
                     sul bordo dOmega.

    Ritorna:
        A: matrice sparsa (num_unknowns x num_unknowns), formato CSR
           (un formato efficiente per risolvere sistemi lineari).
        b_channels: array (num_unknowns, C), il termine noto per ogni canale.
        index_map: mappa pixel -> indice, servira' dopo per rimettere la
                   soluzione (un vettore piatto) al suo posto nell'immagine.
    """
    H, W = mask.shape
    C = destination.shape[2]
    index_map, omega_coords = _build_index_map(mask)
    num_unknowns = len(omega_coords)

    # Costruiamo la matrice "riempiendola" con tre liste parallele
    # (riga, colonna, valore) invece che con una matrice piena: e' il modo
    # standard di costruire una matrice sparsa in SciPy (formato "COO").
    rows = []
    cols = []
    values = []
    b_channels = np.zeros((num_unknowns, C), dtype=np.float64)

    for p_index, (y, x) in enumerate(omega_coords):
        diagonal = 0  # accumulera' |Np|, il numero di vicini validi di p

        for direction, (dy, dx) in _NEIGHBOR_OFFSETS.items():
            ny, nx = y + dy, x + dx

            # Il vicino deve esistere dentro i confini dell'immagine: se p
            # e' proprio sul bordo dell'immagine intera (non di Omega!),
            # ha meno di 4 vicini disponibili.
            if not (0 <= ny < H and 0 <= nx < W):
                continue

            diagonal += 1

            # v_pq, il valore del campo guida verso questo vicino, va
            # SEMPRE sommato al termine noto (e' a destra nell'equazione).
            b_channels[p_index] += guidance[direction][y, x]

            if mask[ny, nx]:
                # Il vicino e' ANCH'ESSO incognito (dentro Omega): il
                # termine "-f_q" resta a sinistra dell'equazione, quindi
                # mettiamo un -1 nella colonna corrispondente di A.
                q_index = index_map[ny, nx]
                rows.append(p_index)
                cols.append(q_index)
                values.append(-1.0)
            else:
                # Il vicino e' NOTO (siamo sul bordo dOmega): il suo
                # valore f*_q va portato a destra dell'equazione, quindi
                # lo sommiamo direttamente al termine noto b.
                b_channels[p_index] += destination[ny, nx]

        # Il coefficiente sulla diagonale, per il pixel p stesso, e'
        # sempre |Np| (il numero di vicini validi), come nell'equazione 7.
        rows.append(p_index)
        cols.append(p_index)
        values.append(float(diagonal))

    A = sparse.coo_matrix(
        (values, (rows, cols)), shape=(num_unknowns, num_unknowns)
    ).tocsr()

    return A, b_channels, index_map


def solve(mask, guidance, destination):
    """
    Risolve il sistema lineare per tutti e 3 i canali colore e ricostruisce
    l'immagine finale: identica alla destinazione FUORI da Omega, con i
    valori appena calcolati DENTRO Omega.

    Ritorna:
        result: array (H, W, C), soluzione float non limitata. Il clipping
                e' responsabilita' del chiamante: e' indispensabile per le
                operazioni nel dominio logaritmico dell'illuminazione.
    """
    A, b_channels, index_map = build_system(mask, guidance, destination)
    C = destination.shape[2]

    result = destination.copy()

    # La maschera determina A e A e' identica per R, G e B.  La
    # fattorizzazione LU e' la parte costosa di una soluzione diretta:
    # calcolarla una volta sola e riusarla sotto evita di ripetere lo stesso
    # lavoro tre volte. ``splu`` richiede il formato CSC per efficienza.
    lu_factorization = splinalg.splu(A.tocsc())

    for c in range(C):
        # Risolviamo i tre termini noti con la stessa fattorizzazione di A.
        solution = lu_factorization.solve(b_channels[:, c])
        result[:, :, c][mask] = solution

    return result
