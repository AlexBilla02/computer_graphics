"""Operazioni di *selection editing* della Sezione 4 del paper.

Diversamente dal seamless cloning, queste operazioni modificano una sola
immagine all'interno di Omega. Il solver resta completamente condiviso: qui
decidiamo soltanto quale spazio colore usare e quale campo guida costruire.
"""

from functools import partial

import numpy as np

import clone
import guidance
import solver


def _validate_inplace_input(image, mask):
    """Controlli comuni alle operazioni che lavorano su una sola immagine."""
    if image.ndim != 3:
        raise ValueError("L'immagine deve avere shape (H, W, C).")
    if mask.shape != image.shape[:2]:
        raise ValueError("La maschera deve avere le stesse dimensioni dell'immagine.")
    if not np.any(mask):
        raise ValueError("La maschera e' vuota: seleziona almeno un pixel.")


def guided_inplace_edit(image, mask, guidance_fn):
    """Risolve un editing in-place con condizioni al contorno dell'immagine.

    ``guidance_fn`` riceve l'immagine sia come source sia come destination
    per usare la stessa firma dei campi guida del progetto. Non viene fatto
    clipping qui: il chiamante sa se sta lavorando in RGB oppure in log-RGB.
    """
    _validate_inplace_input(image, mask)
    field = guidance_fn(image, image, mask)
    return solver.solve(mask, field, image)


def texture_flatten(image, mask, edge_threshold=0.10):
    """Appiattisce la texture preservando i bordi (eq. 14--15 del paper)."""
    field_builder = partial(
        guidance.texture_flattening_guidance, edge_threshold=edge_threshold
    )
    return np.clip(guided_inplace_edit(image, mask, field_builder), 0.0, 1.0)


def local_illumination_change(image, mask, alpha_scale=0.2, beta=0.2):
    """Modifica localmente l'illuminazione con la trasformazione eq. (16).

    Il paper applica la trasformazione nel dominio logaritmico. Convertiamo
    quindi RGB -> log(RGB), risolviamo con condizioni al contorno anch'esse
    in log-RGB e solo alla fine torniamo a RGB con exp().
    """
    _validate_inplace_input(image, mask)
    # Un pixel RGB esattamente nero non ha logaritmo finito. Il floor e'
    # solo numerico e non cambia i pixel non neri dell'immagine.
    log_image = np.log(np.maximum(image, 1e-6))
    field_builder = partial(
        guidance.local_illumination_guidance,
        alpha_scale=alpha_scale,
        beta=beta,
    )
    solved_log_image = guided_inplace_edit(log_image, mask, field_builder)
    # exp(log(x)) puo' differire da x di pochi ulp anche FUORI da Omega.
    # Riportiamo esplicitamente fuori dalla selezione i pixel originali:
    # e' la condizione di Dirichlet dell'editing locale, non solo un test.
    result = image.copy()
    result[mask] = np.clip(np.exp(solved_log_image[mask]), 0.0, 1.0)
    return result


def _luminance_rgb(image):
    """Restituisce una versione monocromatica RGB con luminanza Rec. 601."""
    if image.shape[2] != 3:
        raise ValueError("La conversione in luminanza richiede un'immagine RGB.")
    luminance = np.dot(image, np.array([0.299, 0.587, 0.114]))
    return np.repeat(luminance[..., None], 3, axis=2)


def background_decolorization(image, mask):
    """Mantiene colorato l'oggetto in Omega e desatura il suo contorno.

    Implementa l'esempio di *local color changes* del paper: g e' la foto
    originale e f* e' la sua luminanza. Il cloning Poisson ricostruisce
    l'oggetto colorato dentro una selezione anche imprecisa, raccordandolo
    alla destinazione monocromatica.
    """
    _validate_inplace_input(image, mask)
    monochrome_destination = _luminance_rgb(image)
    return clone.seamless_clone(image, mask, monochrome_destination, (0, 0))


def recolor_selection(image, mask, multipliers=(1.5, 0.5, 0.5)):
    """Ricolora localmente un oggetto con seamless cloning in-place.

    Il paper forma g moltiplicando i canali RGB dell'originale per
    (1.5, 0.5, 0.5), quindi risolve con f* uguale all'immagine originale.
    ``multipliers`` rende esplicita e regolabile quella stessa operazione.
    """
    _validate_inplace_input(image, mask)
    multipliers = np.asarray(multipliers, dtype=np.float64)
    if multipliers.shape != (image.shape[2],) or np.any(multipliers < 0):
        raise ValueError("multipliers deve contenere un fattore non negativo per ogni canale.")
    modified_source = np.clip(image * multipliers[None, None, :], 0.0, 1.0)
    return clone.seamless_clone(modified_source, mask, image, (0, 0))


def seamless_tile(tile):
    """Rende tileable un rettangolo, imponendo bordi periodici (Sezione 4).

    Il bordo Nord/Sud viene fissato alla media dei due bordi originali,
    come nel paper; lo stesso vale per Est/Ovest. Il perimetro e' noto e
    l'interno e' ricostruito con il gradiente dell'immagine originale.
    Le quattro celle d'angolo ricevono una media comune, necessaria per
    soddisfare contemporaneamente entrambe le periodicita'.
    """
    if tile.ndim != 3 or tile.shape[0] < 3 or tile.shape[1] < 3:
        raise ValueError("Il rettangolo da rendere tileable deve essere almeno 3x3 pixel.")

    height, width, _ = tile.shape
    boundary_values = tile.copy()
    north_south = 0.5 * (tile[0, :, :] + tile[-1, :, :])
    east_west = 0.5 * (tile[:, 0, :] + tile[:, -1, :])
    boundary_values[0, :, :] = north_south
    boundary_values[-1, :, :] = north_south
    boundary_values[:, 0, :] = east_west
    boundary_values[:, -1, :] = east_west
    corner_value = 0.25 * (tile[0, 0] + tile[0, -1] + tile[-1, 0] + tile[-1, -1])
    boundary_values[(0, 0, -1, -1), (0, -1, 0, -1), :] = corner_value

    interior = np.zeros((height, width), dtype=bool)
    interior[1:-1, 1:-1] = True
    field = guidance.import_gradient_guidance(tile, boundary_values, interior)
    return np.clip(solver.solve(interior, field, boundary_values), 0.0, 1.0)


def seamless_tile_selection(image, mask):
    """Rende tileable il rettangolo selezionato e lo ricopia nell'immagine."""
    _validate_inplace_input(image, mask)
    ys, xs = np.nonzero(mask)
    top, bottom, left, right = ys.min(), ys.max(), xs.min(), xs.max()
    # Il tiling del paper e' definito su Omega rettangolare: non prendiamo
    # silenziosamente il bounding box di una selezione poligonale.
    if not np.all(mask[top:bottom + 1, left:right + 1]):
        raise ValueError("Seamless tiling richiede una selezione rettangolare piena.")
    result = image.copy()
    result[top:bottom + 1, left:right + 1] = seamless_tile(
        image[top:bottom + 1, left:right + 1]
    )
    return result


# Registro delle operazioni in-place: GUI e CLI ne usano gli stessi nomi.
INPLACE_OPERATIONS = {
    "texture-flatten": texture_flatten,
    "illumination": local_illumination_change,
    "background-decolorize": background_decolorization,
    "recolor": recolor_selection,
    "tile": seamless_tile_selection,
}

INPLACE_DISPLAY_NAMES = {
    "texture-flatten": "Texture flattening",
    "illumination": "Local illumination changes",
    "background-decolorize": "Background decolorization",
    "recolor": "Local recoloring",
    "tile": "Seamless tiling",
}
