"""
cli.py
------
Interfaccia da riga di comando per usare il seamless cloning su immagini
VERE (non piu' sintetiche come nelle demo).

Poiche' non abbiamo ancora una GUI per selezionare la regione col mouse
(verra' in futuro), per ora ci sono DUE modi per definire la maschera Ω:

  1) Una maschera gia' pronta, disegnata in un qualsiasi editor di
     immagini ed esportata come PNG bianco/nero (bianco = dentro Omega):

         python3 -m poisson_editing.cli \\
             --source foto_oggetto.jpg --destination sfondo.jpg \\
             --mask maschera.png \\
             --offset-top 40 --offset-left 60 \\
             --output risultato.png

  2) Una forma geometrica semplice descritta direttamente qui, comoda
     per provare velocemente senza dover disegnare nulla:

         python3 -m poisson_editing.cli \\
             --source foto_oggetto.jpg --destination sfondo.jpg \\
             --shape ellipse --center-y 100 --center-x 120 --radius-y 60 --radius-x 80 \\
             --offset-top 40 --offset-left 60 \\
             --output risultato.png

In entrambi i casi, PRIMA di lanciare il calcolo vero e proprio, conviene
generare un'anteprima con --preview-only: salva due immagini (la maschera
sovrapposta alla sorgente, e la posizione di destinazione sovrapposta allo
sfondo) senza risolvere il sistema lineare, cosi' puoi aggiustare i numeri
in pochi secondi finche' l'allineamento non ti sembra giusto:

    python3 -m poisson_editing.cli \\
        --source foto_oggetto.jpg --destination sfondo.jpg \\
        --shape ellipse --center-y 100 --center-x 120 --radius-y 60 --radius-x 80 \\
        --offset-top 40 --offset-left 60 \\
        --preview-only
"""

import argparse
import sys

import numpy as np

# Il progetto puo' essere usato sia come package sia direttamente dalla
# cartella del repository (``python3 cli.py ...``).  Il fallback evita che
# l'uso diretto fallisca per via degli import relativi.
try:
    from . import io_utils, domain, clone, guidance, editing
except ImportError:
    import io_utils
    import domain
    import clone
    import guidance
    import editing


def build_mask_from_args(args, source_shape):
    """
    Costruisce la maschera Omega (nelle coordinate della SORGENTE) in
    base agli argomenti da riga di comando: o caricandola da file, o
    generandola come forma geometrica semplice.
    """
    if args.mask is not None:
        return io_utils.load_mask(args.mask)

    if args.shape == "ellipse":
        required = ["center_y", "center_x", "radius_y", "radius_x"]
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            sys.exit(f"Per --shape ellipse servono anche: {', '.join('--' + m.replace('_', '-') for m in missing)}")
        return domain.elliptical_mask(
            source_shape,
            center=(args.center_y, args.center_x),
            axes=(args.radius_y, args.radius_x),
        )

    if args.shape == "rectangle":
        required = ["top", "left", "height", "width"]
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            sys.exit(f"Per --shape rectangle servono anche: {', '.join('--' + m for m in missing)}")
        return domain.rectangular_mask(
            source_shape, top=args.top, left=args.left,
            height=args.height, width=args.width,
        )

    sys.exit("Devi specificare --mask oppure --shape (ellipse o rectangle).")


def overlay_mask(image, mask, color, alpha):
    """Sovrappone una maschera booleana a un'immagine, solo per le
    immagini di anteprima (non fa parte dell'algoritmo del paper)."""
    result = image.copy()
    color_array = np.array(color, dtype=np.float64).reshape(1, 1, 3)
    result[mask] = (1 - alpha) * result[mask] + alpha * color_array
    return result


def save_preview(source, mask, destination, offset):
    """Salva due immagini di anteprima, per verificare l'allineamento
    PRIMA di lanciare il calcolo vero e proprio."""
    source_preview = overlay_mask(source, mask, color=(1.0, 0.0, 0.0), alpha=0.4)
    io_utils.save_image("preview_source.png", source_preview)

    _, aligned_mask = clone.place_source_in_canvas(source, mask, destination.shape[:2], offset)
    destination_preview = overlay_mask(destination, aligned_mask, color=(1.0, 0.0, 0.0), alpha=0.4)
    io_utils.save_image("preview_destination.png", destination_preview)

    print("Anteprima salvata: preview_source.png, preview_destination.png")
    print(f"Pixel selezionati in Omega: {domain.num_pixels(mask)}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Seamless cloning (Poisson Image Editing) da riga di comando."
    )
    parser.add_argument("--source", help="Percorso dell'immagine sorgente (richiesta solo per cloning).")
    parser.add_argument("--destination", required=True, help="Percorso dell'immagine di sfondo.")
    parser.add_argument("--output", default="result.png", help="Percorso del file di output.")

    parser.add_argument("--mask", default=None, help="Maschera Omega come immagine bianco/nero.")
    parser.add_argument("--shape", choices=["ellipse", "rectangle"], default=None,
                         help="In alternativa a --mask: una forma geometrica semplice.")

    # Parametri per --shape ellipse
    parser.add_argument("--center-y", type=int, default=None, dest="center_y")
    parser.add_argument("--center-x", type=int, default=None, dest="center_x")
    parser.add_argument("--radius-y", type=int, default=None, dest="radius_y")
    parser.add_argument("--radius-x", type=int, default=None, dest="radius_x")

    # Parametri per --shape rectangle
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--left", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)

    parser.add_argument("--offset-top", type=int, default=None, dest="offset_top",
                         help="Riga della destinazione in cui posizionare l'angolo in alto a sinistra della sorgente.")
    parser.add_argument("--offset-left", type=int, default=None, dest="offset_left",
                         help="Colonna della destinazione in cui posizionare l'angolo in alto a sinistra della sorgente.")

    parser.add_argument("--guidance", choices=list(guidance.GUIDANCE_STRATEGIES), default="import",
                         help="Strategia per il campo guida v (default: import).")

    parser.add_argument("--operation", choices=["clone", *editing.INPLACE_OPERATIONS], default="clone",
                        help="Operazione: clone (default), texture-flatten, illumination, background-decolorize, recolor o tile.")
    parser.add_argument("--edge-threshold", type=float, default=0.10,
                        help="Soglia del bordo per texture-flatten (default: 0.10).")
    parser.add_argument("--alpha-scale", type=float, default=0.2,
                        help="Fattore di alpha nell'eq. 16 (default paper: 0.2).")
    parser.add_argument("--beta", type=float, default=0.2,
                        help="Esponente beta nell'eq. 16 (default paper: 0.2).")
    parser.add_argument("--color-factors", type=float, nargs=3, metavar=("R", "G", "B"),
                        default=(1.5, 0.5, 0.5),
                        help="Fattori RGB per --operation recolor (default: 1.5 0.5 0.5).")

    parser.add_argument("--preview-only", action="store_true",
                         help="Salva solo le immagini di anteprima, senza risolvere il sistema lineare.")

    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    destination = io_utils.load_image(args.destination)
    if args.operation == "clone":
        if args.source is None:
            sys.exit("--source e' obbligatorio per --operation clone.")
        if args.offset_top is None or args.offset_left is None:
            sys.exit("--offset-top e --offset-left sono obbligatori per --operation clone.")
        source = io_utils.load_image(args.source)
        mask = build_mask_from_args(args, source.shape[:2])
        if mask.shape != source.shape[:2]:
            sys.exit("Le dimensioni della maschera devono coincidere con quelle della sorgente.")
        offset = (args.offset_top, args.offset_left)
    else:
        # Le operazioni di Sezione 4 selezionano direttamente una regione
        # della destinazione: non esistono ne' sorgente ne' offset.
        source = None
        mask = build_mask_from_args(args, destination.shape[:2])
        if mask.shape != destination.shape[:2]:
            sys.exit("Per questa operazione, la maschera deve coincidere con la destinazione.")
        offset = None

    if args.preview_only and args.operation == "clone":
        save_preview(source, mask, destination, offset)
        return
    if args.preview_only:
        io_utils.save_image("preview_destination.png", overlay_mask(destination, mask, color=(1.0, 0.0, 0.0), alpha=0.4))
        print("Anteprima salvata: preview_destination.png")
        return

    if args.operation == "clone":
        guidance_fn = guidance.GUIDANCE_STRATEGIES[args.guidance]
        result = clone.seamless_clone(source, mask, destination, offset, guidance_fn=guidance_fn)
    elif args.operation == "texture-flatten":
        result = editing.texture_flatten(destination, mask, edge_threshold=args.edge_threshold)
    elif args.operation == "illumination":
        result = editing.local_illumination_change(
            destination, mask, alpha_scale=args.alpha_scale, beta=args.beta
        )
    elif args.operation == "background-decolorize":
        result = editing.background_decolorization(destination, mask)
    elif args.operation == "recolor":
        result = editing.recolor_selection(destination, mask, args.color_factors)
    else:
        result = editing.seamless_tile_selection(destination, mask)
    io_utils.save_image(args.output, result)
    print(f"Fatto. Risultato salvato in: {args.output}")


if __name__ == "__main__":
    main()
