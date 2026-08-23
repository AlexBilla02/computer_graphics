# Poisson Image Editing

Implementazione con NumPy e SciPy del paper *Poisson Image Editing* (Pérez, Gangnet, Blake, 2003). La parte matematica è separata dalla UI: `guidance.py` contiene i campi guida, `solver.py` risolve il sistema sparso, `clone.py` orchestra il cloning e `editing.py` raccoglie le operazioni in-place della Sezione 4.

## Interfaccia grafica

Avvia (nell'ambiente corrente, l'interprete di sistema include Tkinter):

```bash
python gui.py
```

1. Apri sorgente e destinazione.
2. Sulla sorgente, fai clic sui vertici del contorno dell'oggetto; premi `Invio` per chiuderlo.
3. Sulla destinazione, fai clic nel punto che deve corrispondere al centro dell'oggetto.
4. Per *Seamless cloning*, seleziona *Import gradients* o *Mixing gradients*, poi salva il risultato.

Per *Texture flattening*, *Local illumination changes*, *Background decolorization* e *Local recoloring*, seleziona prima l'operazione: disegnerai il contorno direttamente sulla destinazione. La soglia dei bordi controlla quanto il texture flattening è aggressivo (più alta = meno dettagli conservati). I valori predefiniti `α = 0.2` e `β = 0.2` corrispondono alla formula (16) del paper per l'illuminazione locale. I tre campi `RGB` definiscono i fattori di ricolorazione; il preset `(1.5, 0.5, 0.5)` è quello mostrato nel paper.

Per *Seamless tiling*, carica l'immagine e fai due clic su angoli opposti del rettangolo da trasformare in tile. Il rettangolo risultante avrà bordi opposti identici e potrà essere ripetuto senza cuciture.

La GUI richiede Tkinter. Se l'interprete del virtual environment segnala `No module named '_tkinter'`, avviala con un Python che includa Tkinter (ad esempio `python3 gui.py` nel sistema corrente) oppure ricrea l'ambiente con una distribuzione Python compilata con supporto Tk.

## Riga di comando

Il comando diretto è supportato:

```bash
python cli.py --source sorgente.jpg --destination sfondo.jpg --mask maschera.png --offset-top 40 --offset-left 60 --output risultato.png
```

Una maschera è un'immagine bianca/nera delle stesse dimensioni della sorgente: bianco è la regione da clonare.

Per le operazioni in-place, la maschera ha invece le dimensioni della destinazione:

```bash
python cli.py --operation texture-flatten --destination foto.jpg --mask selezione.png --edge-threshold 0.10 --output piatta.png
python cli.py --operation illumination --destination foto.jpg --mask selezione.png --alpha-scale 0.2 --beta 0.2 --output illuminata.png
```

```bash
python cli.py --operation background-decolorize --destination foto.jpg --mask selezione.png --output sfondo_grigio.png
python cli.py --operation recolor --destination foto.jpg --mask selezione.png --color-factors 1.5 0.5 0.5 --output ricolorata.png
python cli.py --operation tile --destination texture.png --shape rectangle --top 20 --left 20 --height 200 --width 200 --output tileabile.png
```

## Estendere l'algoritmo

Per aggiungere un nuovo campo guida, crea in `guidance.py` una funzione con firma `funzione(source, destination, mask) -> {"N", "S", "E", "W"}`. Per un nuovo cloning, registrala in `GUIDANCE_STRATEGIES`; per una nuova operazione in-place, crea un wrapper in `editing.py` e registralo in `INPLACE_OPERATIONS`. Il solver non deve essere modificato.
