"""
poisson_editing
================

Implementazione da zero (senza OpenCV) del paper "Poisson Image Editing"
(Perez, Gangnet, Blake - SIGGRAPH 2003).

Moduli:
    io_utils  -> caricare/salvare immagini su disco
    domain    -> gestione della regione selezionata Omega e del suo bordo
    guidance  -> diversi campi guida v (cloning, mixing, flattening, luce)
    solver    -> costruzione e risoluzione del sistema lineare sparso
    clone     -> funzione ad alto livello: seamless_clone(...)
    editing   -> texture, luce, colore e tiling (Sezione 4)

Stato attuale: tutte le funzionalita' descritte nel paper sono disponibili
da GUI e riga di comando.
"""
