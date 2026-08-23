"""GUI per cloning (Sezione 3) e selection editing (Sezione 4) del paper.

In modalita' cloning la maschera si disegna sulla sorgente e un clic sulla
destinazione ne sceglie il centro. In texture flattening e illumination la
maschera viene invece disegnata direttamente sulla destinazione, perche'
l'immagine viene modificata in-place.
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

import clone
import editing
import guidance
import io_utils


MAX_DISPLAY_SIZE = (520, 520)
MODE_DISPLAY_NAMES = {
    "clone": "Seamless cloning",
    **editing.INPLACE_DISPLAY_NAMES,
}
DISPLAY_TO_MODE = {label: name for name, label in MODE_DISPLAY_NAMES.items()}
DISPLAY_TO_GUIDANCE = {
    guidance.GUIDANCE_DISPLAY_NAMES[name]: name
    for name in guidance.GUIDANCE_STRATEGIES
}


class PoissonEditingApp:
    """Converte l'interazione utente in maschere, parametri e chiamate API."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Poisson Image Editing")

        self.source: np.ndarray | None = None
        self.destination: np.ndarray | None = None
        # In cloning e' nelle coordinate della sorgente; nelle operazioni
        # in-place e' nelle coordinate della destinazione.
        self.mask: np.ndarray | None = None
        self.points: list[tuple[int, int]] = []
        self.offset: tuple[int, int] | None = None
        self.source_scale = self.destination_scale = 1.0
        self._source_photo = self._destination_photo = None

        toolbar = ttk.Frame(root, padding=8)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Apri sorgente", command=self.open_source).pack(side="left")
        ttk.Button(toolbar, text="Apri destinazione", command=self.open_destination).pack(side="left", padx=(6, 0))
        ttk.Button(toolbar, text="Chiudi selezione (Invio)", command=self.finish_selection).pack(side="left", padx=(16, 0))
        ttk.Button(toolbar, text="Cancella selezione", command=self.clear_selection).pack(side="left", padx=(6, 0))

        controls = ttk.Frame(root, padding=(8, 0, 8, 8))
        controls.pack(fill="x")
        ttk.Label(controls, text="Operazione:").pack(side="left")
        self.mode_name = tk.StringVar(value=MODE_DISPLAY_NAMES["clone"])
        mode_box = ttk.Combobox(controls, textvariable=self.mode_name,
                                values=list(DISPLAY_TO_MODE), state="readonly", width=28)
        mode_box.pack(side="left", padx=(4, 12))
        mode_box.bind("<<ComboboxSelected>>", self.on_mode_change)

        ttk.Label(controls, text="Guidance cloning:").pack(side="left")
        self.guidance_name = tk.StringVar(value=next(iter(DISPLAY_TO_GUIDANCE)))
        ttk.Combobox(controls, textvariable=self.guidance_name,
                     values=list(DISPLAY_TO_GUIDANCE), state="readonly", width=31).pack(side="left", padx=(4, 12))
        ttk.Label(controls, text="Soglia bordi:").pack(side="left")
        self.edge_threshold = tk.StringVar(value="0.10")
        ttk.Entry(controls, textvariable=self.edge_threshold, width=6).pack(side="left", padx=(4, 12))
        ttk.Label(controls, text="α:").pack(side="left")
        self.alpha_scale = tk.StringVar(value="0.2")
        ttk.Entry(controls, textvariable=self.alpha_scale, width=5).pack(side="left", padx=(4, 8))
        ttk.Label(controls, text="β:").pack(side="left")
        self.beta = tk.StringVar(value="0.2")
        ttk.Entry(controls, textvariable=self.beta, width=5).pack(side="left", padx=4)
        ttk.Label(controls, text="RGB:").pack(side="left", padx=(8, 0))
        self.color_factors = [tk.StringVar(value=value) for value in ("1.5", "0.5", "0.5")]
        for factor in self.color_factors:
            ttk.Entry(controls, textvariable=factor, width=4).pack(side="left", padx=1)
        self.run_button = ttk.Button(controls, text="Esegui e salva", command=self.run_operation)
        self.run_button.pack(side="right")

        panels = ttk.Frame(root, padding=(8, 0, 8, 8))
        panels.pack(fill="both", expand=True)
        self.source_frame, self.source_canvas = self._make_panel(panels, "1. Sorgente")
        self.destination_frame, self.destination_canvas = self._make_panel(panels, "2. Destinazione")
        self.source_canvas.bind("<Button-1>", self.add_source_point)
        self.destination_canvas.bind("<Button-1>", self.destination_click)
        root.bind("<Return>", lambda _event: self.finish_selection())

        self.status = tk.StringVar(value="Carica le immagini per iniziare.")
        ttk.Label(root, textvariable=self.status, padding=(8, 0, 8, 8)).pack(anchor="w")
        self._refresh_panel_titles()

    @staticmethod
    def _make_panel(parent: ttk.Frame, label: str) -> tuple[ttk.LabelFrame, tk.Canvas]:
        frame = ttk.LabelFrame(parent, text=label, padding=5)
        frame.pack(side="left", fill="both", expand=True, padx=4)
        canvas = tk.Canvas(frame, width=MAX_DISPLAY_SIZE[0], height=MAX_DISPLAY_SIZE[1], bg="#303030", highlightthickness=0)
        canvas.pack()
        return frame, canvas

    @property
    def mode(self) -> str:
        return DISPLAY_TO_MODE[self.mode_name.get()]

    @property
    def is_clone_mode(self) -> bool:
        return self.mode == "clone"

    def _refresh_panel_titles(self) -> None:
        if self.is_clone_mode:
            self.source_frame.configure(text="1. Sorgente: clic per disegnare il contorno")
            self.destination_frame.configure(text="2. Destinazione: clic per scegliere il centro")
        else:
            self.source_frame.configure(text="Sorgente: non richiesta per questa operazione")
            if self.mode == "tile":
                self.destination_frame.configure(text="Destinazione: due clic per gli angoli del rettangolo")
            else:
                self.destination_frame.configure(text="Destinazione: clic per disegnare il contorno")

    @staticmethod
    def _display_image(image: np.ndarray) -> tuple[Image.Image, float]:
        """Riduce soltanto la preview: il solver riceve l'originale."""
        height, width = image.shape[:2]
        scale = min(1.0, MAX_DISPLAY_SIZE[0] / width, MAX_DISPLAY_SIZE[1] / height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        pixels = (np.clip(image, 0, 1) * 255).round().astype(np.uint8)
        return Image.fromarray(pixels, "RGB").resize(size, Image.Resampling.LANCZOS), scale

    @staticmethod
    def _with_mask_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
        overlay = Image.new("RGBA", image.size, (255, 0, 0, 0))
        visible_mask = Image.fromarray((mask * 130).astype(np.uint8), "L").resize(image.size, Image.Resampling.NEAREST)
        overlay.putalpha(visible_mask)
        return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    @staticmethod
    def _mask_in_destination(mask: np.ndarray, destination_shape: tuple[int, int],
                             offset: tuple[int, int]) -> np.ndarray:
        """Trasla la maschera sorgente nel canvas della destinazione.

        Usa la stessa convenzione e lo stesso ritaglio ai bordi di
        ``clone.place_source_in_canvas``: e' solo una preview, non modifica
        ne' la maschera originale ne' i dati passati al solver.
        """
        destination_mask = np.zeros(destination_shape, dtype=bool)
        top, left = offset
        source_height, source_width = mask.shape
        destination_height, destination_width = destination_shape

        source_top = max(0, -top)
        source_left = max(0, -left)
        destination_top = max(0, top)
        destination_left = max(0, left)
        height = min(source_height - source_top, destination_height - destination_top)
        width = min(source_width - source_left, destination_width - destination_left)

        if height > 0 and width > 0:
            destination_mask[
                destination_top:destination_top + height,
                destination_left:destination_left + width,
            ] = mask[source_top:source_top + height, source_left:source_left + width]
        return destination_mask

    @staticmethod
    def _draw_points(canvas: tk.Canvas, points: list[tuple[int, int]], scale: float) -> None:
        shown = [(x * scale, y * scale) for x, y in points]
        for x, y in shown:
            canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#ff3333", outline="white")
        if len(shown) > 1:
            canvas.create_line(*[coordinate for point in shown for coordinate in point], fill="#ff3333", width=2)

    def _show_source(self) -> None:
        if self.source is None:
            return
        image, self.source_scale = self._display_image(self.source)
        if self.is_clone_mode and self.mask is not None:
            image = self._with_mask_overlay(image, self.mask)
        self._source_photo = ImageTk.PhotoImage(image)
        self.source_canvas.delete("all")
        self.source_canvas.create_image(0, 0, anchor="nw", image=self._source_photo)
        if self.is_clone_mode and self.mask is None:
            self._draw_points(self.source_canvas, self.points, self.source_scale)

    def _show_destination(self) -> None:
        if self.destination is None:
            return
        image, self.destination_scale = self._display_image(self.destination)
        if not self.is_clone_mode and self.mask is not None:
            image = self._with_mask_overlay(image, self.mask)
        elif self.is_clone_mode and self.offset is not None and self.mask is not None:
            placed_mask = self._mask_in_destination(
                self.mask, self.destination.shape[:2], self.offset
            )
            image = self._with_mask_overlay(image, placed_mask)
        self._destination_photo = ImageTk.PhotoImage(image)
        self.destination_canvas.delete("all")
        self.destination_canvas.create_image(0, 0, anchor="nw", image=self._destination_photo)
        if not self.is_clone_mode and self.mask is None:
            self._draw_points(self.destination_canvas, self.points, self.destination_scale)

    def open_source(self) -> None:
        path = filedialog.askopenfilename(title="Scegli l'immagine sorgente", filetypes=[("Immagini", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if path:
            self.source = io_utils.load_image(path)
            if self.is_clone_mode:
                self.clear_selection()
            self._show_source()
            self.status.set(f"Sorgente caricata: {Path(path).name}.")

    def open_destination(self) -> None:
        path = filedialog.askopenfilename(title="Scegli l'immagine destinazione", filetypes=[("Immagini", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if path:
            self.destination = io_utils.load_image(path)
            if not self.is_clone_mode:
                self.clear_selection()
            else:
                self.offset = None
            self._show_destination()
            self.status.set(f"Destinazione caricata: {Path(path).name}.")

    def on_mode_change(self, _event=None) -> None:
        self.clear_selection()
        self._refresh_panel_titles()
        self._show_source()
        self._show_destination()
        if self.is_clone_mode:
            self.status.set("Cloning: disegna il contorno sulla sorgente, poi scegli il centro nella destinazione.")
        elif self.mode == "tile":
            self.status.set("Tiling: fai clic su due angoli opposti del rettangolo, poi premi Invio.")
        else:
            self.status.set("Selection editing: disegna il contorno direttamente sulla destinazione.")

    def _add_point(self, event: tk.Event, image: np.ndarray, scale: float, redraw) -> None:
        if self.mask is not None:
            self.status.set("La selezione e' gia' chiusa: usa 'Cancella selezione' per ridisegnarla.")
            return
        x = int(np.clip(round(event.x / scale), 0, image.shape[1] - 1))
        y = int(np.clip(round(event.y / scale), 0, image.shape[0] - 1))
        self.points.append((x, y))
        redraw()
        self.status.set(f"Vertici del contorno: {len(self.points)}. Premi Invio quando hai finito.")

    def add_source_point(self, event: tk.Event) -> None:
        if not self.is_clone_mode:
            self.status.set("In questa modalita' la selezione si disegna sulla destinazione.")
        elif self.source is None:
            self.status.set("Prima carica una sorgente.")
        else:
            self._add_point(event, self.source, self.source_scale, self._show_source)

    def destination_click(self, event: tk.Event) -> None:
        if not self.is_clone_mode:
            if self.destination is None:
                self.status.set("Prima carica una destinazione.")
            elif self.mode == "tile" and len(self.points) >= 2:
                self.status.set("Per il tiling servono solo due angoli: premi Invio o cancella la selezione.")
            else:
                self._add_point(event, self.destination, self.destination_scale, self._show_destination)
            return
        self.set_destination_position(event)

    def finish_selection(self) -> None:
        image = self.source if self.is_clone_mode else self.destination
        minimum_points = 2 if self.mode == "tile" else 3
        if image is None or len(self.points) < minimum_points:
            self.status.set("Per il tiling servono due angoli; per le altre modalita' almeno tre vertici.")
            return
        if self.mode == "tile":
            (x0, y0), (x1, y1) = self.points[:2]
            left, right = sorted((x0, x1))
            top, bottom = sorted((y0, y1))
            self.mask = np.zeros(image.shape[:2], dtype=bool)
            self.mask[top:bottom + 1, left:right + 1] = True
        else:
            polygon = Image.new("L", (image.shape[1], image.shape[0]), 0)
            ImageDraw.Draw(polygon).polygon(self.points, fill=255)
            self.mask = np.asarray(polygon, dtype=bool)
        self.points.clear()
        self.offset = None
        self._show_source()
        self._show_destination()
        message = "Maschera creata. Ora clicca nella destinazione per scegliere il centro dell'incolla." if self.is_clone_mode else "Maschera creata. Puoi eseguire l'operazione sulla destinazione."
        self.status.set(message)

    def clear_selection(self) -> None:
        self.mask = None
        self.points.clear()
        self.offset = None
        self._show_source()
        self._show_destination()

    def set_destination_position(self, event: tk.Event) -> None:
        if self.destination is None or self.mask is None:
            self.status.set("Prima carica la destinazione e completa la selezione sulla sorgente.")
            return
        center_x, center_y = round(event.x / self.destination_scale), round(event.y / self.destination_scale)
        ys, xs = np.nonzero(self.mask)
        self.offset = (round(center_y - (ys.min() + ys.max()) / 2), round(center_x - (xs.min() + xs.max()) / 2))
        self._show_destination()
        self.status.set("Posizione scelta. Premi 'Esegui e salva'.")

    def _parameters(self) -> tuple[float, float, float, tuple[float, float, float]]:
        try:
            color_factors = tuple(float(factor.get()) for factor in self.color_factors)
            return float(self.edge_threshold.get()), float(self.alpha_scale.get()), float(self.beta.get()), color_factors
        except ValueError as error:
            raise ValueError("Soglia bordi, alpha, beta e fattori RGB devono essere numeri.") from error

    def run_operation(self) -> None:
        if self.destination is None or self.mask is None:
            messagebox.showerror("Dati mancanti", "Carica la destinazione e completa la selezione.")
            return
        if self.is_clone_mode and (self.source is None or self.offset is None):
            messagebox.showerror("Dati mancanti", "Per il cloning servono sorgente e posizione nella destinazione.")
            return
        try:
            parameters = self._parameters()
        except ValueError as error:
            messagebox.showerror("Parametri non validi", str(error))
            return
        output = filedialog.asksaveasfilename(title="Salva il risultato", defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg")])
        if not output:
            return
        self.run_button.configure(state="disabled")
        self.status.set("Calcolo in corso: la finestra resta utilizzabile...")
        threading.Thread(target=self._solve_and_save, args=(output, parameters), daemon=True).start()

    def _solve_and_save(self, output: str, parameters: tuple[float, float, float, tuple[float, float, float]]) -> None:
        try:
            edge_threshold, alpha_scale, beta, color_factors = parameters
            if self.is_clone_mode:
                name = DISPLAY_TO_GUIDANCE[self.guidance_name.get()]
                result = clone.seamless_clone(self.source, self.mask, self.destination, self.offset, guidance.GUIDANCE_STRATEGIES[name])
            elif self.mode == "texture-flatten":
                result = editing.texture_flatten(self.destination, self.mask, edge_threshold)
            elif self.mode == "illumination":
                result = editing.local_illumination_change(self.destination, self.mask, alpha_scale, beta)
            elif self.mode == "background-decolorize":
                result = editing.background_decolorization(self.destination, self.mask)
            elif self.mode == "recolor":
                result = editing.recolor_selection(self.destination, self.mask, color_factors)
            else:
                result = editing.seamless_tile_selection(self.destination, self.mask)
            io_utils.save_image(output, result)
        except Exception as error:
            self.root.after(0, lambda: self._finish_run(error))
        else:
            self.root.after(0, lambda: self._finish_run(None, output))

    def _finish_run(self, error: Exception | None, output: str | None = None) -> None:
        self.run_button.configure(state="normal")
        if error is not None:
            self.status.set("Errore durante il calcolo.")
            messagebox.showerror("Poisson Image Editing", str(error))
        else:
            self.status.set(f"Risultato salvato in: {output}")
            messagebox.showinfo("Poisson Image Editing", f"Risultato salvato in:\n{output}")


def main() -> None:
    root = tk.Tk()
    PoissonEditingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
