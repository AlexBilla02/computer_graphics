"""Esegue test di Poisson Image Editing descritti in file YAML.

Ogni test usa le API esistenti del progetto; questo modulo serve solo a
caricare immagini, maschere e parametri da file per rendere i test
ripetibili da un unico comando.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

try:
    import yaml
except ImportError as error:
    raise SystemExit(
        "Per eseguire i test YAML installa le dipendenze: "
        "python3 -m pip install -r requirements.txt"
    ) from error

import clone
import editing
import guidance
import io_utils


_PREVIEW_RED = np.array((1.0, 0.0, 0.0), dtype=np.float64)


TEMPLATE = {
    "name": "seam1",
    "operation": "clone",
    "source": "img/sorgente.png",
    "destination": "img/destinazione.png",
    "mask": "mask/maschera_sorgente.png",
    "position": {"x": 250, "y": 180},
    "guidance": "import",
    "output": "results/seam1.png",
}


def _fail(message: str) -> None:
    raise ValueError(message)


def _required(test: dict, key: str, label: str):
    value = test.get(key)
    if value is None:
        _fail(f"{label}: campo YAML obbligatorio mancante: '{key}'.")
    return value


def _offset_from_position(mask, position: object, label: str) -> tuple[int, int]:
    """Converte il centro x/y nello stesso offset usato dalla GUI."""
    if not isinstance(position, dict) or "x" not in position or "y" not in position:
        _fail(f"{label}: 'position' deve contenere 'x' e 'y'.")
    try:
        center_x, center_y = float(position["x"]), float(position["y"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: position.x e position.y devono essere numeri.") from error
    ys, xs = mask.nonzero()
    if len(ys) == 0:
        _fail(f"{label}: la maschera e' vuota.")
    return (
        round(center_y - (ys.min() + ys.max()) / 2),
        round(center_x - (xs.min() + xs.max()) / 2),
    )


def _load_tests(path: Path) -> list[tuple[Path, dict]]:
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"YAML non valido in {path}: {error}") from error
    if not isinstance(content, dict):
        _fail(f"{path}: il contenuto deve essere un oggetto YAML.")
    raw_tests = content.get("tests", [content])
    if not isinstance(raw_tests, list) or not all(isinstance(test, dict) for test in raw_tests):
        _fail(f"{path}: 'tests' deve essere una lista di test.")
    return [(path, test) for test in raw_tests]


def _config_paths(config_path: Path) -> list[Path]:
    if config_path.is_file():
        return [config_path]
    if config_path.is_dir():
        files = sorted((*config_path.glob("*.yaml"), *config_path.glob("*.yml")))
        if files:
            return files
        _fail(f"Nessun file .yaml o .yml trovato in {config_path}.")
    _fail(f"Config YAML non trovata: {config_path}.")


def run_test(test: dict, config_path: Path, preview_only: bool = False) -> Path | None:
    """Esegue una singola descrizione YAML e restituisce l'output, se creato."""
    name = str(test.get("name", config_path.stem))
    label = f"{config_path} ({name})"
    operation = _required(test, "operation", label)
    if operation not in {"clone", *editing.INPLACE_OPERATIONS}:
        _fail(f"{label}: operation non supportata: {operation!r}.")

    destination = io_utils.load_image(_required(test, "destination", label))
    mask = io_utils.load_mask(_required(test, "mask", label))
    expected_shape = None

    if operation == "clone":
        source = io_utils.load_image(_required(test, "source", label))
        expected_shape = source.shape[:2]
        if mask.shape != expected_shape:
            _fail(f"{label}: la maschera deve avere le dimensioni della sorgente.")
        offset = _offset_from_position(mask, _required(test, "position", label), label)
        aligned_source, aligned_mask = clone.place_source_in_canvas(
            source, mask, destination.shape[:2], offset
        )
        if preview_only:
            preview = destination.copy()
            preview[aligned_mask] = 0.6 * preview[aligned_mask] + 0.4 * _PREVIEW_RED
            output = Path(_required(test, "output", label)).with_name(f"{name}_preview.png")
            output.parent.mkdir(parents=True, exist_ok=True)
            io_utils.save_image(output, preview)
            print(f"[{name}] anteprima salvata: {output} (x/y -> offset={offset})")
            return output
        guidance_name = test.get("guidance", "import")
        if guidance_name not in guidance.GUIDANCE_STRATEGIES:
            _fail(f"{label}: guidance non supportata: {guidance_name!r}.")
        result = clone.seamless_clone(
            source, mask, destination, offset, guidance.GUIDANCE_STRATEGIES[guidance_name]
        )
    else:
        expected_shape = destination.shape[:2]
        if mask.shape != expected_shape:
            _fail(f"{label}: per {operation}, la maschera deve avere le dimensioni della destinazione.")
        if preview_only:
            preview = destination.copy()
            preview[mask] = 0.6 * preview[mask] + 0.4 * _PREVIEW_RED
            output = Path(_required(test, "output", label)).with_name(f"{name}_preview.png")
            output.parent.mkdir(parents=True, exist_ok=True)
            io_utils.save_image(output, preview)
            print(f"[{name}] anteprima salvata: {output}")
            return output
        if operation == "texture-flatten":
            result = editing.texture_flatten(destination, mask, float(test.get("edge_threshold", 0.10)))
        elif operation == "illumination":
            result = editing.local_illumination_change(
                destination, mask,
                alpha_scale=float(test.get("alpha_scale", 0.2)),
                beta=float(test.get("beta", 0.2)),
            )
        elif operation == "background-decolorize":
            result = editing.background_decolorization(destination, mask)
        elif operation == "recolor":
            result = editing.recolor_selection(destination, mask, test.get("color_factors", (1.5, 0.5, 0.5)))
        else:
            result = editing.seamless_tile_selection(destination, mask)

    output = Path(_required(test, "output", label))
    output.parent.mkdir(parents=True, exist_ok=True)
    io_utils.save_image(output, result)
    print(f"[{name}] completato: {output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Esegue test descritti in YAML.")
    parser.add_argument("config", nargs="?", help="File YAML oppure cartella contenente i test.")
    parser.add_argument("--preview-only", action="store_true", help="Salva solo l'anteprima della maschera.")
    parser.add_argument("--write-template", metavar="FILE", help="Crea un template YAML modificabile e termina.")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.write_template:
        target = Path(args.write_template)
        if target.exists():
            sys.exit(f"Il file esiste gia': {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(TEMPLATE, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"Template creato: {target}")
        return
    if not args.config:
        sys.exit("Specifica un file/cartella YAML oppure usa --write-template FILE.")
    for config_path in _config_paths(Path(args.config)):
        for _, test in _load_tests(config_path):
            run_test(test, config_path, args.preview_only)


if __name__ == "__main__":
    main()
