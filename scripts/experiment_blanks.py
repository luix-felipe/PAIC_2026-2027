"""Ensaio de leitura com ROIs manuais; reconhecimento de motherboard não avaliado."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter
from typing import Protocol

import cv2
import numpy as np
import yaml
import zxingcpp


VARIANTS = ("raw", "clahe", "otsu")


@dataclass(frozen=True)
class Symbol:
    text: str
    data: bytes


class Decoder(Protocol):
    def decode(self, roi: np.ndarray) -> tuple[Symbol, ...]: ...


class ZXingDecoder:
    def decode(self, roi: np.ndarray) -> tuple[Symbol, ...]:
        results = zxingcpp.read_barcodes(
            roi,
            formats=zxingcpp.BarcodeFormat.DataMatrix,
            try_rotate=True,
            try_invert=True,
            try_downscale=False,
        )
        return tuple(Symbol(result.text, bytes(result.bytes)) for result in results)


def extract_roi(frame: np.ndarray, bounds: list[int]) -> np.ndarray:
    """Pixels originais, limites [x0, y0, x1, y1) sem truncamento nem resize."""
    if len(bounds) != 4 or any(type(value) is not int for value in bounds):
        raise ValueError("ROI deve conter quatro coordenadas inteiras")
    x0, y0, x1, y1 = bounds
    height, width = frame.shape[:2]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("ROI fora do frame ou com área inválida")
    return frame[y0:y1, x0:x1]


def preprocess_roi(roi: np.ndarray, variant: str) -> np.ndarray:
    if variant == "raw":
        return roi
    if variant == "clahe":
        # Parâmetros exploratórios, sem calibração de produção.
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi)
    if variant == "otsu":
        return cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    raise ValueError(f"Tratamento desconhecido: {variant}")


def inspect_sample(
    frame: np.ndarray, rois: list[list[int]], variants: list[str], decoder: Decoder
) -> dict:
    attempts = []
    for index, bounds in enumerate(rois):
        start = perf_counter()
        try:
            roi = extract_roi(frame, bounds)
        except ValueError as error:
            attempts.append({"roi_index": index, "status": "INVALID_ROI", "error": str(error)})
            continue
        crop_ms = (perf_counter() - start) * 1000
        for variant in variants:
            start = perf_counter()
            prepared = preprocess_roi(roi, variant)
            preprocess_ms = (perf_counter() - start) * 1000
            start = perf_counter()
            symbols = decoder.decode(prepared)
            decode_ms = (perf_counter() - start) * 1000
            attempts.append({
                "roi_index": index,
                "roi_xyxy": bounds,
                "variant": variant,
                "status": "READ" if symbols else "NO_READ",
                "symbol_count": len(symbols),
                # Conteúdo real fica apenas na interface do decoder, fora do relatório.
                "crop_ms": crop_ms,
                "preprocess_ms": preprocess_ms,
                "decode_ms": decode_ms,
            })
    return {
        "motherboard_detection": "NOT_EVALUATED",
        "status": "READ" if any(a["status"] == "READ" for a in attempts) else "NO_READ",
        "attempts": attempts,
    }


def run(config_path: Path, variants: list[str]) -> dict:
    if not variants or any(item not in VARIANTS for item in variants):
        raise ValueError("Selecione pelo menos um tratamento válido")
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    dataset = config_path.parent / config["dataset"]
    samples = []
    decoder = ZXingDecoder()
    for sample in config["samples"]:
        path = dataset / sample["image"]
        start = perf_counter()
        # IMREAD_GRAYSCALE respeita o EXIF no OpenCV testado. Não há resize.
        # Cinza já na aquisição evita guardar também uma cópia colorida de 50 MP.
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        load_ms = (perf_counter() - start) * 1000
        if frame is None:
            result = {"status": "CAPTURE_FAILED", "attempts": []}
        elif [frame.shape[1], frame.shape[0]] != sample["frame_size"]:
            result = {"status": "FRAME_SIZE_MISMATCH", "attempts": []}
        else:
            result = inspect_sample(frame, sample["rois"], variants, decoder)
        result.update({"image": sample["image"], "load_ms": load_ms})
        samples.append(result)
    return {
        "experiment": "blanks_manual_roi",
        "unit": "panel",
        "class": "motherboard",
        "motherboard_detection": "NOT_EVALUATED",
        "reference_content_validation": "NOT_AVAILABLE",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "opencv-python", "PyYAML", "zxing-cpp")},
        },
        "summary": {
            "images": len(samples),
            "images_with_any_read": sum(s["status"] == "READ" for s in samples),
            "images_with_read_by_variant": {
                variant: sum(
                    any(a.get("variant") == variant and a["status"] == "READ" for a in s["attempts"])
                    for s in samples
                )
                for variant in variants
            },
        },
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/blanks_manual_rois.yaml"))
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=["raw"])
    parser.add_argument("--output", type=Path, default=Path("artifacts/blanks-report.json"))
    args = parser.parse_args()
    report = run(args.config, args.variants)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"Relatório: {args.output}")


if __name__ == "__main__":
    main()

