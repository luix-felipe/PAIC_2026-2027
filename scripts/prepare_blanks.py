"""Anotação manual, visualização de augmentation e grupos físicos de Blanks.

Execute com python -m scripts.prepare_blanks --help, na raiz do workspace.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from scripts.augmentation import augment, letterbox, validate_box


MANIFEST = Path("config/blanks_annotations.yaml")
POLICY = Path("config/augmentation.example.yaml")
PIECES = ("piece_01", "piece_02", "piece_03")
FACES = ("unknown", "top", "bottom")


def read_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    temporary.replace(path)


def initialize(manifest_path, dataset_path):
    if manifest_path.exists():
        raise ValueError("O manifesto já existe; as anotações foram preservadas")
    import os
    files = sorted(p for p in dataset_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise ValueError("Nenhuma imagem encontrada")
    data = {
        "version": 1,
        "dataset": Path(os.path.relpath(dataset_path.resolve(), manifest_path.parent.resolve())).as_posix(),
        "class_name": "motherboard",
        "unit": "panel",
        "coordinate_system": "exif_oriented_pixels_xyxy_exclusive",
        "samples": [
            {"image": p.name, "piece_id": None, "face": "unknown", "frame_size": None,
             "box": None, "status": "pending"} for p in files
        ],
    }
    save_yaml(manifest_path, data)
    return data


def load_frame(manifest_path, data, sample):
    path = manifest_path.parent / data["dataset"] / sample["image"]
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Falha ao abrir {sample['image']}")
    size = [frame.shape[1], frame.shape[0]]
    if sample["frame_size"] is not None and sample["frame_size"] != size:
        raise ValueError("Resolução/orientação diverge da anotação salva")
    return frame


def selection_to_original(rect, display_size, original_size):
    """A seleção ocorre numa visualização reduzida; a caixa salva é original."""
    x, y, width, height = rect
    if width <= 0 or height <= 0:
        raise ValueError("Seleção cancelada ou vazia")
    sx, sy = np.asarray(original_size) / np.asarray(display_size)
    box = [int(np.floor(x * sx)), int(np.floor(y * sy)),
           int(np.ceil((x + width) * sx)), int(np.ceil((y + height) * sy))]
    validate_box(box, original_size)
    return box


def ask_choice(label, choices, current=None):
    while True:
        reply = input(f"{label} {choices} [{current or ''}]: ").strip() or current
        if reply in choices:
            return reply
        print("Selecione um dos valores indicados.")


def annotate(manifest_path, image=None, review=False):
    data = read_yaml(manifest_path)
    selected = [s for s in data["samples"] if image is None or s["image"] == image]
    if not selected:
        raise ValueError("Imagem não encontrada no manifesto")
    previous_piece = None
    for sample in selected:
        if sample["status"] == "reviewed" and not review and image is None:
            continue
        frame = load_frame(manifest_path, data, sample)
        height, width = frame.shape[:2]
        ratio = min(1200 / width, 800 / height, 1.0)
        display = cv2.resize(frame, (round(width * ratio), round(height * ratio)))
        print(f"\n{sample['image']}: marque o PAINEL INTEIRO. Enter confirma; C cancela a seleção.")
        print("A caixa será redesenhada; seleções canceladas mantêm a anotação anterior.")
        try:
            rect = cv2.selectROI("Painel inteiro - Enter confirma / C cancela", display,
                                 showCrosshair=True, fromCenter=False)
        finally:
            cv2.destroyAllWindows()
        if rect[2] == 0 or rect[3] == 0:
            continue
        box = selection_to_original(rect, (display.shape[1], display.shape[0]), (width, height))
        piece = ask_choice("Peça física", PIECES, sample["piece_id"] or previous_piece)
        face = ask_choice("Face (pode permanecer unknown)", FACES, sample["face"])
        sample.update(piece_id=piece, face=face, frame_size=[width, height], box=box, status="reviewed")
        save_yaml(manifest_path, data)
        previous_piece = piece
        print(f"Salvo: {sample['image']} / {piece} / {box}")
    print(dict(Counter(s["status"] for s in data["samples"])))


def piece_folds(samples):
    """Três rodadas exploratórias; fotos e faces da mesma peça nunca se separam."""
    if not samples or any(s["status"] != "reviewed" for s in samples):
        raise ValueError("Revise todas as anotações antes de gerar as divisões")
    if any(s["piece_id"] not in PIECES for s in samples):
        raise ValueError("Toda foto deve ter uma peça física conhecida")
    if len({s["image"] for s in samples}) != len(samples):
        raise ValueError("Uma imagem não pode aparecer em mais de uma entrada")
    for sample in samples:
        validate_box(sample["box"], sample["frame_size"])
    if {s["piece_id"] for s in samples} != set(PIECES):
        raise ValueError("São necessários os três grupos físicos confirmados")
    return [
        {"held_out_piece": piece,
         "train": [s["image"] for s in samples if s["piece_id"] != piece],
         "validation": [s["image"] for s in samples if s["piece_id"] == piece]}
        for piece in PIECES
    ]


def make_preview(manifest_path, policy_path, image, output, allow_draft=False):
    data, policy = read_yaml(manifest_path), read_yaml(policy_path)
    sample = next((s for s in data["samples"] if s["image"] == image), None)
    if sample is None:
        raise ValueError("Imagem não encontrada no manifesto")
    allowed = {"reviewed", "draft"} if allow_draft else {"reviewed"}
    if sample["status"] not in allowed or sample["box"] is None:
        raise ValueError("Revise a caixa; --allow-draft permite visualizar apenas um rascunho")
    if sample["piece_id"] not in PIECES:
        raise ValueError("Peça física não definida")
    frame = load_frame(manifest_path, data, sample)
    size = policy["preview_size"]
    base, box, source_to_input = letterbox(frame, sample["box"], size)
    rng = np.random.default_rng(policy["seed"])
    variants = [("original", {})] + [(name, params) for name, params in policy["isolated"].items()]
    variants += [(f"random_{i+1}", {name: float(rng.uniform(*limits))
                                  for name, limits in policy["random_ranges"].items()})
                 for i in range(policy["random_examples"])]
    cells, records = [], []
    for name, params in variants:
        record = {"name": name, "parameters": params}
        try:
            candidate, new_box, center, matrix = augment(base, box, **params)
            x0, y0, x1, y1 = np.rint(new_box).astype(int)
            cv2.rectangle(candidate, (x0, y0), (x1-1, y1-1), (0, 255, 0), 2)
            cv2.drawMarker(candidate, tuple(np.rint(center).astype(int)), (0, 0, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
            record.update(status="VALID", box=new_box.tolist(), center=center.tolist(),
                          input_to_augmented=matrix.tolist())
        except ValueError as error:
            candidate = base.copy()
            cv2.putText(candidate, "REJECTED: panel cut", (8, size//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            record.update(status="REJECTED", error=str(error))
        label = np.full((45, size, 3), 245, dtype=np.uint8)
        cv2.putText(label, f"{name} [{sample['status']}]", (8, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
        cells.append(np.vstack((candidate, label)))
        records.append(record)
    while len(cells) % 3:
        cells.append(np.full_like(cells[0], 245))
    sheet = np.vstack([np.hstack(cells[i:i+3]) for i in range(0, len(cells), 3)])
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise ValueError("Falha ao salvar a visualização")
    report = {"purpose": "visual_inspection_only", "image": image, "piece_id": sample["piece_id"],
              "annotation_status": sample["status"], "source_frame_size": sample["frame_size"],
              "source_box": sample["box"], "policy": policy,
              "source_to_input": source_to_input.tolist(), "variants": records}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Cria lista pendente sem sobrescrever marcações")
    init.add_argument("--dataset", type=Path, default=Path("Datasets/Blanks"))
    annotation = sub.add_parser("annotate", help="Marcação manual com mouse")
    annotation.add_argument("--image")
    annotation.add_argument("--review", action="store_true", help="Inclui fotos já revisadas")
    preview = sub.add_parser("preview", help="Gera comparação visual, sem criar dataset de treino")
    preview.add_argument("--image", required=True)
    preview.add_argument("--policy", type=Path, default=POLICY)
    preview.add_argument("--output", type=Path, default=Path("artifacts/augmentation-preview.jpg"))
    preview.add_argument("--allow-draft", action="store_true")
    folds = sub.add_parser("folds", help="Exporta grupos após revisão de todas as fotos")
    folds.add_argument("--output", type=Path, default=Path("artifacts/blanks-folds.json"))
    args = parser.parse_args()
    try:
        if args.command == "init":
            data = initialize(args.manifest, args.dataset)
            print(f"Criadas {len(data['samples'])} entradas pendentes")
        elif args.command == "annotate":
            annotate(args.manifest, args.image, args.review)
        elif args.command == "preview":
            report = make_preview(args.manifest, args.policy, args.image, args.output, args.allow_draft)
            print(f"Visualização: {args.output}; {Counter(r['status'] for r in report['variants'])}")
        elif args.command == "folds":
            result = piece_folds(read_yaml(args.manifest)["samples"])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"purpose": "exploratory_grouped_validation",
                                              "folds": result}, indent=2), encoding="utf-8")
            print(f"Divisões: {args.output}")
    except (ValueError, OSError, cv2.error) as error:
        parser.exit(1, f"Erro: {error}\n")
    except (KeyboardInterrupt, EOFError):
        parser.exit(0, "\nInterrompido; marcações já salvas foram preservadas.\n")


if __name__ == "__main__":
    main()

