from copy import deepcopy
import csv
import struct

import cv2
import numpy as np
import pytest

from scripts.prepare_blanks import (
    annotate, export_catalog, initialize, load_frame, make_preview, piece_folds, read_yaml,
    save_yaml, selection_to_original,
)


def example_manifest(tmp_path, status="draft"):
    image = np.zeros((100, 150, 3), dtype=np.uint8)
    image[20:70, 30:90] = (200, 100, 80)
    cv2.imwrite(str(tmp_path / "sample.png"), image)
    path = tmp_path / "annotations.yaml"
    data = {"dataset": ".", "class_name": "motherboard", "unit": "panel", "samples": [{
        "image": "sample.png", "piece_id": "piece_01", "face": "unknown",
        "frame_size": [150, 100], "box": [30, 20, 90, 70], "status": status,
    }]}
    save_yaml(path, data)
    return path


def test_selection_maps_to_original_with_separate_axis_scales():
    assert selection_to_original((10, 20, 30, 40), (100, 100), (700, 300)) == [70, 60, 280, 180]


def test_cancelled_selection_is_explicit():
    with pytest.raises(ValueError):
        selection_to_original((0, 0, 0, 0), (100, 100), (1000, 1000))


def test_initialization_does_not_assign_physical_groups_or_overwrite(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "one.jpg").write_bytes(b"not read during inventory")
    path = tmp_path / "config" / "annotations.yaml"
    result = initialize(path, images)
    assert result["samples"][0]["status"] == "pending"
    assert result["samples"][0]["piece_id"] is None
    previous = path.read_bytes()
    with pytest.raises(ValueError, match="já existe"):
        initialize(path, images)
    assert path.read_bytes() == previous


def reviewed_samples():
    return [{"image": f"piece_{piece:02}_{face}.png", "piece_id": f"piece_{piece:02}",
             "face": face, "box": [1, 1, 9, 9], "frame_size": [10, 10], "status": "reviewed"}
            for piece in range(1, 4) for face in ("top", "bottom")]


def test_grouped_folds_keep_top_and_bottom_together():
    samples = reviewed_samples()
    by_image = {s["image"]: s["piece_id"] for s in samples}
    folds = piece_folds(samples)
    assert len(folds) == 3
    for fold in folds:
        train_groups = {by_image[name] for name in fold["train"]}
        validation_groups = {by_image[name] for name in fold["validation"]}
        assert train_groups.isdisjoint(validation_groups)
        assert len(train_groups) == 2
        assert validation_groups == {fold["held_out_piece"]}
        assert set(fold["train"]) | set(fold["validation"]) == set(by_image)


@pytest.mark.parametrize("status", ["draft", "pending"])
def test_folds_refuse_unreviewed_annotations(status):
    samples = reviewed_samples()
    samples[0]["status"] = status
    with pytest.raises(ValueError, match="Revise"):
        piece_folds(samples)


def test_duplicate_source_image_cannot_leak_between_groups():
    samples = reviewed_samples()
    duplicate = deepcopy(samples[0])
    duplicate["piece_id"] = "piece_02"
    with pytest.raises(ValueError, match="mais de uma"):
        piece_folds(samples + [duplicate])


def test_preview_requires_explicit_draft_opt_in_and_preserves_source(tmp_path):
    manifest = example_manifest(tmp_path)
    policy = tmp_path / "policy.yaml"
    save_yaml(policy, {"preview_size": 100, "seed": 2027, "isolated": {"brightness": {"brightness": 15}},
                       "random_examples": 2, "random_ranges": {"angle": [-3, 3]}})
    source_bytes = (tmp_path / "sample.png").read_bytes()
    output = tmp_path / "output.jpg"
    with pytest.raises(ValueError, match="Revise"):
        make_preview(manifest, policy, "sample.png", output)
    first = make_preview(manifest, policy, "sample.png", output, allow_draft=True)
    second = make_preview(manifest, policy, "sample.png", output, allow_draft=True)
    assert first == second
    assert first["purpose"] == "visual_inspection_only"
    assert all(v["status"] == "VALID" for v in first["variants"])
    assert (tmp_path / "sample.png").read_bytes() == source_bytes
    assert read_yaml(manifest)["samples"][0]["status"] == "draft"
    assert cv2.imread(str(output)).shape == (290, 300, 3)


def test_annotator_saves_reviewed_selection_with_mocked_gui(tmp_path, monkeypatch):
    manifest = example_manifest(tmp_path)
    monkeypatch.setattr(cv2, "selectROI", lambda *a, **kw: (10, 10, 50, 60))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    answers = iter(["piece_02", "bottom"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    annotate(manifest)
    sample = read_yaml(manifest)["samples"][0]
    assert sample["status"] == "reviewed"
    assert sample["piece_id"] == "piece_02"
    assert sample["face"] == "bottom"
    assert sample["box"] == [10, 10, 60, 70]
    assert sample["annotation_source"] == "manual_gui"


def test_annotator_cancel_keeps_previous_draft(tmp_path, monkeypatch):
    manifest = example_manifest(tmp_path)
    before = manifest.read_bytes()
    monkeypatch.setattr(cv2, "selectROI", lambda *a, **kw: (0, 0, 0, 0))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    annotate(manifest)
    assert manifest.read_bytes() == before


def test_original_frame_respects_exif_orientation(tmp_path):
    # JPEG sintético 20x10 com orientação EXIF 6, exibido como 10x20.
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    tiff = b"II" + struct.pack("<HI", 42, 8)
    tiff += struct.pack("<H", 1) + struct.pack("<HHI", 274, 3, 1)
    tiff += struct.pack("<H", 6) + b"\x00\x00" + struct.pack("<I", 0)
    payload = b"Exif\x00\x00" + tiff
    jpeg = encoded.tobytes()
    (tmp_path / "rotated.jpg").write_bytes(
        jpeg[:2] + b"\xff\xe1" + struct.pack(">H", len(payload)+2) + payload + jpeg[2:]
    )
    sample = {"image": "rotated.jpg", "frame_size": [10, 20]}
    frame = load_frame(tmp_path / "annotations.yaml", {"dataset": "."}, sample)
    assert frame.shape == (20, 10, 3)


def test_changed_resolution_refuses_stale_annotation(tmp_path):
    path = example_manifest(tmp_path)
    data = read_yaml(path)
    data["samples"][0]["frame_size"] = [10, 10]
    with pytest.raises(ValueError, match="diverge"):
        load_frame(path, data, data["samples"][0])


def test_preview_marks_cropped_panel_as_rejected(tmp_path):
    path = example_manifest(tmp_path)
    policy = tmp_path / "policy.yaml"
    save_yaml(policy, {"preview_size": 100, "seed": 2027,
                       "isolated": {"outside": {"tx": 2}},
                       "random_examples": 0, "random_ranges": {}})
    report = make_preview(path, policy, "sample.png", tmp_path / "preview.jpg", allow_draft=True)
    assert report["variants"][0]["status"] == "VALID"
    assert report["variants"][1]["status"] == "REJECTED"
    assert "box" not in report["variants"][1]


def test_catalog_exports_original_coordinates_and_preserves_annotations(tmp_path):
    manifest = example_manifest(tmp_path, status="reviewed")
    data = read_yaml(manifest)
    data["samples"][0].update(side_id="side_A", edge_contact=["left"],
                              annotation_source="assistant_visual")
    data["samples"][0]["datamatrix_roi_candidate"] = {"box": [35, 25, 40, 30]}
    save_yaml(manifest, data)
    source_before = (tmp_path / "sample.png").read_bytes()
    manifest_before = manifest.read_bytes()
    target = tmp_path / "catalog"
    summary = export_catalog(manifest, target)
    assert summary == {"images": 1, "pieces": {"piece_01": 1},
                       "statuses": {"reviewed": 1}, "sheets": 1}
    with (target / "catalog.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert [int(rows[0][key]) for key in ("x0", "y0", "x1", "y1")] == [30, 20, 90, 70]
    assert rows[0]["side_id"] == "side_A"
    assert rows[0]["edge_contact"] == "left"
    assert cv2.imread(str(target / "catalog-01.jpg")).shape == (1400, 1920, 3)
    assert cv2.imread(str(target / "catalog-overview.jpg")).shape == (700, 960, 3)
    assert (tmp_path / "sample.png").read_bytes() == source_before
    assert manifest.read_bytes() == manifest_before


def test_reannotation_preserves_datamatrix_candidate(tmp_path, monkeypatch):
    manifest = example_manifest(tmp_path)
    data = read_yaml(manifest)
    candidate = {"box": [35, 25, 40, 30], "annotation_source": "existing_manual_annotation"}
    data["samples"][0]["datamatrix_roi_candidate"] = candidate
    save_yaml(manifest, data)
    monkeypatch.setattr(cv2, "selectROI", lambda *a, **kw: (10, 10, 50, 60))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    answers = iter(["piece_01", "unknown"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    annotate(manifest)
    assert read_yaml(manifest)["samples"][0]["datamatrix_roi_candidate"] == candidate
