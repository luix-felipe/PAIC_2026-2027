import json

import cv2
import numpy as np
import pytest
import yaml
import zxingcpp

from scripts.experiment_blanks import (
    Symbol,
    ZXingDecoder,
    extract_roi,
    inspect_sample,
    preprocess_roi,
    run,
)


def test_roi_preserves_original_pixels_and_resolution():
    frame = np.arange(200, dtype=np.uint8).reshape(10, 20)
    roi = extract_roi(frame, [3, 2, 9, 7])
    np.testing.assert_array_equal(roi, frame[2:7, 3:9])
    assert roi.shape == (5, 6)
    assert np.shares_memory(roi, frame)


@pytest.mark.parametrize("bounds", [
    [-1, 0, 4, 4], [0, 0, 21, 4], [0, 0, 4, 11],
    [4, 0, 4, 4], [4, 4, 2, 2], [0, 0, 4.5, 4], [0, 0, 4],
])
def test_roi_rejects_invalid_bounds(bounds):
    with pytest.raises(ValueError):
        extract_roi(np.zeros((10, 20), dtype=np.uint8), bounds)


def synthetic_code(format, text):
    # Dados artificiais exclusivos dos testes; nenhuma foto de produção.
    return np.asarray(zxingcpp.create_barcode(text, format).to_image(scale=6))


def test_decoder_reads_real_ecc200_symbol():
    roi = synthetic_code(zxingcpp.BarcodeFormat.DataMatrix, "TEST-ONLY-2027")
    assert ZXingDecoder().decode(roi) == (Symbol("TEST-ONLY-2027", b"TEST-ONLY-2027"),)


def test_decoder_returns_no_read_on_blank_image():
    assert ZXingDecoder().decode(np.full((200, 200), 255, dtype=np.uint8)) == ()


def test_decoder_restricts_reading_to_datamatrix():
    qr = synthetic_code(zxingcpp.BarcodeFormat.QRCode, "TEST-ONLY-QR")
    assert ZXingDecoder().decode(qr) == ()


@pytest.mark.parametrize("variant", ["raw", "clahe", "otsu"])
def test_preprocessing_preserves_pixel_dimensions_and_source(variant):
    roi = np.arange(144, dtype=np.uint8).reshape(12, 12)
    original = roi.copy()
    assert preprocess_roi(roi, variant).shape == roi.shape
    np.testing.assert_array_equal(roi, original)


def test_report_omits_payload_and_distinguishes_detection_from_reading():
    roi = synthetic_code(zxingcpp.BarcodeFormat.DataMatrix, "TEST-PRIVATE-PAYLOAD")
    height, width = roi.shape
    report = inspect_sample(roi, [[0, 0, width, height]], ["raw"], ZXingDecoder())
    assert report["status"] == "READ"
    assert report["motherboard_detection"] == "NOT_EVALUATED"
    assert "TEST-PRIVATE-PAYLOAD" not in json.dumps(report)


def test_invalid_roi_does_not_call_decoder():
    class UnexpectedDecoder:
        def decode(self, roi):
            raise AssertionError("ROI inválida chegou ao decoder")

    report = inspect_sample(np.zeros((10, 10), dtype=np.uint8), [[0, 0, 11, 10]], ["raw"], UnexpectedDecoder())
    assert report["status"] == "NO_READ"
    assert report["attempts"][0]["status"] == "INVALID_ROI"


def test_multiple_readable_rois_produce_one_image_result():
    roi = synthetic_code(zxingcpp.BarcodeFormat.DataMatrix, "TEST-SAME-CODE")
    height, width = roi.shape
    frame = np.concatenate((roi, roi), axis=1)
    report = inspect_sample(frame, [[0, 0, width, height], [width, 0, 2*width, height]], ["raw"], ZXingDecoder())
    assert report["status"] == "READ"
    assert sum(a["status"] == "READ" for a in report["attempts"]) == 2
    # Uma imagem é uma amostra. Isto não implementa deduplicação temporal de passagem.


def test_loading_failure_is_explicit(tmp_path):
    config = tmp_path / "test.yaml"
    config.write_text("dataset: .\nsamples:\n  - image: missing.jpg\n    frame_size: [10, 10]\n    rois: [[0, 0, 10, 10]]\n", encoding="utf-8")
    report = run(config, ["raw"])
    assert report["samples"][0]["status"] == "CAPTURE_FAILED"
    assert report["summary"]["images_with_any_read"] == 0


def test_resolution_mismatch_prevents_decoding(tmp_path):
    cv2.imwrite(str(tmp_path / "frame.png"), np.full((20, 30), 255, dtype=np.uint8))
    config = tmp_path / "test.yaml"
    config.write_text(yaml.safe_dump({
        "dataset": ".",
        "samples": [{"image": "frame.png", "frame_size": [300, 200], "rois": [[0, 0, 10, 10]]}],
    }), encoding="utf-8")
    report = run(config, ["raw"])
    assert report["samples"][0]["status"] == "FRAME_SIZE_MISMATCH"
    assert report["samples"][0]["attempts"] == []


def test_file_to_report_integration_with_synthetic_code(tmp_path):
    frame = synthetic_code(zxingcpp.BarcodeFormat.DataMatrix, "TEST-INTEGRATION")
    height, width = frame.shape
    cv2.imwrite(str(tmp_path / "frame.png"), frame)
    config = tmp_path / "test.yaml"
    config.write_text(yaml.safe_dump({
        "dataset": ".",
        "samples": [{"image": "frame.png", "frame_size": [width, height], "rois": [[0, 0, width, height]]}],
    }), encoding="utf-8")
    report = run(config, ["raw"])
    assert report["summary"]["images_with_any_read"] == 1
    assert report["motherboard_detection"] == "NOT_EVALUATED"
    assert "TEST-INTEGRATION" not in json.dumps(report)

