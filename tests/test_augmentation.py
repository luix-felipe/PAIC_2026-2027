import cv2
import numpy as np
import pytest

from scripts.augmentation import augment, letterbox, transform_points, validate_box


def rectangle_image():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[20:40, 30:50] = 255
    return image, [30, 20, 50, 40]


def white_bounds(image):
    ys, xs = np.where(np.all(image == 255, axis=2))
    return [xs.min(), ys.min(), xs.max()+1, ys.max()+1]


def test_translation_moves_pixels_box_and_center_together():
    source, box = rectangle_image()
    result, new_box, center, _ = augment(source, box, tx=0.1, ty=-0.1)
    np.testing.assert_allclose(new_box, [40, 10, 60, 30])
    np.testing.assert_allclose(center, [50, 20])
    np.testing.assert_array_equal(white_bounds(result), new_box)


def test_rotation_respects_pixel_edges_without_half_pixel_drift():
    source, box = rectangle_image()
    result, new_box, center, _ = augment(source, box, angle=90)
    np.testing.assert_allclose(new_box, [20, 50, 40, 70])
    np.testing.assert_allclose(center, [30, 60])
    np.testing.assert_allclose(white_bounds(result), new_box)


def test_diagonal_rotation_uses_all_four_corners():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    _, box, center, _ = augment(frame, [40, 40, 60, 60], angle=45)
    radius = 10 * np.sqrt(2)
    np.testing.assert_allclose(box, [50-radius, 50-radius, 50+radius, 50+radius])
    np.testing.assert_allclose(center, [50, 50])


def test_scale_moves_box_about_image_center():
    source, box = rectangle_image()
    _, new_box, center, _ = augment(source, box, scale=0.5)
    np.testing.assert_allclose(new_box, [40, 35, 50, 45])
    np.testing.assert_allclose(center, [45, 40])


def test_panel_cropping_is_rejected():
    source, box = rectangle_image()
    with pytest.raises(ValueError, match="fora"):
        augment(source, box, tx=1)


def test_photometric_changes_do_not_mutate_source_or_geometry():
    source, box = rectangle_image()
    original = source.copy()
    _, new_box, center, _ = augment(source, box, contrast=0.9, brightness=15)
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(new_box, box)
    np.testing.assert_array_equal(center, [40, 30])


def test_negative_brightness_clips_instead_of_taking_absolute_value():
    source = np.full((20, 20, 3), 5, dtype=np.uint8)
    result, *_ = augment(source, [2, 2, 10, 10], brightness=-10)
    assert result.max() == 0


def test_letterbox_records_padding_and_actual_rounded_scales():
    source = np.zeros((3, 7, 3), dtype=np.uint8)
    output, box, matrix = letterbox(source, [0, 0, 7, 3], 10)
    assert output.shape == (10, 10, 3)
    np.testing.assert_allclose(matrix, [[10/7, 0, 0], [0, 4/3, 3]])
    np.testing.assert_allclose(box, [0, 3, 10, 7])
    inverse = cv2.invertAffineTransform(matrix)
    points = [[0, 0], [7, 3], [2.5, 1.7]]
    np.testing.assert_allclose(transform_points(transform_points(points, matrix), inverse), points)


@pytest.mark.parametrize("box", [[-1, 0, 5, 5], [0, 0, 101, 5], [2, 2, 2, 5], [0, 0, float('nan'), 5]])
def test_invalid_annotations_are_rejected(box):
    with pytest.raises(ValueError):
        validate_box(box, (100, 100))


@pytest.mark.parametrize("params", [{"scale": 0}, {"contrast": -1}, {"angle": float('inf')}])
def test_invalid_augmentation_parameters_are_rejected(params):
    source, box = rectangle_image()
    with pytest.raises(ValueError):
        augment(source, box, **params)

