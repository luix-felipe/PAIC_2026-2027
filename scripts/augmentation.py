"""Geometria explícita para estudar augmentation do detector, sem decodificação."""

from __future__ import annotations

import cv2
import numpy as np


def validate_box(box, size):
    """XYXY em coordenadas de bordas: 0 <= x0 < x1 <= largura (idem para y)."""
    values = np.asarray(box, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Caixa deve conter quatro coordenadas finitas")
    x0, y0, x1, y1 = values
    width, height = size
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Caixa vazia ou fora da imagem; não será truncada")
    return values


def transform_points(points, matrix):
    points = np.asarray(points, dtype=np.float64)
    return points @ matrix[:, :2].T + matrix[:, 2]


def transform_box(box, matrix, output_size):
    x0, y0, x1, y1 = box
    corners = transform_points([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], matrix)
    result = [*corners.min(axis=0), *corners.max(axis=0)]
    return validate_box(result, output_size)


def letterbox(frame, box, size):
    """Resize somente para o detector; retorna matriz original -> entrada."""
    if type(size) is not int or size < 1:
        raise ValueError("Tamanho da entrada deve ser inteiro positivo")
    height, width = frame.shape[:2]
    validate_box(box, (width, height))
    ratio = min(size / width, size / height)
    new_w, new_h = max(1, round(width * ratio)), max(1, round(height * ratio))
    left, top = (size - new_w) // 2, (size - new_h) // 2
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = cv2.copyMakeBorder(
        resized, top, size - new_h - top, left, size - new_w - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    # O arredondamento pode produzir escalas x/y ligeiramente diferentes.
    matrix = np.array([[new_w / width, 0, left], [0, new_h / height, top]])
    return canvas, transform_box(box, matrix, (size, size)), matrix


def augment(frame, box, *, angle=0.0, scale=1.0, tx=0.0, ty=0.0,
            contrast=1.0, brightness=0.0):
    """tx/ty são frações da imagem; brightness é um deslocamento em [0,255].

    A caixa e o centro usam a mesma transformação da imagem. Uma transformação
    que corta a caixa inteira do painel é rejeitada, em vez de ajustar o rótulo.
    """
    values = [angle, scale, tx, ty, contrast, brightness]
    if not np.isfinite(values).all() or scale <= 0 or contrast <= 0:
        raise ValueError("Parâmetros de augmentation inválidos")
    height, width = frame.shape[:2]
    box = validate_box(box, (width, height))
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[:, 2] += [tx * width, ty * height]
    transformed = transform_box(box, matrix, (width, height))

    # A anotação usa bordas de pixel; warpAffine usa centros de pixel.
    # p_borda = p_centro + 0.5. Esta conversão evita um deslocamento de meio pixel.
    pixel_matrix = matrix.copy()
    pixel_matrix[:, 2] += matrix[:, :2] @ np.array([0.5, 0.5]) - 0.5
    warped = cv2.warpAffine(
        frame, pixel_matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114),
    )
    # Não usar convertScaleAbs: ele toma o valor absoluto de intensidades negativas.
    result = np.clip(warped.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)
    center = (box[:2] + box[2:]) / 2
    transformed_center = transform_points([center], matrix)[0]
    return result, transformed, transformed_center, matrix

