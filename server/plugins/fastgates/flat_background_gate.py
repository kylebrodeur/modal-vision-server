"""Flat-background deterministic fast gate.

Skips the segmentation pass when the frame is essentially a single flat
surface: nearly every pixel sits within a tight per-channel distance of the
average corner color, and the center of the frame carries almost no gradient
energy (scanner beds, flat-lay backdrops, blank backgrounds).

Pure and deterministic: no randomness, no network, no torch. The image is
only queried through duck-typed PIL-style methods (convert/resize/getdata).
"""

# Per-channel tolerance: 12 of 255 levels.
_CHANNEL_TOLERANCE = 12.0
# Fraction of pixels that must hug the corner color to qualify as "flat".
_FLAT_FRACTION = 0.92
# Strong-gradient transitions allowed inside the center third.
_GRADIENT_BUDGET = 12

_SIZE = 64
_PATCH = 8


def _luma(pixel) -> float:
    r, g, b = pixel[0], pixel[1], pixel[2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def decide(image, meta):
    small = image.convert("RGB").resize((_SIZE, _SIZE))
    pixels = list(small.getdata())

    # Average the four 8x8 corner patches into one reference background color.
    corners = []
    for origin_y in (0, _SIZE - _PATCH):
        for origin_x in (0, _SIZE - _PATCH):
            for y in range(origin_y, origin_y + _PATCH):
                for x in range(origin_x, origin_x + _PATCH):
                    corners.append(pixels[y * _SIZE + x])
    n = float(len(corners))
    ref = tuple(sum(px[c] for px in corners) / n for c in range(3))

    close = 0
    for px in pixels:
        if all(abs(px[c] - ref[c]) <= _CHANNEL_TOLERANCE for c in range(3)):
            close += 1
    if close / float(len(pixels)) < _FLAT_FRACTION:
        return None  # plenty of structure here; let the real pipeline decide

    # Count strong horizontal+vertical luma steps inside the center third.
    low, high = _SIZE // 3, 2 * _SIZE // 3
    gradients = 0
    for y in range(low, high):
        for x in range(low, high):
            px = _luma(pixels[y * _SIZE + x])
            if x + 1 < high and abs(_luma(pixels[y * _SIZE + x + 1]) - px) > _CHANNEL_TOLERANCE:
                gradients += 1
            if y + 1 < high and abs(_luma(pixels[(y + 1) * _SIZE + x]) - px) > _CHANNEL_TOLERANCE:
                gradients += 1

    if gradients <= _GRADIENT_BUDGET:
        return {"decision": "skip", "reason": "flat or isolated: cheap pre-pass"}
    return None
