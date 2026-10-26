"""Create a best-effort stitched panorama and a guaranteed eight-view overview."""

from __future__ import annotations


def make_overview(cv2, frames, out_path):
    height, width, band = 300, 430, 38
    canvas = cv2.hconcat([
        _tile(cv2, frame, width, height, band, i + 1) for i, frame in enumerate(frames)
    ])
    cv2.imwrite(str(out_path), canvas)
    return out_path


def make_grid(cv2, frames, out_path):
    tiles = [_tile(cv2, frame, 430, 300, 38, i + 1) for i, frame in enumerate(frames)]
    top = cv2.hconcat(tiles[:4])
    bottom = cv2.hconcat(tiles[4:])
    canvas = cv2.vconcat([top, bottom])
    cv2.imwrite(str(out_path), canvas)
    return out_path


def _tile(cv2, frame, width, height, band, number):
    import numpy as np

    tile = np.full((height + band, width, 3), (27, 37, 51), dtype=np.uint8)
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    img = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
    y, x = (height - img.shape[0]) // 2, (width - img.shape[1]) // 2
    tile[y:y + img.shape[0], x:x + img.shape[1]] = img
    cv2.putText(tile, f"VIEW {number}", (15, height + 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (235, 240, 250), 2)
    return tile


def make_panorama(cv2, frames, out_path):
    """Return (success, reason); requires shared visual features between views."""
    try:
        prepared = []
        for frame in frames:
            h, w = frame.shape[:2]
            scale = min(1.0, 900.0 / max(h, w))
            prepared.append(cv2.resize(frame, (int(w * scale), int(h * scale))))
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status, pano = stitcher.stitch(prepared)
        if status != cv2.Stitcher_OK or pano is None:
            return False, f"OpenCV stitch status {status}; adjacent views may lack overlap"
        cv2.imwrite(str(out_path), pano)
        return True, "Panorama stitched"
    except Exception as exc:
        return False, f"Stitching failed: {exc}"
