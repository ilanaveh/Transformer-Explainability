"""
31/5/26
"""
import cv2
import numpy as np
from PIL import Image
import os


# -----------------------------
# Panel detection
# -----------------------------

def detect_panels(image_path, min_area=50_000):
    """
    Detect rectangular face panels in a multi-panel figure.
    Returns panel boxes as (x, y, w, h), sorted left-to-right.
    """

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect dark panel borders
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        if area > min_area and h > w * 0.8:
            boxes.append((x, y, w, h))

    if len(boxes) == 0:
        raise RuntimeError("No panels detected. Try lowering min_area.")

    boxes = sorted(boxes, key=lambda b: b[0])
    return boxes


def crop_panel_face(image, box, border_px=3):
    """
    Crop inside a detected panel border, excluding the black boundary.
    """
    x, y, w, h = box

    return image[
        y + border_px : y + h - border_px,
        x + border_px : x + w - border_px
    ]


def crop_to_square_image(image, mode="center"):
    """
    Crop an image to a square before resizing.

    Parameters
    ----------
    image : np.ndarray
        Input image.

    mode : str
        How to align the crop when height != width.

        Options:
            "center" : crop equally from both sides/top-bottom
            "top"    : preserve top of image, crop lower part if needed
            "bottom" : preserve bottom of image, crop upper part if needed
            "left"   : preserve left side, crop right side if needed
            "right"  : preserve right side, crop left side if needed
    """

    h, w = image.shape[:2]
    side = min(h, w)

    if h == w:
        return image

    if h > w:
        x1 = 0
        x2 = w

        if mode == "top":
            y1 = 0
        elif mode == "bottom":
            y1 = h - side
        else:
            y1 = (h - side) // 2

        y2 = y1 + side

    else:
        y1 = 0
        y2 = h

        if mode == "left":
            x1 = 0
        elif mode == "right":
            x1 = w - side
        else:
            x1 = (w - side) // 2

        x2 = x1 + side

    return image[y1:y2, x1:x2]

# -----------------------------
# Row-based composition
# -----------------------------

def make_composite_face_by_rows(
    image_path,
    row_to_panel,
    output_path="composite_face_224.png",
    panel_size=224,
    crop_to_square=True,
    square_crop_mode="center"

):
    """
    Create a single face image by taking each horizontal row from a chosen panel.

    Parameters
    ----------
    image_path : str
        Path to the multi-panel input image.

    row_to_panel : list[int]
        Length must equal n_rows.

        Each entry says which panel to use for that output row.

        Panel indices are zero-based:
            panel 0 = leftmost panel
            panel 1 = second panel
            panel 2 = third panel
            panel 3 = fourth panel

        Example:
            row_to_panel = [0, 0, 1, 1, 2, 2, 3, 3]

        This means:
            rows 0-1 come from panel 0
            rows 2-3 come from panel 1
            rows 4-5 come from panel 2
            rows 6-7 come from panel 3

    output_path : str
        Where to save the final 224x224 output image.

    panel_size : int
        Output width and height.

    n_rows : int
        Number of horizontal rows to split the face into.
    """

    n_rows = len(row_to_panel)

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    boxes = detect_panels(image_path)

    print(f"Detected {len(boxes)} panels:")
    for i, box in enumerate(boxes):
        print(f"  Panel {i}: {box}")

    panel_faces = []

    for box in boxes:
        face = crop_panel_face(image_bgr, box)

        if crop_to_square:
            face = crop_to_square_image(face, mode=square_crop_mode)

        # Resize every panel to the same square canvas
        face = cv2.resize(face, (panel_size, panel_size), interpolation=cv2.INTER_AREA)

        panel_faces.append(face)

    max_panel_idx = len(panel_faces) - 1

    for panel_idx in row_to_panel:
        if panel_idx < 0 or panel_idx > max_panel_idx:
            raise IndexError(
                f"Panel index {panel_idx} is invalid. "
                f"Detected panels are 0 through {max_panel_idx}."
            )

    output = np.zeros_like(panel_faces[0])

    row_height = face.shape[0] // n_rows

    for row_idx, panel_idx in enumerate(row_to_panel):
        y1 = row_idx * row_height

        if row_idx == n_rows - 1:
            y2 = face.shape[0]
        else:
            y2 = (row_idx + 1) * row_height

        output[y1:y2, :] = panel_faces[panel_idx][y1:y2, :]

    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    Image.fromarray(output_rgb).save(output_path)

    print(f"Saved: {output_path}")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":

    choose_parts = {
        'f_ang': [3, 0],
        'f_dis': [0, 0, 0, 0, 3, 3, 3, 3, 3, 3],
        'f_fea': [2, 0],
        'f_hap': [0, 0, 0, 0, 1, 1, 2, 2, 2, 2],
        'f_sad': [0, 1],
        'f_sup': [0, 1],
        'm_ang': [1, 0],
        'm_dis': [0, 0, 0, 0, 1, 1, 1, 1, 1],
        'm_fea': [2, 0],
        'm_hap': [1, 2],
        'm_sad': [1, 0],
        'm_sup': [3, 0],

    }

    for im, row_list in choose_parts.items():
        print(im)

        image_path = os.path.join('from_paper', f'{im}.png')
        make_composite_face_by_rows(
            image_path=image_path,
            row_to_panel=row_list,
            panel_size=224,
            output_path=image_path.replace('.png', '_comp_crop_224.png'),
        )

        make_composite_face_by_rows(
            image_path=image_path,
            row_to_panel=row_list,
            panel_size=224,
            crop_to_square=False,
            output_path=image_path.replace('.png', '_comp_224.png'),
        )


