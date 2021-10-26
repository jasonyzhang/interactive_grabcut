"""
Interactive script to run OpenCV's grabcut algorithm.

Usage:
    python interactive_grabcut.py <image_path> [--mask_path <mask_path>]

Example:
    python interactive_grabcut.py images/car.jpg --mask_path masks/car.png

"""
import argparse
import os

import cv2
import numpy as np


class DataBbox(object):
    def __init__(self, image=None):
        self.image = image
        self.image_copy = image.copy()
        self.ref_point = []
        self.box_exists = False
        self.cropping = True


class DataSegmentation(object):
    def __init__(self, image=None, mask=None):
        self.image = image
        self.image_copy = image.copy()
        self.ref_point = []
        self.current_mask = mask
        self.previous_mask = None
        self.foreground = True
        self.tinted = None


def draw_bbox(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param.ref_point = [(x, y)]
        param.cropping = True
    elif event == cv2.EVENT_LBUTTONUP:
        param.ref_point.append((x, y))
        param.cropping = False
        param.image_copy = param.image.copy()
        cv2.rectangle(
            param.image_copy, param.ref_point[0], param.ref_point[1], (0, 255, 0), 2
        )
        cv2.imshow("image", param.image_copy)


def mark_mask(event, x, y, flags, param):
    value = 1 if param.foreground else 0

    if event == cv2.EVENT_LBUTTONDOWN:
        param.ref_point = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        param.ref_point.append((x, y))
        param.previous_mask = param.current_mask.copy()
        x1, y1 = param.ref_point[0]
        x2, y2 = param.ref_point[1]
        param.current_mask[y1:y2, x1:x2] = value
        param.tinted = tint_current(param.image, param.current_mask).copy()
        cv2.imshow("image", param.tinted)


def tint_current(image, mask):
    image = image.copy() / 255.0
    tinted = tint(image, mask == 0, color=(0, 0, 1))
    tinted = tint(tinted, mask == 2, color=(0, 0.5, 1.0), alpha=0.6)
    tinted = tint(tinted, mask == 1, color=(1, 0, 0))
    tinted = tint(tinted, mask == 3, color=(1, 0.5, 0), alpha=0.6)
    return (tinted * 255).astype(np.uint8)


def tint(image, mask, color=(0, 0, 0), alpha=0.4):
    color = np.array(color)
    image[mask] = color - (color - image[mask]) * alpha
    return image.clip(0, 1)


def get_bounding_box(image):
    """
    Get Bounding Box to initialize GrabCut.

    Controls:
        q: quit/done.
        r: reset.
        left-click-down: start bounding box.
        left-click-up: end bounding box.
    """
    param = DataBbox(image)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_bbox, param)
    while True:
        cv2.imshow("image", param.image_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            param.image_copy = param.image.copy()
        elif key == ord("q"):
            break
    cv2.destroyAllWindows()
    rect = np.array(param.ref_point).flatten()
    if len(rect) != 4:
        raise Exception("No Bounding Box selected")
    rect[2:] -= rect[:2]
    return rect


def segment_image(image, bounding_box):
    """
    Segments image using GrabCut.

    Controls:
        q: quit/done.
        r: reset.
        f: Switch to marking foreground.
        b: Switch to marking background.
        u: undo.
        g: Re-run GrabCut segmentation.
        left-click-down: start marking region.
        left-click-up: end marking region.
    """
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    mask = np.zeros(image.shape[:2], np.uint8)
    print("Running GrabCut.")
    cv2.grabCut(
        image, mask, tuple(bounding_box), bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT
    )
    param = DataSegmentation(image, mask)
    param.previous_mask = mask.copy()
    param.tinted = tint_current(param.image, param.current_mask).copy()

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mark_mask, param)

    while True:
        cv2.imshow("image", param.tinted)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            param.tinted = tint_current(param.image, param.current_mask).copy()
        elif key == ord("q"):
            break
        elif key == ord("g"):
            print("Re-running grabcut.")
            cv2.grabCut(
                image,
                param.current_mask,
                tuple(bounding_box),
                bgd_model,
                fgd_model,
                5,
                cv2.GC_INIT_WITH_MASK,
            )
            param.tinted = tint_current(param.image, param.current_mask).copy()
            print("done")
        elif key == ord("f"):
            param.foreground = True
            print("Marking foreground.")
        elif key == ord("b"):
            param.foreground = False
            print("Marking background.")
        elif key == ord("u"):
            print("Undo")
            param.current_mask = param.previous_mask.copy()
            param.tinted = tint_current(param.image, param.current_mask).copy()
    cv2.destroyAllWindows()
    final_mask = np.logical_or(param.current_mask == 1, param.current_mask == 3)
    final_mask = final_mask.astype(np.uint8) * 255
    cv2.namedWindow("Mask")
    cv2.imshow("Mask", final_mask)
    cv2.waitKey(3000) & 0xFF
    cv2.destroyAllWindows()
    return final_mask


def main(args):
    image_path = args.image_path
    mask_path = args.mask_path
    if mask_path is None:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join("masks", image_name + ".png")
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)

    print("Loading image:", image_path)
    print("Saving mask to:", mask_path)

    image = cv2.imread(image_path)
    bbox = get_bounding_box(image)
    mask = segment_image(image, bbox)
    cv2.imwrite(mask_path, mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument("--mask_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
