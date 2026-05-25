import cv2
import imagehash
from PIL import Image

BLUR_THRESHOLD = 100.0
MIN_BRIGHTNESS = 20
MAX_BRIGHTNESS = 235
MIN_CONTRAST = 15


def check_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray,cv2.CV_64F).var()

    if variance < BLUR_THRESHOLD:
        raise ValueError(
            f"Blurry image detected "
            f"(variance = {variance:.2f})"
        )


def check_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    brightness = gray.mean()

    if brightness < MIN_BRIGHTNESS:
        raise ValueError(
            f"Image too dark "
            f"(brightness = {brightness:.2f})"
        )

    if brightness > MAX_BRIGHTNESS:
        raise ValueError(
            f"Image too bright "
            f"(brightness = {brightness:.2f})"
        )


def check_contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    contrast = gray.std()

    if contrast < MIN_CONTRAST:
        raise ValueError(
            f"Low contrast image "
            f"(std = {contrast:.2f})"
        )


def compute_image_hash(img):
    pil_img = Image.fromarray(img)
    return str(imagehash.phash(pil_img))


def check_duplicate(img_hash, seen_hashes):
    if img_hash in seen_hashes:
        raise ValueError("Duplicate image detected.")

    seen_hashes.add(img_hash)


def run_quality_checks(
    img,
    seen_hashes = None
):

    check_blur(img)
    check_brightness(img)
    check_contrast(img)

    if seen_hashes is not None:
        img_hash = compute_image_hash(img)
        check_duplicate(img_hash, seen_hashes)
