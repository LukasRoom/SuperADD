import numpy as np
import cv2

def create_line_kernel(radius: int, angle_deg: float, thickness: int=1) -> np.ndarray:
    """
    Create a binary line structuring element of given radius and angle.
    Angle in degrees, 0° = horizontal line.
    """
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=np.uint8)

    center = size // 2
    rad = np.deg2rad(angle_deg)

    # Half-radius in each direction
    dx = radius * np.cos(rad)
    dy = radius * np.sin(rad)

    x0 = int(round(center - dx))
    y0 = int(round(center - dy))
    x1 = int(round(center + dx))
    y1 = int(round(center + dy))

    cv2.line(kernel, (x0, y0), (x1, y1), color=1, thickness=thickness)

    return kernel

def multi_oriented_closing(img: np.ndarray, threshold: float, radius: int, n_angles: int = 8,
                           lower_factor: float = None, padding: bool = True) -> np.ndarray:
    """
    Apply morphological closing with a line SE at multiple orientations.
    The closings are applied sequentially.
    """
    img_closed_versions = []
    img_thres = np.where(img > threshold, 255, 0).astype(np.uint8)

    pad_width = radius + 1

    if padding:
        img_thres = np.pad(img_thres, pad_width=pad_width, mode='constant', constant_values=0)

    kernel_line_thick = np.ones((2, 2), dtype=np.uint8)

    # Angles from 0 to <180 degrees (180 is redundant with 0 for a line)
    for i in range(n_angles):
        img_gray_ = img_thres.copy()
        angle = 180.0 * i / n_angles
        kernel1 = create_line_kernel(radius, angle, thickness=2)
        kernel2 = cv2.morphologyEx(kernel1, cv2.MORPH_DILATE, kernel_line_thick)
        eroded = cv2.morphologyEx(img_gray_, cv2.MORPH_DILATE, kernel2)
        closed = cv2.morphologyEx(eroded, cv2.MORPH_ERODE, kernel1)
        img_closed_versions.append(closed)

    # Add up the versions using or (logical max)
    final_closed = img_thres.copy()  # Start with the original thresholded image to ensure we only keep areas that were above the threshold to begin with

    for closed in img_closed_versions:
        final_closed = np.maximum(final_closed, closed)

    if padding:
        final_closed = final_closed[pad_width:-pad_width, pad_width:-pad_width]  # Remove padding after closing

    # And operation with the lower thresholded image to only keep the closed areas that are above the lower threshold
    if lower_factor is not None:
        img_lower_thres = np.where(img > threshold * lower_factor, 255, 0).astype(np.uint8)
        final_closed = np.where(img_lower_thres > 0, final_closed,0)

    return final_closed

def fill_closed_regions(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(image)
    cv2.drawContours(filled, contours, contourIdx=-1, color=255, thickness=cv2.FILLED)
    return filled


def erosion_on_binary_maps(x, erosion):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion + 1, 2 * erosion + 1))
    eroded_images = cv2.morphologyEx(x, cv2.MORPH_ERODE, kernel)
    return eroded_images
