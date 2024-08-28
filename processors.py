import cv2
import math
from scipy import ndimage
import numpy as np

class RotationCorrector:
    def __init__(self, output_process=False):
        self.output_process = output_process
        print(f"Initialized RotationCorrector with output_process={output_process}")

    def __call__(self, image):
        print("Starting rotation correction")

        img_before = image.copy()
        print(f"Original image shape: {img_before.shape}")

        # Edge detection
        img_edges = cv2.Canny(img_before, 50, 150, apertureSize=3)
        print(f"Edges detected. Shape: {img_edges.shape}")

        # Line detection
        lines = cv2.HoughLinesP(
            img_edges,
            1,
            math.pi / 90.0,
            100,
            minLineLength=100,
            maxLineGap=5
        )
        if lines is not None:
            print(f"Number of lines found: {len(lines)}")
        else:
            print("No lines found.")
            lines = []

        def get_angle(line):
            x1, y1, x2, y2 = line[0]
            return math.degrees(math.atan2(y2 - y1, x2 - x1))

        if lines:
            angles = np.array([get_angle(line) for line in lines])
            median_angle = np.median(angles)
            print(f"Angles: {angles}")
            print(f"Median angle: {median_angle}")
        else:
            median_angle = 0
            print("No angles to calculate median.")

        # Rotate image
        img_rotated = ndimage.rotate(
            img_before,
            median_angle,
            cval=255,
            reshape=False
        )
        print(f"Rotated image shape: {img_rotated.shape}")

        if self.output_process:
            cv2.imwrite('output/10. tab_extract rotated.jpg', img_rotated)
            print("Rotated image saved to 'output/10. tab_extract rotated.jpg'")

        return img_rotated


class Resizer:
    def __init__(self, height=1280, output_process=False):
        self._height = height
        self.output_process = output_process
        print(f"Initialized Resizer with height={height} and output_process={output_process}")

    def __call__(self, image):
        print("Starting resizing")

        if image.shape[0] <= self._height:
            print("Image height is already less than or equal to the specified height. No resizing performed.")
            return image

        ratio = round(self._height / image.shape[0], 3)
        width = int(image.shape[1] * ratio)
        dim = (width, self._height)
        print(f"Resizing image to dimensions: {dim}")

        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        print(f"Resized image shape: {resized.shape}")

        if self.output_process:
            cv2.imwrite('output/resized.jpg', resized)
            print("Resized image saved to 'output/resized.jpg'")

        return resized


class OtsuThresholder:
    def __init__(self, thresh1=0, thresh2=255, output_process=False):
        self.output_process = output_process
        self.thresh1 = thresh1
        self.thresh2 = thresh2
        print(f"Initialized OtsuThresholder with thresh1={thresh1}, thresh2={thresh2}, output_process={output_process}")

    def __call__(self, image):
        print("Starting Otsu thresholding")

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(f"Converted image to grayscale. Shape: {image_gray.shape}")

        T_, thresholded = cv2.threshold(image_gray, self.thresh1, self.thresh2, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"Otsu threshold value: {T_}")

        if self.output_process:
            cv2.imwrite('output/thresholded.jpg', thresholded)
            print("Thresholded image saved to 'output/thresholded.jpg'")

        return thresholded


class FastDenoiser:
    def __init__(self, strength=7, output_process=False):
        self._strength = strength
        self.output_process = output_process
        print(f"Initialized FastDenoiser with strength={strength} and output_process={output_process}")

    def __call__(self, image):
        print("Starting fastNlMeansDenoising")

        temp = cv2.fastNlMeansDenoising(image, h=self._strength)
        print(f"Denoised image shape: {temp.shape}")

        if self.output_process:
            cv2.imwrite('output/denoised.jpg', temp)
            print("Denoised image saved to 'output/denoised.jpg'")

        return temp


class Closer:
    def __init__(self, kernel_size=3, iterations=10, output_process=False):
        self._kernel_size = kernel_size
        self._iterations = iterations
        self.output_process = output_process
        print(f"Initialized Closer with kernel_size={kernel_size}, iterations={iterations}, output_process={output_process}")

    def __call__(self, image):
        print("Starting morphological closing")

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self._kernel_size, self._kernel_size)
        )
        closed = cv2.morphologyEx(
            image,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=self._iterations
        )
        print(f"Closed image shape: {closed.shape}")

        if self.output_process:
            cv2.imwrite('output/closed.jpg', closed)
            print("Closed image saved to 'output/closed.jpg'")

        return closed


class Opener:
    def __init__(self, kernel_size=3, iterations=25, output_process=False):
        self._kernel_size = kernel_size
        self._iterations = iterations
        self.output_process = output_process
        print(f"Initialized Opener with kernel_size={kernel_size}, iterations={iterations}, output_process={output_process}")

    def __call__(self, image):
        print("Starting morphological opening")

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self._kernel_size, self._kernel_size)
        )
        opened = cv2.morphologyEx(
            image,
            cv2.MORPH_OPEN,
            kernel,
            iterations=self._iterations
        )
        print(f"Opened image shape: {opened.shape}")

        if self.output_process:
            cv2.imwrite('output/opened.jpg', opened)
            print("Opened image saved to 'output/opened.jpg'")

        return opened


class EdgeDetector:
    def __init__(self, output_process=False):
        self.output_process = output_process
        print(f"Initialized EdgeDetector with output_process={output_process}")

    def __call__(self, image, thresh1=50, thresh2=150, apertureSize=3):
        print("Starting edge detection")

        edges = cv2.Canny(image, thresh1, thresh2, apertureSize=apertureSize)
        print(f"Edges detected. Shape: {edges.shape}")

        if self.output_process:
            cv2.imwrite('output/edges.jpg', edges)
            print("Edges image saved to 'output/edges.jpg'")

        return edges
