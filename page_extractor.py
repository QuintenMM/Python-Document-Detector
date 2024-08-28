import cv2
import numpy as np

from skimage.filters import threshold_otsu
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from itertools import combinations
from collections import defaultdict

class PageExtractor:
    def __init__(self, preprocessors, corner_detector, output_process=False):
        assert isinstance(preprocessors, list), "List of processors expected"
        self._preprocessors = preprocessors
        self._corner_detector = corner_detector
        self.output_process = output_process
        print(f"Initialized PageExtractor with preprocessors: {preprocessors} and corner_detector: {corner_detector}")

    def __call__(self, image_path):
        print(f"Processing image: {image_path}")

        # Step 1: Read image from file
        self._image = cv2.imread(image_path)
        if self._image is None:
            print(f"Error: Image at {image_path} could not be read.")
            return None
        print(f"Image shape: {self._image.shape}")

        # Step 2: Preprocess image
        self._processed = self._image
        for preprocessor in self._preprocessors:
            self._processed = preprocessor(self._processed)
            print(f"Preprocessed image shape: {self._processed.shape}")

        # Detect corners
        self._intersections = self._corner_detector(self._processed)
        print(f"Intersections detected: {self._intersections}")

        # Step 3: Deskew and extract page
        return self._extract_page()

    def _extract_page(self):
        print("Starting _extract_page")

        # Obtain and print points
        pts = np.array([
            (x, y)
            for intersection in self._intersections
            for x, y in intersection
        ])
        print(f"Points for ordering: {pts}")

        rect = self._order_points(pts)
        print(f"Ordered points: {rect}")

        (tl, tr, br, bl) = rect

        # Compute width and height
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        print(f"Max width: {maxWidth}, Max height: {maxHeight}")

        # Construct destination points and perform perspective transform
        dst = np.array([
            [0, 0],                         # Top left point
            [maxWidth - 1, 0],              # Top right point
            [maxWidth - 1, maxHeight - 1],  # Bottom right point
            [0, maxHeight - 1]],            # Bottom left point
            dtype="float32"
        )
        print(f"Destination points for perspective transform: {dst}")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(self._processed, M, (maxWidth, maxHeight))

        if self.output_process:
            cv2.imwrite('output/deskewed.jpg', warped)
            print("Warped image saved to 'output/deskewed.jpg'")

        return warped

    def _order_points(self, pts):
        """
        Function for getting the bounding box points in the correct
        order.

        Params
        pts     The points in the bounding box. Usually (x, y) coordinates

        Returns
        rect    The ordered set of points
        """
        print("Ordering points")

        # Initialize a list of coordinates that will be ordered such that:
        # 1st point -> Top left
        # 2nd point -> Top right
        # 3rd point -> Bottom right
        # 4th point -> Bottom left
        rect = np.zeros((4, 2), dtype="float32")

        # The top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Compute the difference between the points
        # The top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        print(f"Ordered points: {rect}")
        return rect

if __name__ == "__main__":
    import argparse
    from hough_line_corner_detector import HoughLineCornerDetector
    from processors import Resizer, OtsuThresholder, FastDenoiser

    parser = argparse.ArgumentParser(description="Python script to detect and extract documents.")

    parser.add_argument(
        '-i',
        '--input-image',
        help="Image containing the document",
        required=True,
        dest='input_image'
    )

    page_extractor = PageExtractor(
        preprocessors=[
            Resizer(height=1280, output_process=True),
            FastDenoiser(strength=9, output_process=True),
            OtsuThresholder(output_process=True)
        ],
        corner_detector=HoughLineCornerDetector(
            rho_acc=1,
            theta_acc=180,
            thresh=100,
            output_process=True
        )
    )
    args = parser.parse_args()
    extracted = page_extractor(args.input_image)
    if extracted is not None:
        cv2.imshow("Extracted page", extracted)
        cv2.waitKey(0)
    else:
        print("Extraction failed.")
