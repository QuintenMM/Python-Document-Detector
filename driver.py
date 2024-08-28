import os
from page_extractor import PageExtractor
from hough_line_corner_detector import HoughLineCornerDetector


def process_images_in_directory(directory_path):
    # Ensure directory exists
    if not os.path.isdir(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return

    # Get list of images
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("No image files found in the directory.")
        return

    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        print(f"Processing {image_path}")

        try:
            # Initialize PageExtractor with image
            page_extractor = PageExtractor(image_path)

            # Process the image
            extracted = page_extractor()
            print(f"Extracted: {extracted}")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")


if __name__ == "__main__":
    directory_path = r'C:\Users\quintenwildemeersch\PycharmProjects\receipt_detection\Ticket_Text_Extraction\2_Picture_Preprocessing\Output_Ticket_Orientation'
    process_images_in_directory(directory_path)

