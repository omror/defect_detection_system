import cv2
import numpy as np

def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Apply GaussianBlur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Make edges more visible
    edges = cv2.Canny(blurred, 50, 150)

    # Resize the image to a fixed size
    resized = cv2.resize(image, (224, 224))

    return resized, edges

#Testing
if __name__ == '__main__':
    r, e = process_image("data/train/patches/image_1.jpg")
    cv2.imshow("Corners", e)
    cv2.waitKey(0)

