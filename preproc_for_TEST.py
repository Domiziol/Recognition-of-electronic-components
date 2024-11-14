import cv2
import os
import numpy as np
from pathlib import Path

# klasy = ['R', 'C', 'D', 'T', 'S', 'L', 'B']


#for kl in klasy:
input_dir = Path(os.path.dirname(os.path.abspath(__file__)))/"TEST"
output_dir = Path(os.path.dirname(os.path.abspath(__file__)))/"TEST_output"
if not output_dir.exists():
    output_dir.mkdir(parents=True)

# Read the image using OpenCV
image_files = os.listdir(input_dir)

for image_file in image_files:
    image_path = os.path.join(input_dir, image_file)
    if not image_path.endswith('.txt'):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (256, 256))
        assert image is not None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       
        gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,69,9)
       
        for row in range(gray.shape[0]):
            avg = np.average(gray[row, :] > 1)
            if avg > 0.5:
                cv2.line(gray, (0, row), (gray.shape[1] - 1, row), (0, 0, 0), 1)

        for col in range(gray.shape[1]):
            avg = np.average(gray[:, col] > 1)
            if avg > 0.4:
                cv2.line(gray, (col, 0), (col, gray.shape[0] - 1), (0, 0, 0), 1)

        gray = cv2.fastNlMeansDenoising(gray, None,  27, 11, 23)
       
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, gray)
       
