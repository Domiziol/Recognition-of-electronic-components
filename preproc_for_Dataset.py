import cv2
import os
import numpy as np
from pathlib import Path

klasy = ['R', 'C', 'D', 'T', 'S', 'L', 'B']


for kl in klasy:
    input_dir = Path(os.path.dirname(os.path.abspath(__file__)))/"Dataset"/f'{kl}'
    output_dir = Path(os.path.dirname(os.path.abspath(__file__)))/"Dataset_output"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

# Read the image using OpenCV
    image_files = os.listdir(input_dir)

    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        if not image_path.endswith('.txt') and image_path.endswith('CCJ.jpg'):
            image = cv2.imread(image_path)
            image = cv2.resize(image, (1024, 1024))
            assert image is not None 

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.fastNlMeansDenoising(gray, None,  3, 7, 21)
            gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
         
            output_path = os.path.join(output_dir, image_file)
            cv2.imwrite(output_path, gray)
