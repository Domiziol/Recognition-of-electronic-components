import cv2
import os
import numpy as np
from pathlib import Path

klasy = ['R', 'C', 'D', 'T', 'S', 'L', 'B']

for klasa in klasy:

    input_dir = Path(os.path.dirname(os.path.abspath(__file__)))/"aso_single_photos"/f'{klasa}'
    output_dir = Path(os.path.dirname(os.path.abspath(__file__)))/"nowy_out5"

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    output_dir = Path(os.path.dirname(os.path.abspath(__file__)))/"nowy_out5"/f'{klasa}'

    if not output_dir.exists():
        output_dir.mkdir(parents=True)


    image_files = os.listdir(input_dir)

    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        if not image_path.endswith('.txt'):
            image = cv2.imread(image_path)
            image = cv2.resize(image, (64,64))
            assert image is not None 
        

            gray = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            __, bw = cv2.threshold(cv2.dilate(gray, None), 128, 255, cv2.THRESH_BINARY or cv2.THRESH_OTSU)
            gray = cv2.bitwise_and(gray, bw)

            # Scan each row and remove horizontal lines
            for row in range(gray.shape[0]):
                avg = np.average(gray[row, :] > 16)
                if avg > 0.82:
                    cv2.line(gray, (0, row), (gray.shape[1] - 1, row), (0, 0, 0), 1)

            for col in range(gray.shape[1]):
                avg = np.average(gray[:, col] > 16)
                if avg > 0.75:
                    cv2.line(gray, (col, 0), (col, gray.shape[0] - 1), (0, 0, 0), 1)

            gray = cv2.fastNlMeansDenoising(gray, None,  10, 7, 21)
            _, gray = cv2.threshold(gray,60, 255, cv2.THRESH_BINARY)
            st = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 1)) 
            gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, st, iterations=2)
            st = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
            gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, st, iterations=1)

            # Write the processed image to the output directory
            output_path = os.path.join(output_dir, image_file)
            cv2.imwrite(output_path, gray)

