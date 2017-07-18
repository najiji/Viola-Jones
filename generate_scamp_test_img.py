import numpy as np
import random
from PIL import Image
import os


def get_random_image_name():
    face_path = 'train/face/'
    non_face_path = 'train/non-face/'
    face = random.randint(0,1)
    file_name = random.choice(os.listdir(face_path if face else non_face_path))
    return (face_path if face else non_face_path) + file_name, face


def generate_validation_image():
    images = map(Image.open, [get_random_image_name()[0] for _ in range(169)])
    new_img = Image.new('RGB', (256, 256))
    tlx = 0
    tly = 0
    for img in images:
        new_img.paste(img, (tlx, tly))
        if tlx < 12*19:
            tlx += 19
        else:
            tlx = 0
            tly += 19
    new_img.save('test.png')


def main():
    generate_validation_image()


if __name__ == '__main__':
    random.seed(109993439)
    main()