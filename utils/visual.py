import cv2
import numpy as np


def draw_boxes_on_image(img, windows_list):
    """
    Cette fonction permet de dessiner des boites sur une image.
    :param img: Image sur laquelle dessiner les boites.
    :param windows_list: Liste des coordonnées des boites à dessiner. Format ((x1, y1), (x2, y2)).
    :return: Image avec les boites dessinées.
    """

    img_copy = img.copy()
    for (x1, y1), (x2, y2) in windows_list:
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img_copy
