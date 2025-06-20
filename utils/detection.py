from skimage.util import img_as_float
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np

import time
import importlib
from functools import partial

module_name = "local_data.4_normalized_patches.normalizer"
normalizer = importlib.import_module(module_name)
target_shape = normalizer.get_target_shape()
normalize_patch = partial(normalizer.normalize_patch, target_shape)


def sliding_window(img, h, w, x_step, y_step):
    """
    Applique une fenêtre glissante sur une image et retourne les sous-images et leurs coordonnées.
    :param img: Image sur laquelle appliquer la fenêtre glissante.
    :param h: Hauteur de la fenêtre.
    :param w: Largeur de la fenêtre.
    :param x_step: Pas de la fenêtre en hauteur.
    :param y_step: Pas de la fenêtre en largeur.
    :return: Liste des sous-images et liste des coordonnées ((upper_left), (lower_right)).
    """
    img_parts = []
    parts_coords = []

    x_starts = np.arange(0, img.shape[0] - h + 1, x_step)
    y_starts = np.arange(0, img.shape[1] - w + 1, y_step)
    for i in x_starts:
        for j in y_starts:
            upper_left = (i, j)
            lower_right = (i + h, j + w)
            img_parts.append(img[i : i + h, j : j + w])
            parts_coords.append((upper_left, lower_right))
    return img_parts, parts_coords


def get_iou(window1, window2):
    """
    Cet algorithme permet de calculer l'Intersection over Union (IoU), soit aire de recouvrement, entre deux fenetres.
    :param window1: Fenetre 1. Format ((upper_left_x, upper_left_y), (lower_right_x, lower_right_y)).
    :param window2: Fenetre 2.
    :return float: Aire de recouvrement entre les deux fenetres.
    """

    # print("Calcul de l'IoU entre les fenetres : ", window1, window2)

    upper_left_1, lower_right_1 = window1
    upper_left_2, lower_right_2 = window2

    window1_area = (lower_right_1[0] - upper_left_1[0]) * (
        lower_right_1[1] - upper_left_1[1]
    )
    window2_area = (lower_right_2[0] - upper_left_2[0]) * (
        lower_right_2[1] - upper_left_2[1]
    )

    upper_left_inter = (
        max(upper_left_1[0], upper_left_2[0]),
        max(upper_left_1[1], upper_left_2[1]),
    )
    lower_right_inter = (
        min(lower_right_1[0], lower_right_2[0]),
        min(lower_right_1[1], lower_right_2[1]),
    )

    inter_width = max(0, lower_right_inter[0] - upper_left_inter[0])
    inter_height = max(0, lower_right_inter[1] - upper_left_inter[1])

    inter_area = inter_height * inter_width
    union_area = window1_area + window2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


# def group_windows_by_iou(windows_list, decision_criteria=0.5):
#     """
#     Cet algorithme permet de grouper les fenetres qui se recouvrent en fonction de l'Intersection over Union (IoU).
#     :param windows_list: Liste de fenetres. Format ((upper_left_x, upper_left_y), (lower_right_x, lower_right_y), score).
#     :param decision_criteria: Seuil de recouvrement pour regrouper les fenetres.
#     :return: Liste de fenetres regroupées.
#     """
#     grouped_windows = []
#     while windows_list:  # Tant qu'il reste des fenetres à traiter
#         current_window = windows_list.pop(0)
#         group = [
#             other_window
#             for other_window in windows_list
#             if get_iou(current_window[:2], other_window[:2]) > decision_criteria
#         ]
#         # Supprimer les fenetres du groupe de la liste des fenetres restantes
#         for window in group:
#             windows_list.remove(window)
#         group = [current_window] + group
#         grouped_windows.append(group)
#     return grouped_windows

# def non_maxima_suppression(windows_list, decision_criteria=0.5):
#     """
#     Cet algorithme permet de supprimer les fenetres qui se recouvrent trop, en gardant la fenetre avec le score le plus haut.
#     :param windows_list: Liste de liste de fenetres. Format ((upper_left_x, upper_left_y), (lower_right_x, lower_right_y), score).
#     :param decision_criteria: Seuil de recouvrement pour regrouper les fenetres.
#     """
#     grouped_windows = group_windows_by_iou(windows_list, decision_criteria)
#     best_windows = []
#     for group in grouped_windows:
#         if len(group) == 0:
#             continue
#         elif len(group) == 1:
#             best_window = group[0]
#         else:
#             best_window = max(
#                 group, key=lambda x: x[2]
#             )  # On prend la fenetre avec le score le plus haut
#         best_windows.append(best_window)
#     return best_windows


def non_maxima_suppression_v2(
    windows_list, scores, iou_decision_criteria=0.5, score_decision_criteria=0.5, output_score=False
):
    """
    Cet algorithme permet de supprimer les fenetres qui se recouvrent trop, en gardant la fenetre avec le score le plus haut.
    """
    best_windows = []
    best_scores = []

    # Filtrer les fenetres par rapprort au seuil de score
    valid_indices = np.where(scores >= score_decision_criteria)[0]
    windows_list = windows_list[valid_indices]
    scores = scores[valid_indices]

    # Ordonner les fenetres par score
    order = np.argsort(scores)[::-1]
    windows_list = windows_list[order]
    scores = scores[order]

    while len(windows_list) > 0:
        best_window = windows_list[0]
        best_windows.append(best_window)
        best_scores.append(scores[0])

        # Supprimer la meilleure fenetre de la liste
        windows_list = windows_list[1:]
        scores = scores[1:]

        # Supprimer les fenetres qui se recouvrent trop avec la meilleure fenetre
        iou_scores = np.array([get_iou(best_window, window) for window in windows_list])
        valid_indices = np.where(iou_scores < iou_decision_criteria)[0]
        windows_list = windows_list[valid_indices]
        scores = scores[valid_indices]
    
    if output_score:
        return best_windows, best_scores
    
    return best_windows


# ------------------ FONCTION PRINCIPALE DE DETECTION ------------------#
# Fonction à tuner


from skimage.transform import resize
from skimage.util import img_as_float
import numpy as np


def detect_ecocup(img, classifier, features_func, min_ratio, max_ratio, min_scale, max_scale, px_step, scales_nb, ratios_nb, confidence_threshold):
    """
    Détecte les gobelets en plastique dans une image à l'aide d'un classifieur et d'une fenêtre glissante.
    :param img: Image sur laquelle appliquer la détection.
    :param classifier: Classificateur entraîné.
    :param img_target_shape: Shape cible pour la classification.
    :return: Liste des coordonnées des gobelets détectés.
    """

    start = time.time()
    global_start = start

    # Générations des limites de fenêtres à tester
    ratios = np.linspace(min_ratio, max_ratio, ratios_nb)
    scales = np.linspace(min_scale, max_scale, scales_nb)

    # Croisement des couples
    ratios, scales = np.meshgrid(ratios,scales, indexing="ij")

    ratios = ratios.flatten()
    heights = scales.flatten().astype("uint16")
    widths = (ratios*heights).astype("uint16")
    
    all_windows = []
    all_scores = []

    print(f"Temps setup : {time.time() - start}")
    print(f"Nombre d'itérations : {scales_nb * ratios_nb}")
    for i in range(scales_nb * ratios_nb):
        h = heights[i]
        w = widths[i]
        print(f"\n\tItération {i+1} : h={h:04d} | w={w:04d}")

        start = time.time()        
        img_parts, windows_coords = sliding_window(img, h, w, px_step, px_step)
        # Même chose avec le rectangle retourné
        img_parts_2, windows_coords_2 = sliding_window(img, w, h, px_step, px_step)

        img_parts = (img_parts + img_parts_2)
        windows_coords = np.array(windows_coords + windows_coords_2)
        print(f"\tTemps sliding_window x2 ({len(img_parts)} généré): {time.time() - start}")

        if len(img_parts) == 0:
            print(f"\tAbandon de l'itération")
            continue

        start = time.time()
        normalized_parts = [
            normalize_patch(img_part) for img_part in img_parts
        ]
        print(f"\tTemps de traitement pour {len(img_parts)} : {time.time() - start}")

        start = time.time()
        features = features_func(normalized_parts) # np.array
        print(f"\tTemps d'extraction de features pour {len(img_parts)} : {time.time() - start}")

        start = time.time()
        # preds = classifier.predict(features) # pour moi inutile si on calcule déjà les probas ?
        probas = classifier.predict_proba(features)[:, 1]  # proba classe "gobelet" # np.array
        print(f"\tTemps de prédiction pour {len(img_parts)} : {time.time() - start}")

        keep_idx = np.where(probas >= confidence_threshold)[0]
        all_windows += list(windows_coords[keep_idx])
        all_scores += list(probas[keep_idx])
        print(f"\tFenêtres gardées : {len(keep_idx)} sur {len(img_parts)} (p>={confidence_threshold})")
        if len(probas) > 0:
            print(f"\tMeilleur score de l'itération : {np.max(probas)} pour {windows_coords[np.argmax(probas)]}")
        print(f"\tTemps cumulé : {time.time() - global_start}")

    print()
    if not all_windows:
        return np.array([]), np.array([])

    # Conversion en array pour NMS
    all_windows = np.array(all_windows)
    all_scores = np.array(all_scores)

    print(f"Temps total de détection : {time.time() - global_start}")

    return all_windows, all_scores
