
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import shutil

# Paramètres 
NB_NEG_FACTOR = 5       # coefficient : nombre de patchs négatifs = ce coeff * nb patchs positifs
MIN_PATCH_AREA = 30     # px² : aire minimale pour qu'un patch soit considéré valide (sinon considéré comme une erreur d'annotation) 
KEEP_DIFFICULT = False  # booléenne. Si True : les annotations marquées 0 (faciles) ET 1 (difficiles) sont découpées, sinon seuelement celles faciles

def splitter():

    # De manière à ce que la génération aléatoire soit répétable
    random.seed(0)

    # Clear des dossiers avant mise à jour
    clear_folder(os.path.join("local_data", "2_patches", "pos"))
    clear_folder(os.path.join("local_data", "2_patches", "neg"))
    
    print(f"Rappel des paramètres:")
    print(f"\tNB_NEG_FACTOR = {NB_NEG_FACTOR}")
    print(f"\tMIN_PATCH_AREA = {MIN_PATCH_AREA}")
    print(f"\tKEEP_DIFFICULT = {KEEP_DIFFICULT}\n")

    # Recherche des fichiers de labels
    f_labels = [
        os.path.splitext(f)[0]
        for f in os.listdir(os.path.join("local_data", "1_data_filtered", "train", "labels_csv"))
        if f.endswith(".csv")
    ]
    print(f"f_labels: {f_labels[:5]} ... len:{len(f_labels)}\n")

    img_pos_train = {}
    img_neg_train = {}
    bbox_train = {}

    # Extraction des données des labels et des images correspondantes
    for f in f_labels:

        # parsing grâce à np.loadtxt
        bbox = np.loadtxt(os.path.join("local_data","1_data_filtered", "train", "labels_csv", f"{f}.csv"), delimiter=",")
        
        # Plusieurs cas possibles : 
        
        # le .csv est vide
        if bbox.size == 0:
            continue

        # le .csv ne contient qu'une annotation -> on force le résultat dans une liste de liste (ce qui est le cas s'il contient plusieurs annotations)
        elif bbox.ndim == 1:
            bbox = [bbox]


        # Extraction de l'image correspondante
        try:
            img_pos = plt.imread(
                os.path.join("local_data", "1_data_filtered", "train", "images", "pos", f"{f}.jpg")
            )
            
        except FileNotFoundError:
            continue

        # Enregistrement des appairages que si le tout est cohérent
        img_pos_train[f"{f}"] = img_pos
        bbox_train[f"{f}"] = bbox


    # Chargement des images négatives
    neg_paths = [os.path.splitext(f)[0] for f in  os.listdir(os.path.join("local_data", "1_data_filtered", "train", "images", "neg")) if f.endswith(".jpg")]
    for f in neg_paths:
        try:
            img_neg = plt.imread(
                os.path.join("local_data", "1_data_filtered", "train", "images", "neg", f"{f}.jpg")
            )
            img_neg_train[f"{f}"] = img_neg

        except FileNotFoundError:
            continue

    print(f"len img_pos_train : {len(img_pos_train)}")
    print(f"len bbox_train : {len(bbox_train)}")
    print(f"len img_neg_train : {len(img_neg_train)}\n")


    # Découpe des images positives, filtre et enregistrement
    nb_pos_patches = 0
    for f_name in img_pos_train.keys():
        bbox_list = bbox_train[f_name]
        img = img_pos_train[f_name]

        for j, bbox in enumerate(bbox_list):
            upper_left_corner_Y = int(bbox[0])
            upper_left_corner_X = int(bbox[1])
            height = int(bbox[2])
            width = int(bbox[3])
            is_difficult = bool(bbox[4])

            # Filtres sur les patchs positifs
            if height*width < MIN_PATCH_AREA:
                print(f"pos/{f_name}.png -> bbox n°{j} : Patch trop petit détecté")
                continue
            if is_difficult and not KEEP_DIFFICULT:
                continue

            nb_pos_patches +=1

            lower_right_corner_Y = upper_left_corner_Y + height
            lower_right_corner_X = upper_left_corner_X + width

            img_cut = img[
                upper_left_corner_Y:lower_right_corner_Y + 1,
                upper_left_corner_X:lower_right_corner_X + 1,
            ]

            save_file = os.path.join("local_data", "2_patches", "pos",f"{f_name}_{j:02d}")
            save_file += "_d" if is_difficult else ""
            save_file += ".jpg"
            plt.imsave(save_file, img_cut)

    print()
    print(f"Nombre de patchs positifs conservés : {nb_pos_patches}")

    neg_f_names = list(img_neg_train.keys())
    all_bboxs = list(bbox_train.values())
    # Découpe de patchs négatifs (uniquement sur les images complètement négatives) et enregistrement
    nb_neg_patches = 0
    while nb_neg_patches < nb_pos_patches * NB_NEG_FACTOR:

        f_name = random.choice(neg_f_names)
        img = img_neg_train[f_name]
        random_bbox = random.choice(random.choice(all_bboxs))

        upper_left_corner_Y = int(random_bbox[0])
        upper_left_corner_X = int(random_bbox[1])
        height = int(random_bbox[2])
        width = int(random_bbox[3])
        is_difficult = bool(random_bbox[4])

        # Mêmes filtres sur les patchs positifs (pour garder la cohérence des tailles des patchs entre pos et neg)
        if height*width < MIN_PATCH_AREA:
            continue
        if is_difficult and not KEEP_DIFFICULT:
            continue


        lower_right_corner_Y = upper_left_corner_Y + height
        lower_right_corner_X = upper_left_corner_X + width

        img_cut = img[
            upper_left_corner_Y:lower_right_corner_Y + 1,
            upper_left_corner_X:lower_right_corner_X + 1,
        ]

        save_file = os.path.join("local_data", "2_patches", "neg",f"{f_name}_{nb_neg_patches:05d}.jpg")
        try:
            plt.imsave(save_file, img_cut)
        except ValueError:
            continue # cas où l'image découpée ne continent rien (sûrement à cause d'un bbox sur un bord)
        nb_neg_patches +=1

    print()
    print(f"Nombre de patchs négatifs conservés : {nb_neg_patches}")


# Généré avec chatGPT
def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # supprime le fichier ou le lien
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # supprime le dossier récursivement
        except Exception as e:
            print(f"Erreur lors de la suppression de {file_path} : {e}")


if __name__ == "__main__":
    splitter()