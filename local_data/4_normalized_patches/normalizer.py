import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2gray

TARGET_SCALE = 128 # Paramètre d'échelle des patchs normalisés arbitraire

def normalizer(patchs_base: dict):
    target_shape = get_target_shape()
    patchs_normalized = {}
    for patch_name, patch in patchs_base.items():
        patchs_normalized[patch_name] = normalize_patch(target_shape, patch)
    return patchs_normalized


def get_target_shape():
    shape_tuple = (TARGET_SCALE, TARGET_SCALE) # valeur par défaut
    with open(os.path.join("local_data", "4_normalized_patches", "target_shape.txt"), "r", encoding="utf-8") as f:
        line = f.readline().strip()
        # Extraire le contenu entre les parenthèses
        if line.startswith("target_shape="):
            shape_str = line.split("=", 1)[1]
            shape_tuple = eval(shape_str)
    return shape_tuple

# Fonction qui peut être aussi appelée par l'algorithme de détection itératif (après avoir découpé une fenêtre)
def normalize_patch(target_shape, patch):
    """
    patch: nd.array (height, width, 3) (3 car RGB)
    """
    if patch.ndim == 3:
        patch = rgb2gray(patch)
    height = patch.shape[0]
    width = patch.shape[1]
    if width > height:
        patch = rotate_90(patch)

    return resize(patch, target_shape, anti_aliasing=True)

##############################################################################################
########################### Utilitaires ######################################################
##############################################################################################

def rotate_90(image):
    return np.ascontiguousarray(np.rot90(image, k=1))

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

def load():

    # Chargement des patchs positifs augmentés
    pos_patch_fs = [f for f in  os.listdir(os.path.join("local_data", "3_augmented_patches", "pos")) if f.endswith(".jpg")]
    pos_patchs_base = {}
    for f in pos_patch_fs:
        try:
            patch = plt.imread(
                os.path.join("local_data", "3_augmented_patches", "pos", f)
            )
            pos_patchs_base[os.path.splitext(f)[0]] = patch

        except FileNotFoundError:
            continue


    # Calculs statistiques pour calculer la target_shape -> dans stats.txt et target_shape.txt
    # Uniquement avec les patchs positifs
    ratios = [] # width/height
    max_scale = 0 # "max_height" donc
    min_scale = float("inf")

    for patch in pos_patchs_base.values():
        height = patch.shape[0]
        width = patch.shape[1]
        if width > height:
            height, width = width, height # inversion des définitons (plutôt que de retourner "physiquement" l'image)
        
        ratios.append(width/height)

        if height > max_scale:
            max_scale = height

        if height < min_scale:
            min_scale = height

    ratios = np.array(ratios)
    moy_ratio = np.mean(ratios)

    with open(os.path.join("local_data", "4_normalized_patches", "stats.txt"), "w", encoding="utf-8") as f:
        f.write(f"min_ratio={np.min(ratios)}\n")
        f.write(f"max_ratio={np.max(ratios)}\n")
        f.write(f"moy_ratio={moy_ratio}\n")
        f.write(f"min_scale={min_scale}\n")
        f.write(f"max_scale={max_scale}")

    print("Statistiques des données : ")
    print(f"\tmin_ratio={np.min(ratios)}")
    print(f"\tmax_ratio={np.max(ratios)}")
    print(f"\tmoy_ratio={moy_ratio}")
    print(f"\tmin_scale={min_scale}")
    print(f"\tmax_scale={max_scale}")
    print()

    target_width = int(moy_ratio * TARGET_SCALE)
    target_width = target_width - target_width%8 # on force la largeur à être aussi un multiple de 8

    target_shape = (int(TARGET_SCALE), int(target_width)) # height, width
    with open(os.path.join("local_data", "4_normalized_patches", "target_shape.txt"), "w", encoding="utf-8") as f:
        f.write(f"target_shape={target_shape}")

    print(f"target_shape={target_shape}\n")

        

    # Chargement des patchs négatifs
    neg_patch_fs = [f for f in  os.listdir(os.path.join("local_data", "3_augmented_patches", "neg")) if f.endswith(".jpg")]
    neg_patchs_base = {}
    for f in neg_patch_fs:
        try:
            patch = plt.imread(
                os.path.join("local_data", "3_augmented_patches", "neg", f)
            )
            neg_patchs_base[os.path.splitext(f)[0]] = patch

        except FileNotFoundError:
            continue
        
    print(f"len pos_patchs_base : {len(pos_patchs_base)}")
    print(f"len neg_patchs_base : {len(neg_patchs_base)}") 
    print()   

    return pos_patchs_base, neg_patchs_base


def save(pos_patchs: dict, neg_patchs: dict):

    # Clear des dossiers avant mise à jour
    clear_folder(os.path.join("local_data", "4_normalized_patches", "pos"))
    clear_folder(os.path.join("local_data", "4_normalized_patches", "neg"))

    for patch_name, patch in pos_patchs.items():
        save_file = os.path.join("local_data", "4_normalized_patches", "pos", f"{patch_name}.jpg")
        plt.imsave(save_file, patch, cmap="gray")

    for patch_name, patch in neg_patchs.items():
        save_file = os.path.join("local_data", "4_normalized_patches", "neg", f"{patch_name}.jpg")
        plt.imsave(save_file, patch, cmap="gray")

    print(f"len pos_patchs : {len(pos_patchs)}")
    print(f"len neg_patchs : {len(neg_patchs)}")
    print()


if __name__ == "__main__":
    pos_patchs_base, neg_patchs_base = load()
    pos_patchs = normalizer(pos_patchs_base)
    neg_patchs = normalizer(neg_patchs_base)
    save(pos_patchs, neg_patchs)