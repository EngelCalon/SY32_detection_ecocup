import os
import numpy as np
import matplotlib.pyplot as plt
import random
from functools import partial
import shutil



import albumentations as A

TEST = False # booléen. Si True, il n'y a qu'un patch qui est traité, pour voir le résultat de la génération de l'augmentation. Si False, tous les patchs sont traités

def augmenter(patchs_base: dict):

    # Note : Finalement, pas besoin de faire des rotations à 90°, les patchs seront réorientés avant normalisation

    # Ce qui a été retenu comme augmentations utiles (il faut aussi réfléchir à l'explosion du nombre de données qui deviennent longues à traiter)
    transforms = {
    "original": A.Compose([]),  # identité
    "horizontal": A.Compose([A.HorizontalFlip(p=1.0)]),
    "vertical": A.Compose([A.VerticalFlip(p=1.0)]),
    "horizontal_vertical": A.Compose([
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0)
    ]),
}
    
    augmented_patchs = {}

    for patch_name, patch in patchs_base.items():
        for transform_name, transform in transforms.items():
            augmented_patchs[f"{patch_name}_{transform_name}"] = transform(image=patch)["image"]

    return augmented_patchs


##############################################################################################
########################### Utilitaires ######################################################
##############################################################################################

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

    # Chargement des patchs positifs
    pos_patch_fs = [f for f in  os.listdir(os.path.join("local_data", "2_patches", "pos")) if f.endswith(".jpg")]
    pos_patchs_base = {}
    for f in pos_patch_fs:
        try:
            patch = plt.imread(
                os.path.join("local_data", "2_patches", "pos", f)
            )
            pos_patchs_base[os.path.splitext(f)[0]] = patch

        except FileNotFoundError:
            continue

        if TEST: # seulement 1 patch est gardé pour tester le traitement
            break
        

    # Chargement des patchs négatifs
    neg_patch_fs = [f for f in  os.listdir(os.path.join("local_data", "2_patches", "neg")) if f.endswith(".jpg")]
    neg_patchs_base = {}
    for f in neg_patch_fs:
        try:
            patch = plt.imread(
                os.path.join("local_data", "2_patches", "neg", f)
            )
            neg_patchs_base[os.path.splitext(f)[0]] = patch

        except FileNotFoundError:
            continue

        if TEST: # seulement 1 patch est gardé pour tester le traitement
            break
        
    print(f"len pos_patchs_base : {len(pos_patchs_base)}")
    print(f"len neg_patchs_base : {len(neg_patchs_base)}") 
    print()   

    return pos_patchs_base, neg_patchs_base


def save(pos_patchs: dict, neg_patchs: dict):

    # Clear des dossiers avant mise à jour
    clear_folder(os.path.join("local_data", "3_augmented_patches", "pos"))
    clear_folder(os.path.join("local_data", "3_augmented_patches", "neg"))

    for patch_name, patch in pos_patchs.items():
        save_file = os.path.join("local_data", "3_augmented_patches", "pos", f"{patch_name}.jpg")
        plt.imsave(save_file, patch)

    for patch_name, patch in neg_patchs.items():
        save_file = os.path.join("local_data", "3_augmented_patches", "neg", f"{patch_name}.jpg")
        plt.imsave(save_file, patch)

    print(f"len pos_patchs : {len(pos_patchs)}")
    print(f"len neg_patchs : {len(neg_patchs)}")
    print()


if __name__ == "__main__":
    pos_patchs_base, neg_patchs_base = load()
    pos_patchs = augmenter(pos_patchs_base)
    neg_patchs = augmenter(neg_patchs_base)
    save(pos_patchs, neg_patchs)



