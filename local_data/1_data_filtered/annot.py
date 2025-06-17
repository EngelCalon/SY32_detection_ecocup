#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 13:02:48 2024

@author: moreajul

Annotation tool for SY32 project

Arborescence requise :
    "dossier"
    |- "images"
        |- "neg"
        |- "pos"
    |- "labels_csv"
Les images à annoter doivent se trouver dans le sous-dossier "images/pos".
Après les annotations, chaque image aura un csv associé avec le même nom dans le dossier "labels_csv".

Chaque ligne du csv contient une boîte englobante annotée comme suit : coinhgY,coinhgX,Hauteur,Largeur,Difficulté[0;1]
L'ordre Y, X correspond à l'ordre d'accès des coordonnées Numpy

Usage :
annot.py dossier


Améliorations possibles :
    - éditer et ajouter des annotations à des données déjà partiellement annotées
    - interface pour passer facilement d'une image à une autre ?
"""

import sys
import os
import csv
import numpy as np
#import cv2
from skimage import data
import matplotlib.pyplot as plt
from matplotlib.widgets  import RectangleSelector


I = data.chelsea()

# idée d'amélioration, pré-segmentation automatique où l'on choisit des groupes de pixels formant l'objet contenu dans une boîte au plus proche
# superpiel avec méthode ILSC "Superpixel segmentation using Linear Spectral Clustering", 2015
#I2 = cv2.ximgproc.createSuperpixelLSC(I)
#cv2.imshow('I2',I2)

#%%

class BBAnnotator():
    
    rs = None # sera le RectangleSelector
    fig = None # sera le fig matpltotlib
    ax = None # sera le ax matplotlib
    lblfile = None # fichier des labels à écrire
    lbls = []
    rects = [] # liste des instances de plt.Rectangle, partage d'index avec lbls
    edited_difficulty = None # variable encodant à la fois si un label est en édition (None : non, sinon oui) et la difficulté du label édité (0 ou 1)
    stop = False
    
    def __init__(self):
        pass
    
    def set_lblfile(self,lblfile):
        self.lblfile = lblfile
        self.lbls = []
        self.rects = []
        self.edited_difficulty = None

    def load_existing_labels(self):

        if not os.path.exists(self.lblfile):
            return
        
        print("Importation des labels existants pour cette image :")
        with open(self.lblfile, "r", newline='') as csvf:
            spamreader = csv.reader(csvf, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            for row in spamreader:
                row = [int(v) for v in row]
                self.lbls.append(row)
                coinhgy, coinhgx, hauteur,largeur, is_difficult = row
                self.add_rect(coinhgx, coinhgx + largeur, coinhgy, coinhgy + hauteur, is_difficult)
        print(f"{len(self.lbls)} annotations existantes")

    
    def line_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        print(" The button you used were: %s %s" % (eclick.button, erelease.button))


    def add_rect(self, xmin, xmax, ymin, ymax, is_difficult):
        rect = plt.Rectangle( (xmin,ymin), xmax-xmin, ymax-ymin, fill=False, color='r' if is_difficult else 'g', lw=2 )
        self.rects.append(rect)
        self.ax.add_patch(rect)
        self.fig.canvas.draw()

    def save_label(self, is_difficult):
        (xmin, xmax, ymin, ymax) = self.rs.extents
        # dans le csv, ordre coinhgy,coinhgx,hauteur,largeur
        self.lbls.append([ymin,xmin,ymax-ymin,xmax-xmin,is_difficult])
        self.add_rect(xmin, xmax, ymin, ymax, is_difficult)
        self.edited_difficulty = None # plus besoin de chain, le label en édition a été sauvegardé (avec un cas "ctrl+..." ou "right")

    def toggle_selector(self, event):
        #print(' Key pressed.')
        if event.key in ["Q", "q"]: # and toggle_selector.RS.active:
            self.stop = True
        #    print(' RectangleSelector deactivated.')
        #    toggle_selector.RS.set_active(False)
        #if event.key in ["A", "a"]: # and not toggle_selector.RS.active:
        #    print(' RectangleSelector activated.')
        #    toggle_selector.RS.set_active(True)
        #if event.key == "enter":
        if event.key in ["ctrl+0", "ctrl+1"]:
            # permet de sauvegarder l'emplacement de self.rs comme un label
            is_difficult = int(event.key[-1]) # "ctrl+0" -> 0 | "ctrl+1" -> 1
            print(f"Boîte enregistrée (de difficulté {is_difficult})")
            self.save_label(is_difficult)

        if event.key == "d":
            # permet de supprimer un label en édition
            self.edited_difficulty = None # perte de l'information qui permettait de le réenregistrer avec le cas "right"

        if event.key == "right" and (len(self.lbls) > 0 or (len(self.lbls) == 0 and self.edited_difficulty is not None)) : 
            # permet d'éditer le prochain label "à droite" dans la liste
            # lbls et rects peuvent être considérées comme des FIFOs
            # pour avancer dans la liste, il suffit de pop le premier label et de le replacer à la fin
            
            # Replace d'abord le label en cas d'édition précédente (self.edited_difficulty non null) (vient d'un précédent cas "right")
            if self.edited_difficulty is not None:
                self.save_label(self.edited_difficulty)

            coinhgy, coinhgx, hauteur,largeur, self.edited_difficulty = self.lbls.pop(0)
            rect = self.rects.pop(0)
            rect.remove() # de façon à ce que le visuel match les données
            self.fig.canvas.draw()
            self.rs.extents = (coinhgx, coinhgx + largeur, coinhgy, coinhgy + hauteur)

        if event.key == " ":
            
            # S'il restait une boîte en édition -> la sauvegarder
            if self.edited_difficulty is not None:
                self.save_label(self.edited_difficulty)

            if (len(self.lbls) > 0):
                print('Enregistrement des annotations dans {} ...'.format(self.lblfile))
                with open(self.lblfile, "w", newline='') as csvf:
                    spamwriter = csv.writer(csvf, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                    for l in self.lbls:
                        l = [int(v) for v in l]
                        spamwriter.writerow(l)
                print('{} annotations enregistrées !'.format(len(self.lbls)))
            print("Image suivante")
            plt.close()

    def open_interactive_plot(self, I):
        self.fig, self.ax = plt.subplots()
        plt.gca().invert_yaxis() # axes adaptés aux images et non aux courbes
        self.ax.set_title("Cliquer pour construire la boîte, Ctrl+0 ou +1 pour enregistrer la boîte, Espace pour passer à l'image suivante, Q pour arrêter\n" \
        "Flèche droite pour éditer la prochaine boîte, D pour supprimer la boîte en édition")
        #ax.axis('off')
        self.ax.imshow(I)

        self.load_existing_labels()

        #self.toggle_selector.RS = 
        self.rs = RectangleSelector(self.ax, None, #line_select_callback,
                                    minspanx=5, minspany=5,
                                    useblit=True,
                                    button=[1, 3],  # don't use middle button
                                    spancoords='pixels',
                                    interactive=True)
        plt.connect('key_press_event', self.toggle_selector)
        
        plt.show()
        
        return not self.stop

#%%

def main(imgdir, lbldir):
    bba = BBAnnotator()
    for imgfile in os.listdir(imgdir):
        if imgfile.endswith('.jpg'):
            print(imgfile)
            name = imgfile[:-4]
            I = plt.imread(os.path.join(imgdir,imgfile))
            bba.set_lblfile(os.path.join(lbldir,name + '.csv'))
            encore = bba.open_interactive_plot(I) # plt est bloquant, l'image suivante s'ouvre à la fermeture de l'actuelle
            if not encore:
                break
            

#%%
if __name__ == "__main__":
    usage = "Usage: python annot.py dossier\n avec dossier contenant les sous-dossiers images/pos et labels"
    
    if len(sys.argv) != 2:
        print(usage)
        exit(1)
        
    workdir = sys.argv[1]
    if not os.path.isdir(workdir):
        print(usage)
        exit(2)
    imgdir = os.path.join(workdir,"images","pos")
    lbldir = os.path.join(workdir,"labels_csv")
    if not os.path.isdir(imgdir):
        print(usage)
        exit(3)
    if not os.path.exists(lbldir):
        os.mkdir(lbldir)
    if not os.path.isdir(lbldir): # is a file!
        print(usage)
        exit(4)

    main(imgdir, lbldir)