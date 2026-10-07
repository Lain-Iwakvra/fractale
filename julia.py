"""julia.py — film sur la famille de fractales de Julia.

Échantillonne une courbe dans le plan des paramètres `c` (droite, deux
cercles et la cardioïde — les zones « intéressantes » du diagramme de
Mandelbrot), calcule la fractale de Julia correspondante, puis assemble le
tout en vidéo.

Chaîne de traitement :
    echantillonner_courbe(n)  →  n valeurs de c
    → Pool(multiprocessing)    →  1 PNG par c (``julia_NNN.png``)
    → lib.images_to_video      →  ``julia_video.mp4``

Chaque worker force `set_num_threads(1)` : la parallélisation vient des
processus, pas de Numba (éviterait le sur-booking des cœurs).

Dépendances : numpy, matplotlib, numba, **lib** (fournit `julia`,
`images_to_video`)

Usage :
    python julia.py
    # paramètres : generer_images_parallel(n, taille, limite, iterations)
"""

import time
from multiprocessing import Pool, cpu_count

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lib import images_to_video, julia

try:
    from numba import set_num_threads
except ImportError:
    def set_num_threads(_n):
        pass


def arc_cercle(c, r, t0, t1, n):
    """n points répartis sur l'arc de cercle de centre c et rayon r."""
    t = np.linspace(t0, t1, n, endpoint=False)
    return c + r * (np.cos(t) + 1j * np.sin(t))


def echantillonner_courbe(n):
    """Construit n paramètres c répartis sur les zones de Mandelbrot.

    La courbe combine (de gauche à droite, puis en retour) :
      - une droite  ]-2, -1.368[      (avant la « bulle » principale)
      - un petit arc  autour de -1.309 (bulle périphérique)
      - un arc de cercle de centre -1, r=0.25
      - la cardioïde principale  c = 0.25 + r(t)·e^{it}
    et ses images symétriques, pour un aller-retour fluide dans la vidéo.
    """
    n_d = n // 8
    n_c2 = n // 35
    n_c = n // 10
    n_k = n - 2 * n_c - 2 * n_c2 - 2 * n_d

    droite = np.linspace(-2, -1.368, n_d, endpoint=False) + 0j
    droite_ = np.linspace(-1.368, -2, n_d, endpoint=False) + 0j

    cercle = arc_cercle(-1, 0.25, np.pi, 2 * np.pi, n_c)
    cercle_ = arc_cercle(-1, 0.25, 0, np.pi, n_c)
    cercle2 = arc_cercle(-1.309, 0.059, np.pi, 0, n_c2)
    cercle2_ = arc_cercle(-1.309, 0.059, 2 * np.pi, np.pi, n_c2)

    # Cardioïde : r = -(1 + cos t)/2, décalée de 0.25
    t = np.linspace(2 * np.pi, 0, n_k, endpoint=False)
    r = -0.5 * (1 + np.cos(t))
    cardioide = 0.25 + r * np.exp(1j * t)

    return np.concatenate((
        droite, cercle2, cercle, cardioide, cercle_, cercle2_, droite_
    ))


def worker(args):
    """Task de pool : calcule une fractale Julia et écrit son PNG.

    Reçoit (index, c, taille, limite, iterations) et renvoie le nom du
    fichier produit. L'image est normalisée puis gamma-corrigée (γ = 0.7)
    pour éclaircir les zones sombres avant coloration « inferno ».
    """
    idx, c, taille, limite, iterations = args
    set_num_threads(1)

    img = julia(
        c.real, c.imag, taille, taille, iterations,
        xmin=-2.0, xmax=2.0, ymin=-2.0, ymax=2.0, limite=limite,
    )

    img_norm = img.astype(float)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-10)
    img_norm = np.power(img_norm, 0.7)

    nom = f"julia_{idx:03d}.png"
    plt.imsave(nom, img_norm, cmap="inferno", origin="lower", vmin=0, vmax=1)
    return nom


def generer_images_parallel(n, taille=2000, limite=2.0, iterations=150):
    """Calcule n fractales Julia en parallèle (un processus par c).

    n           : nombre d'images (et de valeurs de c)
    taille      : résolution carrée de chaque image (px)
    limite      : rayon d'échappement (2.0 = standard)
    iterations  : nombre d'itérations maximum par pixel
    """
    courbe = echantillonner_courbe(n)
    args = [(i, c, taille, limite, iterations) for i, c in enumerate(courbe)]
    with Pool(processes=cpu_count()) as pool:
        for nom in pool.imap_unordered(worker, args):
            print(f"Généré : {nom}")


if __name__ == "__main__":
    start = time.time()
    generer_images_parallel(100)                     # 100 images de 2000²
    images_to_video("julia_*.png", "julia_video.mp4")
    print(f"Temps total : {time.time() - start:.2f}s")

