"""mandel+.py — 11 variantes expérimentales de fractales, animées.

Chaque variante est une règle de récurrence `z ← f(z, c, d)` où `c` est le
point de la grille et `d` un exposant animé. On fait varier `d` de
`D_MIN` à `D_MAX` (700 images par variante) et on assemble une vidéo.

Les fonctions `f35` … `f45` sont des expériences autour de
`z^d + c` où l'on masque/modifie des parties du nombre complexe avec
`abs()` (partie réelle ou imaginaire), ou l'on combine avec `tanh`.
Elles sont numérotées d'après leur rang dans un prototypage itératif.

Sorties :
    output/{f35,…}/f45_*.png   images intermédiaires (nettoyées à chaque run)
    videos/{f35,…}.mp4         11 vidéos de 700 images à 10 fps

Dépendances : numpy, matplotlib, tqdm, numba, **lib**
    (fournit `njit`, `prange`, `save_escape_image`, `images_to_video`)

Usage :
    python "mandel+.py"
"""

import glob
import os
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
from tqdm import tqdm

from lib import images_to_video, njit, prange, save_escape_image

# =============================
# PARAMÈTRES GLOBAUX
# =============================
WIDTH, HEIGHT = 1200, 1200   # résolution (px)
MAX_ITER = 150               # itérations max par pixel
D_MIN, D_MAX, N_IMAGES = 1.0, 7.0, 700   # plage de l'exposant animé
FPS = 10

# =============================
# FRACTALES (NUMBA)
# =============================
# Notation : `_abs(z)` masque la partie réelle de z, `abs_(z)` masque la
# partie imaginaire, `_abs_(z)` masque les deux. Une apostrophe devant
# = « partie réelle masquée », après = « partie imaginaire masquée ».
# Chaque fNN est ensuite un produit de ces masques avec une non-linéarité
# (identité, tanh, ou exponentiation).

@njit(fastmath=True)
def _abs(z, c, d): return (abs(z.real) + 1j * z.imag) ** d + c      # |Re z| + i·Im z

@njit(fastmath=True)
def abs_(z, c, d): return (z.real + 1j * abs(z.imag)) ** d + c

@njit(fastmath=True)
def _abs_(z, c, d): return (abs(z.real) + 1j * abs(z.imag)) ** d + c

@njit(fastmath=True)
def norm(z, c, d): return z ** d + c

@njit(fastmath=True)
def _absabs_(z, c, d):
    return _abs(z, c, d) * abs_(z, c, d)

@njit(fastmath=True)
def _abs_abs_(z, c, d):
    return _abs(z, c, d) * _abs_(z, c, d)

@njit(fastmath=True)
def abs__abs_(z, c, d):
    return abs_(z, c, d) * _abs_(z, c, d)

@njit(fastmath=True)
def f35(z, c, d):
    return _abs_abs_(z, c, d) * (c**(z**d-1))

@njit(fastmath=True)
def f36(z, c, d):
    return abs__abs_(z, c, d) * (c**(z**d-1))

@njit(fastmath=True)
def f37(z, c, d):
    return _absabs_(z, c, d) * (np.tanh(z**d)+c)

@njit(fastmath=True)
def f38(z, c, d):
    return _abs_abs_(z, c, d) * (np.tanh(z**d)+c)

@njit(fastmath=True)
def f39(z, c, d):
    return abs__abs_(z, c, d) * (np.tanh(z**d)+c)

@njit(fastmath=True)
def f40(z, c, d):
    return _absabs_(z, c, d) * np.tanh(z**d+np.arctan(c))

@njit(fastmath=True)
def f41(z, c, d):
    return _abs_abs_(z, c, d) * np.tanh(z**d+np.arctan(c))

@njit(fastmath=True)
def f42(z, c, d):
    return abs__abs_(z, c, d) * np.tanh(z**d+np.arctan(c))

@njit(fastmath=True)
def f43(z, c, d):
    return _absabs_(z, c, d) * np.tanh(c**((z**d)/c))

@njit(fastmath=True)
def f44(z, c, d):
    return _abs_abs_(z, c, d) * np.tanh(c**((z**d)/c))

@njit(fastmath=True)
def f45(z, c, d):
    return abs__abs_(z, c, d) * np.tanh(c**((z**d)/c))

# =============================
# LISTE DES FRACTALES
# =============================
# (nom, fonction, re_min, re_max, im_min, im_max) — viewport propre à chaque
# variante, ajusté à l'étendue réellement visible de l'ensemble.
fractales = [
    ("f35", f35, -2.5, 2.5, -2.5, 2.5),
    ("f36", f36, -2.5, 2.5, -2.5, 2.5),
    ("f37", f37, -1.5, 1.5, -1.5, 1.5),
    ("f38", f38, -1.5, 1.5, -1.5, 1.5),
    ("f39", f39, -1.5, 1.5, -1.5, 1.5),
    ("f40", f40, -2, 2, -2, 2),
    ("f41", f41, -2, 2, -2, 2),
    ("f42", f42, -2, 2, -2, 2),
    ("f43", f43, -2, 2, -2, 2),
    ("f44", f44, -2, 2, -2, 2),
    ("f45", f45, -2, 2, -2, 2),
]


# =============================
# MOTEUR NUMBA
# =============================
@njit(parallel=True, fastmath=True)
def compute_escape(C, d, max_iter, func):
    """Itère z ← func(z, c, d) pixel par pixel et renvoie le rang d'échappement.

    Sortie dès que |z|² > 4 : on ne paie que les itérations réellement
    effectuées, et aucune grille d'état (24 Mo) n'est allouée.
    """
    h, w = C.shape
    escape = np.full((h, w), max_iter, np.int32)

    for i in prange(h):
        for j in range(w):
            c = C[i, j]
            z = 0.0 + 0.0j
            for n in range(max_iter):
                z = func(z, c, d)
                if z.real * z.real + z.imag * z.imag > 4.0:
                    escape[i, j] = n
                    break

    return escape


def build_grid(re_min, re_max, im_min, im_max):
    """Grille des c. Ne dépend que du viewport : recalculée une fois par fractale."""
    re = np.linspace(re_min, re_max, WIDTH)
    im = np.linspace(im_min, im_max, HEIGHT)
    return re[np.newaxis, :] + 1j * im[:, np.newaxis]


# =============================
# PROGRAMME PRINCIPAL
# =============================
def main():
    start = time.time()
    os.makedirs("videos", exist_ok=True)
    ds = np.linspace(D_MIN, D_MAX, N_IMAGES)

    for name, func, rmin, rmax, imin, imax in fractales:
        print(f"\n🚀 FRACTALE {name}")

        folder = f"output/{name}"
        os.makedirs(folder, exist_ok=True)

        # Vide le dossier : une image périmée d'un réglage précédent
        # resterait dans le tri et fausserait la vidéo.
        for old in glob.glob(f"{folder}/*.png"):
            os.remove(old)

        C = build_grid(rmin, rmax, imin, imax)

        for d in tqdm(ds, desc=f"{name} images"):
            escape = compute_escape(C, d, MAX_ITER, func)
            save_escape_image(
                escape, f"{folder}/{name}_{d:.4f}.png", figsize_px=(WIDTH, HEIGHT)
            )

        images_to_video(f"{folder}/*.png", f"videos/{name}.mp4", fps=FPS)

    print(f"\n⏱ Temps total : {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
