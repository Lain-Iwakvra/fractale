"""Mandelbrot.py — cartographie des Multibrots z → z^d + c.

Fait varier l'exposant `d` de 1.0 à 9.9 par pas de 0.1 (90 images) et
produit pour chacun le rendu de l'ensemble de échappement correspondant.
Pour `d = 2` on retrouve l'ensemble de Mandelbrot classique ; en faisant
varier `d` on voit l'ensemble se déformer continûment.

Chaîne de traitement :
    Pool(cpu_count()) → `multibrot(d, …)` (Numba, parallèle)
    → lib.save_escape_image → ``mandelbrot_{d:.2f}.png``
    → lib.images_to_video   → ``mandelbrot_video.mp4``

Chaque worker force `set_num_threads(1)` : la parallélisation vient des
processus, pas de Numba (éviterait le sur-booking des cœurs).

Dépendances : **lib** (fournit `multibrot`, `save_escape_image`,
`images_to_video`), numba recommandé.

Usage :
    python Mandelbrot.py
"""

import time
from multiprocessing import Pool, cpu_count

from lib import images_to_video, multibrot, save_escape_image

try:
    from numba import set_num_threads
except ImportError:
    def set_num_threads(_n):
        pass

# --- Réglages de l'image ------------------------------------------------
width, height = 1200, 1200    # résolution (px)
max_iter = 150                # itérations max par pixel
re_min, re_max = -2.5, 2      # plage de l'axe réel
im_min, im_max = -2.2, 2.2    # plage de l'axe imaginaire


def multibrot_save(d):
    """Task de pool : calcule l'ensemble Multibrot d'exposant d, écrit le PNG."""
    set_num_threads(1)
    escape = multibrot(d, width, height, max_iter, re_min, re_max, im_min, im_max)
    filename = f"mandelbrot_{d:.2f}.png"
    save_escape_image(escape, filename, cmap="bone", figsize_px=(width, height))
    print(f"✔ Image enregistrée : {filename}")


if __name__ == "__main__":
    start = time.time()
    # d = 1.0, 1.1, … 9.9  →  90 images
    d_values = [i * 0.1 for i in range(10, 100)]
    with Pool(cpu_count()) as pool:
        pool.map(multibrot_save, d_values)
    images_to_video("mandelbrot_*.png", "mandelbrot_video.mp4")
    print(f"\n⏱ Temps total : {time.time() - start:.2f} s")

