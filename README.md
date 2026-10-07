# Fractales

Trois scripts de génération de fractales, tous parallélisés (multiprocessing +
Numba) et produisant des vidéos MP4.

> **Dépendance commune : ce projet nécessite le dépôt [`lib`](../lib).**
> Voir [Installation](#installation).

| Script | Description | Sortie |
|---|---|---|
| `julia.py` | Film parcourant une courbe du plan des paramètres `c` (droite, cercles, cardioïde) | `julia_video.mp4` + 100 PNG |
| `Mandelbrot.py` | Ensembles Multibrot `z → z^d + c` pour `d` = 1.0 → 9.9 | `mandelbrot_video.mp4` + 90 PNG |
| `mandel+.py` | 11 variantes expérimentales (`f35` … `f45`) avec masquage `abs()` et `tanh` | `videos/*.mp4` (11 vidéos de 700 images) |

## Installation

```bash
git clone https://github.com/<ton_compte>/lib.git
cd lib && pip install numpy numba
```

Puis, à côté du dépôt courant :

```bash
ln -s ../lib/lib.py .
```

Dépendances du projet :

```bash
pip install numpy matplotlib numba tqdm
```

## Utilisation

```bash
python julia.py
python Mandelbrot.py
python "mandel+.py"        # le nom contient un « + », pensez aux guillemets
```

> **Premier lancement :** Numba compile les fonctions `@njit`, ajoute
> quelques secondes. Les exécutions suivantes sont rapides (cache).
>
> **Espace disque :** `mandel+.py` produit ~8 400 PNG de 1200×1200
> (dossier `output/`, vidé automatiquement à chaque lancement) plus
> 11 vidéos.

## Réglages

Tous les paramètres sont en tête de chaque script :

| Script | Paramètres |
|---|---|
| `julia.py` | `generer_images_parallel(n, taille=2000, limite=2.0, iterations=150)` |
| `Mandelbrot.py` | `width, height, max_iter, re_min/re_max, im_min/im_max` |
| `mandel+.py` | `WIDTH, HEIGHT, MAX_ITER, D_MIN, D_MAX, N_IMAGES, FPS` |

## Architecture

```
multiprocessing.Pool(cpu_count())
   └─ worker(d)                       # 1 processus = 1 image
        └─ lib.multibrot / lib.julia  # Numba @njit(parallel=True), 1 thread
             └─ save_escape_image()   # PNG sans axes
                  └─ images_to_video()# assemblage MP4 (OpenCV)
```

Chaque worker force `set_num_threads(1)` : la parallélisation vient des
processus, pas de Numba — sinon les deux niveaux se sur-bookent.

## Licence

Projet personnel — Paul Fournier.
