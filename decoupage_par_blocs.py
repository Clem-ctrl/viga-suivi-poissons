"""
Controle de fuite : nouveau decoupage entrainement / validation / test par blocs de temps.

Constats sur le decoupage aleatoire d'origine (labelme_vers_yolo_pose.py) :
  - les images completes sont numerotees dans l'ordre chronologique (deux numeros voisins = deux
    images prises a quelques secondes d'intervalle, avec une partie des poissons au meme endroit) ;
  - les recadrages (un poisson) sont decoupes dans ces images completes, y compris dans des images
    du jeu de test : le meme poisson, a la meme seconde, se retrouve en entrainement et en test.

Nouveau decoupage :
  - images completes seulement (les recadrages, ~4 % des poissons annotes, sont ecartes) ;
  - suites de numeros consecutifs coupees en blocs de TAILLE_BLOC images, chaque bloc entier
    attribue a un seul jeu ;
  - zone tampon : les images d'entrainement a moins de TAMPON numeros d'une image de validation
    ou de test sont retirees.
Temoin : un second jeu, data/fish_pose_yolo_alea/, avec les memes effectifs mais un decoupage
aleatoire image par image (sans recadrages ni tampon). Comparer les deux entrainements isole
l'effet de la fuite de celui de la quantite de donnees.
Sorties : data/fish_pose_yolo_blocs.zip et data/fish_pose_yolo_alea.zip, a envoyer sur Kaggle et a
utiliser chacun avec le notebook 01_entrainement_pose (un seul dataset en entree par execution).
"""
import random
import shutil
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parent
SOURCE = RACINE / "data" / "fish_pose_yolo"
SORTIE = RACINE / "data" / "fish_pose_yolo_blocs"
SORTIE_TEMOIN = RACINE / "data" / "fish_pose_yolo_alea"
TAILLE_BLOC = 50
TAMPON = 10
PART_TEST, PART_VAL = 0.10, 0.20
GRAINE = 42


def lire(split):
    images = {}
    for p in sorted((SOURCE / "images" / split).glob("*.jpg")):
        lab = SOURCE / "labels" / split / f"{p.stem}.txt"
        boites = np.loadtxt(lab, ndmin=2)[:, 1:5] if lab.stat().st_size else np.zeros((0, 4))
        images[p.stem] = {"split": split, "img": p, "lab": lab, "boites": boites}
    return images


def taille_image(p):
    import cv2
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).shape[1]


def similarite(a, b):
    """Part des poissons de a qui ont une boite quasi identique (IoU > 0,5) dans b."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    A = np.c_[a[:, :2] - a[:, 2:] / 2, a[:, :2] + a[:, 2:] / 2]
    B = np.c_[b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2]
    lt = np.maximum(A[:, None, :2], B[None, :, :2])
    rb = np.minimum(A[:, None, 2:], B[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    aa, ab = np.prod(A[:, 2:] - A[:, :2], 1), np.prod(B[:, 2:] - B[:, :2], 1)
    iou = inter / (aa[:, None] + ab[None] - inter + 1e-12)
    return float(np.mean(iou.max(1) > 0.5))


def bilan_fuite(images, nom):
    tr = [k for k, v in images.items() if v["split"] == "train"]
    te = [k for k, v in images.items() if v["split"] == "test"]
    meilleur = np.array([max(similarite(images[t]["boites"], images[k]["boites"]) for k in tr) for t in te])
    voisin = np.mean([any(abs(int(k) - int(t)) <= 2 for k in tr) for t in te])
    print(f"{nom:28s} test : {len(te):3d} images | image d'entrainement a +/-2 numeros : {voisin:5.0%} | "
          f"image d'entrainement avec >= 50 % des poissons a la meme place : {np.mean(meilleur >= 0.5):5.0%}")


def main():
    toutes = {}
    for split in ("train", "val", "test"):
        toutes.update(lire(split))
    completes = {k: v for k, v in toutes.items() if taille_image(v["img"]) >= 600}
    print(f"{len(completes)} images completes conservees, {len(toutes) - len(completes)} recadrages ecartes\n")
    bilan_fuite(completes, "Decoupage aleatoire d'origine")

    # Suites de numeros consecutifs, puis blocs
    numeros = sorted(int(k) for k in completes)
    suites, courante = [], [numeros[0]]
    for a, b in zip(numeros, numeros[1:]):
        if b - a > 1:
            suites.append(courante); courante = []
        courante.append(b)
    suites.append(courante)
    blocs = [s[i:i + TAILLE_BLOC] for s in suites for i in range(0, len(s), TAILLE_BLOC)]
    rng = random.Random(GRAINE)
    rng.shuffle(blocs)
    n_total, affect = len(numeros), {}
    n_test = n_val = 0
    for b in blocs:
        if n_test < PART_TEST * n_total:
            jeu = "test"; n_test += len(b)
        elif n_val < PART_VAL * n_total:
            jeu = "val"; n_val += len(b)
        else:
            jeu = "train"
        for n in b:
            affect[n] = jeu
    # Zone tampon
    hors_train = np.array([n for n, j in affect.items() if j != "train"])
    retires = [n for n, j in affect.items() if j == "train" and np.min(np.abs(hors_train - n)) <= TAMPON]
    for n in retires:
        affect[n] = None

    nouvelles = ecrire(completes, {k: affect[int(k)] for k in completes}, SORTIE)
    bilan_fuite(nouvelles, "Decoupage par blocs")

    # Temoin : memes effectifs, decoupage aleatoire image par image
    effectifs = {j: sum(1 for v in nouvelles.values() if v["split"] == j) for j in ("train", "val", "test")}
    cles = sorted(completes, key=int)
    rng.shuffle(cles)
    aff_temoin, i = {k: None for k in cles}, 0
    for j in ("test", "val", "train"):
        for k in cles[i:i + effectifs[j]]:
            aff_temoin[k] = j
        i += effectifs[j]
    temoin = ecrire(completes, aff_temoin, SORTIE_TEMOIN)
    bilan_fuite(temoin, "Temoin aleatoire, memes effectifs")

    print()
    for nom, jeu_donnees in (("blocs", nouvelles), ("temoin", temoin)):
        resume = ", ".join(f"{j} {sum(1 for v in jeu_donnees.values() if v['split'] == j)} images / "
                           f"{sum(len(v['boites']) for v in jeu_donnees.values() if v['split'] == j)} poissons"
                           for j in ("train", "val", "test"))
        print(f"  {nom:6s} : {resume}")
    print(f"  blocs de {TAILLE_BLOC} images, {len(retires)} images retirees par la zone tampon de {TAMPON} images")
    print(f"\nArchives pour Kaggle :\n  {SORTIE}.zip\n  {SORTIE_TEMOIN}.zip")


def ecrire(completes, affectation, dossier):
    if dossier.exists():
        shutil.rmtree(dossier)
    for split in ("train", "val", "test"):
        (dossier / "images" / split).mkdir(parents=True)
        (dossier / "labels" / split).mkdir(parents=True)
    sortie = {}
    for k, v in completes.items():
        jeu = affectation[k]
        if jeu is None:
            continue
        shutil.copy(v["img"], dossier / "images" / jeu / v["img"].name)
        shutil.copy(v["lab"], dossier / "labels" / jeu / v["lab"].name)
        sortie[k] = {**v, "split": jeu}
    (dossier / "fish-pose.yaml").write_text(
        f"path: {dossier.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "kpt_shape: [3, 3]\nflip_idx: [0, 1, 2]\nnames:\n  0: fish\nkpt_names:\n  0: [head, body, fin]\n")
    shutil.make_archive(str(dossier), "zip", dossier)
    return sortie


if __name__ == "__main__":
    main()
