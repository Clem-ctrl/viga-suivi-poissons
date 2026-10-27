"""
Convertit les annotations LabelMe du dataset FHA-MTT (dossier pose_label) au format YOLO-pose.

Chaque JSON contient l'image (encodee dans "imageData") et, pour chaque poisson :
  - un rectangle (label "0" ou "fish")
  - trois points : "head", "body", "fin" (nageoire caudale), relies au rectangle par group_id
Les points sont rattaches a leur rectangle par group_id quand c'est coherent, sinon par
position (point situe dans la boite). Les anomalies sont comptees et affichees.

Sorties (dans DOSSIER_SORTIE) :
  images/{train,val,test}/*.jpg, labels/{train,val,test}/*.txt, fish-pose.yaml,
  apercu_annotations.jpg (controle visuel), et une archive .zip prete pour Kaggle.
Decoupage 70/20/10 aleatoire (graine fixe), stratifie entre images completes et recadrages.
"""
import base64
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

DOSSIER_LABELS = Path.home() / "Downloads" / "FHA-MTT_data" / "FHA-MTT_data" / "pose_label"
DOSSIER_SORTIE = Path(__file__).resolve().parent / "data" / "fish_pose_yolo"
POINTS = ["head", "body", "fin"]
PARTS = {"train": 0.7, "val": 0.2, "test": 0.1}
GRAINE = 42
MARGE = 0.15  # tolerance (fraction de la boite) pour qu'un point soit "dans" la boite


def dans_boite(pt, boite, marge=MARGE):
    x1, y1, x2, y2 = boite
    mx, my = marge * (x2 - x1), marge * (y2 - y1)
    return x1 - mx <= pt[0] <= x2 + mx and y1 - my <= pt[1] <= y2 + my


def convertir_un(d, stats):
    """Renvoie (lignes YOLO, objets pour l'apercu)."""
    W, H = d["imageWidth"], d["imageHeight"]
    rects = [s for s in d["shapes"] if s["shape_type"] == "rectangle"]
    pts = [s for s in d["shapes"] if s["shape_type"] == "point"]
    stats["labels_boites"].update(s["label"] for s in rects)
    utilises = set()
    lignes, objets = [], []
    for r in rects:
        (xa, ya), (xb, yb) = r["points"][:2]
        x1, x2 = sorted((max(0, xa), min(W, xb)))
        y1, y2 = sorted((max(0, ya), min(H, yb)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            stats["boites_degenerees"] += 1
            continue
        boite = (x1, y1, x2, y2)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        kps = []
        for nom in POINTS:
            cands = [i for i, p in enumerate(pts)
                     if p["label"] == nom and i not in utilises and dans_boite(p["points"][0], boite)]
            meme_groupe = [i for i in cands if pts[i].get("group_id") == r.get("group_id")]
            if meme_groupe:
                choix = meme_groupe[0]
            elif cands:
                choix = min(cands, key=lambda i: (pts[i]["points"][0][0] - cx) ** 2 + (pts[i]["points"][0][1] - cy) ** 2)
                stats["points_rattaches_par_position"] += 1
            else:
                choix = None
                stats["points_manquants"] += 1
            if choix is None:
                kps.append((0.0, 0.0, 0))
            else:
                utilises.add(choix)
                px, py = pts[choix]["points"][0]
                kps.append((min(max(px, 0), W), min(max(py, 0), H), 2))
        champs = [0, cx / W, cy / H, (x2 - x1) / W, (y2 - y1) / H]
        for px, py, v in kps:
            champs += [px / W if v else 0.0, py / H if v else 0.0, v]
        lignes.append(" ".join(f"{c:.6f}" if isinstance(c, float) else str(c) for c in champs))
        objets.append((boite, kps))
    stats["points_non_utilises"] += len(pts) - len(utilises)
    return lignes, objets


def apercu(echantillon, chemin):
    vignettes = []
    for img, objets in echantillon:
        im = img.copy()
        for (x1, y1, x2, y2), kps in objets:
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            for (px, py, v), coul in zip(kps, [(0, 255, 255), (0, 200, 0), (255, 255, 0)]):
                if v:
                    cv2.circle(im, (int(px), int(py)), 4, coul, -1)
        h, w = im.shape[:2]
        s = 400 / max(h, w)
        im = cv2.resize(im, (int(w * s), int(h * s)))
        fond = np.full((400, 400, 3), 255, np.uint8)
        fond[:im.shape[0], :im.shape[1]] = im
        vignettes.append(fond)
    while len(vignettes) % 4:
        vignettes.append(np.full((400, 400, 3), 255, np.uint8))
    lignes = [np.hstack(vignettes[i:i + 4]) for i in range(0, len(vignettes), 4)]
    cv2.imwrite(str(chemin), np.vstack(lignes))


def main():
    fichiers = sorted(DOSSIER_LABELS.glob("*.json"))
    if not fichiers:
        raise SystemExit(f"Aucun JSON trouve dans {DOSSIER_LABELS}")
    if DOSSIER_SORTIE.exists():
        shutil.rmtree(DOSSIER_SORTIE)
    stats = Counter()
    stats["labels_boites"] = Counter()

    # Lecture et typage (image complete vs recadrage) pour stratifier le decoupage
    entrees = []
    for f in fichiers:
        d = json.loads(f.read_text(encoding="utf-8"))
        if not d.get("imageData"):
            stats["sans_image"] += 1
            continue
        type_img = "complete" if d["imageWidth"] >= 600 else "recadrage"
        entrees.append((f, type_img))

    rng = random.Random(GRAINE)
    affectation = {}
    for type_img in ("complete", "recadrage"):
        groupe = [f for f, t in entrees if t == type_img]
        rng.shuffle(groupe)
        n = len(groupe)
        n_tr, n_va = round(n * PARTS["train"]), round(n * PARTS["val"])
        for i, f in enumerate(groupe):
            affectation[f] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"

    for part in PARTS:
        (DOSSIER_SORTIE / "images" / part).mkdir(parents=True, exist_ok=True)
        (DOSSIER_SORTIE / "labels" / part).mkdir(parents=True, exist_ok=True)

    comptes = Counter()
    echantillon = []
    # 4 images completes + 8 recadrages tires au hasard pour l'apercu
    a_montrer = set()
    for type_img, k in (("complete", 4), ("recadrage", 8)):
        groupe = [f for f, t in entrees if t == type_img]
        a_montrer.update(rng.sample(groupe, min(k, len(groupe))))
    for f, type_img in entrees:
        d = json.loads(f.read_text(encoding="utf-8"))
        part = affectation[f]
        octets = base64.b64decode(d["imageData"])
        img = cv2.imdecode(np.frombuffer(octets, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            stats["image_illisible"] += 1
            continue
        if img.shape[1] != d["imageWidth"] or img.shape[0] != d["imageHeight"]:
            stats["taille_incoherente"] += 1
            d["imageWidth"], d["imageHeight"] = img.shape[1], img.shape[0]
        lignes, objets = convertir_un(d, stats)
        (DOSSIER_SORTIE / "images" / part / f"{f.stem}.jpg").write_bytes(octets)
        (DOSSIER_SORTIE / "labels" / part / f"{f.stem}.txt").write_text("\n".join(lignes) + "\n")
        comptes[(part, type_img)] += 1
        comptes[(part, "poissons")] += len(lignes)
        if f in a_montrer:
            echantillon.append((img, objets))

    (DOSSIER_SORTIE / "fish-pose.yaml").write_text(
        f"path: {DOSSIER_SORTIE.as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "kpt_shape: [3, 3]\nflip_idx: [0, 1, 2]\n"
        "names:\n  0: fish\n"
        "kpt_names:\n  0: [head, body, fin]\n"
    )
    apercu(echantillon, DOSSIER_SORTIE / "apercu_annotations.jpg")
    archive = shutil.make_archive(str(DOSSIER_SORTIE), "zip", DOSSIER_SORTIE)

    print(f"Fichiers JSON lus : {len(fichiers)}")
    for part in PARTS:
        print(f"  {part:5s} : {comptes[(part, 'complete')]:4d} images completes, "
              f"{comptes[(part, 'recadrage')]:4d} recadrages, {comptes[(part, 'poissons')]:5d} poissons")
    print("Labels des boites :", dict(stats["labels_boites"]))
    for cle in ["points_rattaches_par_position", "points_manquants", "points_non_utilises",
                "boites_degenerees", "sans_image", "image_illisible", "taille_incoherente"]:
        print(f"  {cle} : {stats[cle]}")
    print(f"Dataset : {DOSSIER_SORTIE}")
    print(f"Archive pour Kaggle : {archive}")
    print("Controle visuel : apercu_annotations.jpg (rouge = boite, jaune = tete, vert = corps, cyan = queue)")


if __name__ == "__main__":
    main()
