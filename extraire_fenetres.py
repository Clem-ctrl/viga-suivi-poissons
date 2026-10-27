"""
Extrait des fenetres de video regulierement espacees, recadrees sur le bassin, pour le suivi.

Plan d'echantillonnage : une fenetre de DUREE_S secondes toutes les PAS_MIN minutes,
dans chacune des 5 videos (90 min chacune, enregistrees a la suite apres l'injection).
Chaque fenetre est recadree sur l'interieur du bassin (comme les images d'entrainement)
et reencodee en H.264, a 25 images/s.

Sorties dans data/fenetres/ :
  T000.mp4, T010.mp4, ...       nom = minutes ecoulees depuis le debut de l'enregistrement
  T000_images.csv, ...          horodatage reel de chaque image de la fenetre
  index_fenetres.csv            une ligne par fenetre (video source, image de debut, heure...)
  apercu_cadrage.jpg            premiere image de chaque video avec le cadre du bassin (controle)
  fenetres.zip                  tout le dossier, pret a envoyer sur Kaggle
"""
import csv
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

DOSSIER_VIDEOS = Path.home() / "Downloads" / "FHA-MTT_data" / "FHA-MTT_data" / "video_data"
DOSSIER_SORTIE = Path(__file__).resolve().parent / "data" / "fenetres"
ROI = (860, 420, 1200, 800)   # x, y, largeur, hauteur de l'interieur du bassin (image 2880 x 1620)
FPS = 25
DUREE_S = 180                 # duree de chaque fenetre
PAS_MIN = 10                  # une fenetre toutes les PAS_MIN minutes
CRF = 20                      # qualite H.264 (plus petit = meilleure qualite)


def trouver_ffmpeg():
    chemin = shutil.which("ffmpeg")
    if chemin:
        return chemin
    racine = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet")
    for dossier, _, fichiers in os.walk(racine):
        if "ffmpeg.exe" in fichiers:
            return os.path.join(dossier, "ffmpeg.exe")
    raise SystemExit("FFmpeg introuvable. Installe-le (winget install -e --id Gyan.FFmpeg).")


def lire_horodatages(csv_images):
    with open(csv_images, newline="") as f:
        return [(int(r["image"]), int(r["horodatage_ms"]), r["heure_chine"]) for r in csv.DictReader(f)]


def apercu_cadrage(ffmpeg, videos, chemin):
    import cv2
    import numpy as np
    vignettes = []
    for mp4 in videos:
        tmp = DOSSIER_SORTIE / "_tmp.png"
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp4),
                        "-frames:v", "1", str(tmp)], check=True)
        im = cv2.imread(str(tmp))
        tmp.unlink()
        x, y, w, h = ROI
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 6)
        cv2.putText(im, mp4.stem, (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 6)
        vignettes.append(cv2.resize(im, (960, 540)))
    while len(vignettes) % 2:
        vignettes.append(np.full((540, 960, 3), 255, np.uint8))
    lignes = [np.hstack(vignettes[i:i + 2]) for i in range(0, len(vignettes), 2)]
    cv2.imwrite(str(chemin), np.vstack(lignes))


def main():
    ffmpeg = trouver_ffmpeg()
    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)
    videos = sorted(DOSSIER_VIDEOS.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"Aucun .mp4 dans {DOSSIER_VIDEOS} : lance d'abord convertir_videos.py")
    apercu_cadrage(ffmpeg, videos, DOSSIER_SORTIE / "apercu_cadrage.jpg")

    t0 = None  # horodatage de la toute premiere image de l'enregistrement
    index = []
    n_par_fenetre = DUREE_S * FPS
    for mp4 in videos:
        horo = lire_horodatages(mp4.with_name(mp4.stem + "_images.csv"))
        if t0 is None:
            t0 = horo[0][1]
        n_images = len(horo)
        debut = 0
        while debut + n_par_fenetre <= n_images:
            minutes = round((horo[debut][1] - t0) / 60000)
            nom = f"T{minutes:03d}"
            sortie = DOSSIER_SORTIE / f"{nom}.mp4"
            if not sortie.exists():
                print(f"[{nom}] {mp4.stem}, images {debut} a {debut + n_par_fenetre - 1}")
                x, y, w, h = ROI
                subprocess.run([
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", f"{debut / FPS:.3f}", "-i", str(mp4),
                    "-frames:v", str(n_par_fenetre), "-vf", f"crop={w}:{h}:{x}:{y}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", str(CRF),
                    "-pix_fmt", "yuv420p", "-an", str(sortie),
                ], check=True)
            else:
                print(f"[{nom}] deja fait")
            with open(DOSSIER_SORTIE / f"{nom}_images.csv", "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["image_fenetre", "image_source", "horodatage_ms", "heure_chine"])
                for k in range(n_par_fenetre):
                    i, ms, heure = horo[debut + k]
                    wr.writerow([k, i, ms, heure])
            index.append({
                "fenetre": nom, "video_source": mp4.stem, "image_debut": debut, "n_images": n_par_fenetre,
                "horodatage_debut_ms": horo[debut][1], "heure_chine_debut": horo[debut][2],
                "minutes_depuis_debut": round((horo[debut][1] - t0) / 60000, 2),
            })
            debut += int(PAS_MIN * 60 * FPS)

    with open(DOSSIER_SORTIE / "index_fenetres.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(index[0]))
        wr.writeheader()
        wr.writerows(index)

    archive = DOSSIER_SORTIE.parent / "fenetres.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
        for p in sorted(DOSSIER_SORTIE.iterdir()):
            z.write(p, p.name)
    taille = sum(p.stat().st_size for p in DOSSIER_SORTIE.glob("*.mp4")) / 1e6
    print(f"\n{len(index)} fenetres de {DUREE_S} s, {taille:.0f} Mo de video")
    print(f"Controle du cadrage : {DOSSIER_SORTIE / 'apercu_cadrage.jpg'}")
    print(f"Archive pour Kaggle : {archive}")


if __name__ == "__main__":
    main()
