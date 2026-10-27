"""
Verification de N0 (Q2) : l'activite globale calculee a partir des trajectoires suit-elle la mesure
qu'utiliserait un systeme classique, l'activite des pixels ?

Activite des pixels (type ZebraLab, mode "quantification") : part des pixels de l'image qui changent d'une image
a la suivante (difference de niveaux de gris > SEUIL_GRIS apres lissage), image par image.
Activite des trajectoires (N0 de la Q2) : somme des vitesses du point "corps" de tous les poissons detectes.
Les deux sont moyennees par seconde, puis comparees :
  - dans chaque extrait (correlation seconde par seconde) ;
  - d'un extrait a l'autre (moyenne de l'extrait).
Une bonne concordance justifie d'approcher N0 par les trajectoires dans les bassins virtuels de la Q2.
L'activite des pixels voit aussi ce qui n'est pas un poisson (bulles de l'aerateur, reflets) : c'est une
faiblesse propre a la mesure classique, a garder en tete en lisant la concordance.

Entrees : data/fenetres/T*.mp4 (extraits recadres) et data/trajectoires/T*.csv.gz
Sorties (results/n0_pixels/) : activite_par_seconde.csv, concordance_par_extrait.csv, fig_n0_pixels.png
Duree : environ 10 a 20 s par extrait sur un PC de bureau (decodage video), soit 10 a 15 min pour les 45.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent
DOSSIER_VIDEOS = RACINE / "data" / "fenetres"
DOSSIER_TRAJ = RACINE / "data" / "trajectoires"
INDEX = DOSSIER_VIDEOS / "index_fenetres.csv"
SORTIE = RACINE / "results" / "n0_pixels"
FPS = 25
REDUCTION = 2           # image reduite de moitie avant la difference (plus rapide, moins de bruit)
SEUIL_GRIS = 12         # changement de niveau de gris compte comme mouvement
ENCRE, ENCRE_2, GRILLE, FOND = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def activite_pixels(video):
    import cv2
    cap = cv2.VideoCapture(str(video))
    avant, valeurs = None, []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (g.shape[1] // REDUCTION, g.shape[0] // REDUCTION), interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(g, (5, 5), 0).astype(np.int16)
        valeurs.append(np.nan if avant is None else float(np.mean(np.abs(g - avant) > SEUIL_GRIS)))
        avant = g
    cap.release()
    return np.array(valeurs)


def activite_trajectoires(fichier, n_images):
    d = pd.read_csv(fichier, usecols=["id", "image", "corps_x", "corps_y"]).sort_values(["id", "image"])
    pas = d.groupby("id")["image"].diff()
    v = np.hypot(d.groupby("id")["corps_x"].diff(), d.groupby("id")["corps_y"].diff())
    v[pas != 1] = np.nan                                  # vitesse seulement entre images consecutives
    somme = pd.Series(v.to_numpy(), index=d["image"].to_numpy()).groupby(level=0).sum(min_count=1)
    return somme.reindex(range(n_images)).to_numpy()      # pixels par image, tous poissons confondus


def style_axe(ax):
    ax.set_facecolor(FOND)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRILLE)
    ax.tick_params(colors=ENCRE_2, labelsize=8)
    ax.grid(True, color=GRILLE, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    SORTIE.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(INDEX).set_index("fenetre")
    videos = sorted(DOSSIER_VIDEOS.glob("T*.mp4"))
    if not videos:
        raise SystemExit(f"Aucun extrait .mp4 dans {DOSSIER_VIDEOS}")
    secondes, lignes = [], []
    for video in videos:
        fen = video.stem
        traj = DOSSIER_TRAJ / f"{fen}.csv.gz"
        if not traj.exists():
            print(f"[{fen}] pas de trajectoires, ignore")
            continue
        px = activite_pixels(video)
        tr = activite_trajectoires(traj, len(px))
        n_s = len(px) // FPS
        par_s = pd.DataFrame({"fenetre": fen, "minutes": index.loc[fen, "minutes_depuis_debut"], "seconde": np.arange(n_s),
                              "activite_pixels": np.nanmean(px[:n_s * FPS].reshape(n_s, FPS), axis=1),
                              "activite_trajectoires": np.nanmean(tr[:n_s * FPS].reshape(n_s, FPS), axis=1)})
        secondes.append(par_s)
        ok = par_s[["activite_pixels", "activite_trajectoires"]].notna().all(axis=1)
        r = np.corrcoef(par_s.loc[ok, "activite_pixels"], par_s.loc[ok, "activite_trajectoires"])[0, 1]
        lignes.append({"fenetre": fen, "minutes": index.loc[fen, "minutes_depuis_debut"],
                       "activite_pixels_moy": par_s["activite_pixels"].mean(),
                       "activite_trajectoires_moy": par_s["activite_trajectoires"].mean(),
                       "correlation_par_seconde": r})
        print(f"[{fen}] correlation seconde par seconde : {r:.2f}")
    secondes = pd.concat(secondes, ignore_index=True)
    ext = pd.DataFrame(lignes)
    secondes.to_csv(SORTIE / "activite_par_seconde.csv", index=False)
    ext.to_csv(SORTIE / "concordance_par_extrait.csv", index=False)

    r_ext = np.corrcoef(ext["activite_pixels_moy"], ext["activite_trajectoires_moy"])[0, 1] if len(ext) > 2 else np.nan
    print(f"\nCorrelation seconde par seconde, dans un extrait : mediane {ext['correlation_par_seconde'].median():.2f} "
          f"(de {ext['correlation_par_seconde'].min():.2f} a {ext['correlation_par_seconde'].max():.2f})")
    print(f"Correlation des moyennes d'un extrait a l'autre ({len(ext)} extraits) : {r_ext:.2f}")

    plt.rcParams.update({"font.size": 9, "text.color": ENCRE, "axes.labelcolor": ENCRE_2,
                         "figure.facecolor": FOND, "savefig.facecolor": FOND})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    exemple = ext.sort_values("correlation_par_seconde").iloc[len(ext) // 2]["fenetre"]
    s = secondes[secondes["fenetre"] == exemple]
    z = lambda v: (v - v.mean()) / v.std()
    ax = axes[0]
    style_axe(ax)
    ax.plot(s["seconde"], z(s["activite_pixels"]), color="#8a8984", linewidth=1.5, label="pixels")
    ax.plot(s["seconde"], z(s["activite_trajectoires"]), color="#6250d6", linewidth=1.5, label="trajectoires")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("Seconde de l'extrait")
    ax.set_ylabel("Activite centree reduite")
    ax.set_title(f"Exemple : extrait {exemple}", loc="left", fontsize=9)
    ax = axes[1]
    style_axe(ax)
    ax.plot(ext["minutes"] / 60, ext["correlation_par_seconde"], "o", color="#6250d6", markersize=5,
            markeredgecolor=FOND)
    ax.set_ylim(min(0, ext["correlation_par_seconde"].min() - 0.05), 1)
    ax.set_xlabel("Heures depuis le debut de l'enregistrement")
    ax.set_ylabel("Correlation (par seconde)")
    ax.set_title("Concordance dans chaque extrait", loc="left", fontsize=9)
    ax = axes[2]
    style_axe(ax)
    ax.plot(ext["activite_trajectoires_moy"], ext["activite_pixels_moy"], "o", color="#6250d6", markersize=5,
            markeredgecolor=FOND)
    ax.set_xlabel("Trajectoires : somme des vitesses (px / image)")
    ax.set_ylabel("Pixels qui changent (part de l'image)")
    ax.set_title(f"D'un extrait a l'autre (r = {r_ext:.2f})", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(SORTIE / "fig_n0_pixels.png", dpi=150)
    plt.close(fig)
    print(f"Resultats dans {SORTIE}")


if __name__ == "__main__":
    main()
