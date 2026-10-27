"""
Q1 : quels indicateurs comportementaux separent les poissons sains des malades,
et combien de temps apres l'injection ?

Entrees :
  data/trajectoires/T*.csv.gz        sorties du notebook 02_suivi (Kaggle)
  data/fenetres/index_fenetres.csv   minutes ecoulees depuis le debut de l'enregistrement

Unite statistique. Il y a 20 poissons (10 sains marques, 10 malades) dans un seul bassin. Le suivi
conserve l'identite d'un poisson quelques dizaines de secondes a quelques minutes, pas d'un extrait
a l'autre : plusieurs pistes peuvent appartenir au meme poisson, et cette appartenance est inconnue.
Les pistes ne sont donc pas des repetitions independantes. D'ou trois niveaux d'analyse :

  1. ANALYSE PRINCIPALE, au niveau de l'extrait (45 extraits) : pour chaque groupe, moyenne de
     l'indicateur ponderee par la duree de presence (un poisson coupe en 5 pistes pese autant qu'un
     poisson suivi d'un bloc), puis ecart malades - sains en ecarts-types des sains. Sains et malades
     partagent le bassin au meme moment : chaque extrait est une comparaison appariee. Sur cette
     courbe, regression segmentee a trois phases (ecart stable, transition, nouveau plateau) dont le
     debut de transition est le debut de la divergence ; IC 95 % et p de la divergence par bootstrap
     par blocs (erreurs autocorrelees : memes poissons d'un extrait a l'autre). Conclusions valables
     pour ces 20 poissons dans ce bassin.
  2. SENSIBILITE : la meme analyse restreinte aux pistes longues (>= PISTE_LONGUE_S), plus souvent
     un poisson chacune ; et un modele mixte sur tranches de 10 s,
     indicateur ~ groupe * temps + (1 | extrait) + (1 | extrait:piste), dont les p sont optimistes.
  3. DESCRIPTIF : tailles d'effet par extrait avec la piste comme unite.

Sorties (results/q1/) :
  qualite_suivi.csv, indicateurs_par_piste.csv
  sous_pistes.csv                pistes coupees aux echanges d'identite detectes par la marque, et leur groupe
  ecart_initial.csv              ecart malades - sains sur les 30 premieres minutes [IC 95 %]
  ecart_par_extrait.csv          moyennes ponderees par groupe et ecart standardise, par extrait
  divergence_segmentee.csv       debut de divergence [IC 95 %], fin de transition, amplitude, p (toutes pistes / pistes longues)
  modeles_mixtes_10s.txt / .csv  modele mixte de sensibilite
  effets_pistes_descriptif.csv   Hedges g par extrait (piste = unite)
  fig1_marque.png, fig2_effectifs.png, fig3_indicateurs.png, fig4_divergence.png (principale), fig5_effets_pistes.png
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.signal import savgol_filter, welch
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning

RACINE = Path(__file__).resolve().parent
DOSSIER_TRAJ = RACINE / "data" / "trajectoires"
INDEX = RACINE / "data" / "fenetres" / "index_fenetres.csv"
SORTIE = RACINE / "results" / "q1"

FPS = 25
LARGEUR, HAUTEUR = 1200, 800      # taille des extraits recadres (pixels)
SEUIL_MARQUE = None               # None = seuil automatique (Otsu) sur toutes les mesures de marque
BANDE_MARQUE = 10                 # zone d'incertitude autour du seuil
LISSAGE_MARQUE_S = 5              # mediane glissante de la mesure de marque le long d'une piste
PERSISTANCE_S = 3                 # un changement marque <-> sans marque doit durer au moins 3 s pour compter
MARGE_ECHANGE_S = 2               # images retirees de part et d'autre d'un echange d'identite detecte
MIN_DETECTIONS_MARQUE = 25        # mesures de marque valides necessaires pour etiqueter une piste
TROU_MAX = 12                     # trous <= 0,5 s interpoles ; au-dela, la piste est coupee
DUREE_MIN_S = 10                  # segments plus courts ignores
SEUIL_IMMOBILE = 0.2              # longueurs de corps par seconde
N_BOOT = 2000                     # bootstrap des tailles d'effet descriptives
TRANCHE_S = 10                    # tranches du modele mixte de sensibilite
PISTE_LONGUE_S = 60               # analyse de sensibilite sur les pistes longues
DUREE_MIN_GROUPE_S = 60           # presence cumulee minimale d'un groupe dans un extrait pour calculer l'ecart
MIN_POISSONS_GROUPE = 2           # poissons visibles en moyenne requis dans chaque groupe (sinon l'extrait est ecarte :
                                  # en fin d'essai il ne reste qu'un ou deux malades, les survivants)
FENETRES_DEBUT = 3                # extraits utilises pour mesurer l'ecart initial (0 a 30 min)
N_BOOT_SEG = 1000                 # bootstrap de la regression segmentee
LONGUEUR_BLOC = 3                 # extraits par bloc (30 min) pour le bootstrap par blocs
PAS_GRILLE_H = 1 / 6              # pas de la grille de recherche des points de rupture (10 min, l'espacement des extraits)
GRAINE = 42

COULEURS = {"sain": "#2a78d6", "malade": "#eb6834"}   # palette categorielle de reference (slots 1-2)
ENCRE, ENCRE_2, GRILLE, FOND = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

INDICATEURS = {
    "vitesse_moy": ("Vitesse moyenne", "longueurs de corps / s", True),
    "vitesse_cv": ("Variabilite de la vitesse (CV)", "sans unite", False),
    "acceleration_moy": ("Acceleration moyenne", "longueurs de corps / s²", True),
    "angle_virage": ("Vitesse angulaire", "degres / s", True),
    "courbure_moy": ("Courbure du corps", "sans unite", False),
    "part_immobile": ("Part du temps immobile", "proportion", False),
    "part_rapide": ("Part du temps en nage rapide", "proportion", False),
    "dist_paroi": ("Distance a la paroi", "longueurs de corps", True),
    "dist_voisin": ("Distance au plus proche voisin", "longueurs de corps", True),
}
# La frequence de flexion du corps (analyse spectrale de la position du point "corps") a ete essayee puis
# retiree : son pic tombe presque toujours sur la borne basse de la recherche (0,5 Hz), signe que les
# trois points cles a 25 images/s ne suffisent pas a resoudre le battement de la queue.


# ---------------------------------------------------------------- utilitaires
def otsu(valeurs):
    v = np.asarray(valeurs, float)
    v = v[np.isfinite(v)]
    seuils = np.arange(np.percentile(v, 2), np.percentile(v, 98), 1.0)
    meilleur, score = 0.0, -1
    for s in seuils:
        a, b = v[v <= s], v[v > s]
        if len(a) < 5 or len(b) < 5:
            continue
        sc = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if sc > score:
            meilleur, score = s, sc
    # milieu de l'intervalle vide entre les deux classes (le critere est plat dans cet intervalle)
    return (v[v <= meilleur].max() + v[v > meilleur].min()) / 2


def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if not np.isfinite(sp) or sp <= 1e-9 * (abs(np.mean(a)) + abs(np.mean(b)) + 1e-12):
        return np.nan
    return (np.mean(a) - np.mean(b)) / sp * (1 - 3 / (4 * (na + nb) - 9))


def bootstrap_g(a, b, rng):
    ia = rng.integers(0, len(a), (N_BOOT, len(a)))
    ib = rng.integers(0, len(b), (N_BOOT, len(b)))
    A, B = a[ia], b[ib]
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * A.var(1, ddof=1) + (nb - 1) * B.var(1, ddof=1)) / (na + nb - 2))
    echelle = np.abs(A.mean(1)) + np.abs(B.mean(1)) + 1e-12
    sp = np.where(sp > 1e-9 * echelle, sp, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (A.mean(1) - B.mean(1)) / sp * (1 - 3 / (4 * (na + nb) - 9))
    g = g[np.isfinite(g)]
    return np.percentile(g, [2.5, 97.5]) if len(g) > 100 else (np.nan, np.nan)


def style_axe(ax):
    ax.set_facecolor(FOND)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRILLE)
    ax.tick_params(colors=ENCRE_2, labelsize=8)
    ax.grid(True, color=GRILLE, linewidth=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- 1. etat de sante, et coupure des pistes aux echanges d'identite
def otsu_histogramme(v):
    """Seuil d'Otsu calcule sur un histogramme (rapide pour des millions de mesures)."""
    v = np.clip(v[np.isfinite(v)], -150, 150)
    h, bords = np.histogram(v, bins=300, range=(-150, 150))
    centres = (bords[:-1] + bords[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * centres) / np.maximum(w0, 1)
    m1 = (np.sum(h * centres) - np.cumsum(h * centres)) / np.maximum(w1, 1)
    return bords[1:][np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def etats_marque(contraste, seuil):
    """+1 = marque nette, -1 = absence nette, 0 = incertain, apres mediane glissante de LISSAGE_MARQUE_S."""
    n = LISSAGE_MARQUE_S * FPS
    lisse = contraste.rolling(n, min_periods=n // 3, center=True).median().to_numpy()
    return np.where(lisse > seuil + BANDE_MARQUE, 1, np.where(lisse < seuil - BANDE_MARQUE, -1, 0))


def plages(etats, images):
    """Plages consecutives de meme etat (hors etats incertains). Les plages plus courtes que
    PERSISTANCE_S sont fusionnees dans leur voisine : un scintillement n'est pas un echange."""
    m = etats != 0
    e, im = etats[m], images[m]
    if len(e) == 0:
        return []
    b = np.r_[0, np.where(np.diff(e) != 0)[0] + 1, len(e)]
    R = [[e[x], im[x], im[y - 1], y - x] for x, y in zip(b[:-1], b[1:])]
    while len(R) > 1:
        k = min(range(len(R)), key=lambda j: R[j][3])
        if R[k][3] >= PERSISTANCE_S * FPS:
            break
        j = 1 if k == 0 else k - 1 if k == len(R) - 1 else (k - 1 if R[k - 1][3] >= R[k + 1][3] else k + 1)
        x, y = sorted((k, j))
        R[x] = [R[j][0], R[x][1], R[y][2], R[x][3] + R[y][3]]
        del R[y]
        fusion = [R[0]]
        for r in R[1:]:
            if r[0] == fusion[-1][0]:
                fusion[-1] = [r[0], fusion[-1][1], r[2], fusion[-1][3] + r[3]]
            else:
                fusion.append(r)
        R = fusion
    return R


def decouper_par_marque(fichiers):
    """Attribue un etat de sante a chaque piste a partir de la marque blanche. Quand la marque change
    durablement le long d'une piste (marquee puis non marquee, ou l'inverse), le tracker a change de
    poisson : la piste est coupee en sous-pistes, et les images proches de l'echange sont retirees."""
    toutes = pd.concat(pd.read_csv(f, usecols=["fenetre", "id", "image", "marque_contraste"]) for f in fichiers)
    seuil = SEUIL_MARQUE if SEUIL_MARQUE is not None else otsu_histogramme(toutes["marque_contraste"].to_numpy())
    marge = MARGE_ECHANGE_S * FPS
    lignes = []
    for (fen, i), g in toutes.sort_values(["fenetre", "id", "image"]).groupby(["fenetre", "id"]):
        img = g["image"].to_numpy()
        c = g["marque_contraste"].reset_index(drop=True)
        base = {"fenetre": fen, "id": i, "n_detections_piste": len(g)}
        R = plages(etats_marque(c, seuil), img) if c.notna().sum() >= MIN_DETECTIONS_MARQUE else []
        if not R:
            lignes.append({**base, "sous_id": f"{i}_0", "debut": img[0], "fin": img[-1], "groupe": "indetermine",
                           "n_echanges_piste": 0, "contraste_median": c.median()})
            continue
        for k, (signe, a, b, _) in enumerate(R):
            debut = img[0] if k == 0 else a + marge
            fin = img[-1] if k == len(R) - 1 else b - marge
            if fin - debut + 1 < FPS:
                continue
            dans = (img >= debut) & (img <= fin)
            lignes.append({**base, "sous_id": f"{i}_{k}", "debut": debut, "fin": fin,
                           "groupe": "sain" if signe > 0 else "malade", "n_echanges_piste": len(R) - 1,
                           "contraste_median": c[dans].median(), "n_marque": int(c[dans].notna().sum())})
    sous = pd.DataFrame(lignes)
    sous["n_marque"] = sous["n_marque"].fillna(0)
    return sous, seuil


def appliquer_sous_pistes(d, sous):
    """Ajoute a chaque detection sa sous-piste et son groupe ('exclu' : pres d'un echange d'identite)."""
    s = sous[sous["fenetre"] == d["fenetre"].iloc[0]][["id", "sous_id", "debut", "fin", "groupe"]]
    d = pd.merge_asof(d.sort_values("image"), s.sort_values("debut"), left_on="image", right_on="debut",
                      by="id", direction="backward")
    hors = d["sous_id"].isna() | (d["image"] > d["fin"])
    d.loc[hors, ["sous_id", "groupe"]] = [np.nan, "exclu"]
    return d.drop(columns=["debut", "fin"])


# ---------------------------------------------------------------- 2-3. trajectoires et indicateurs
def segments(g):
    """Decoupe une piste en segments continus (trous <= TROU_MAX interpoles)."""
    g = g.sort_values("image")
    img = g["image"].to_numpy()
    coupures = np.where(np.diff(img) > TROU_MAX + 1)[0] + 1
    for bloc in np.split(np.arange(len(g)), coupures):
        s = g.iloc[bloc]
        complet = np.arange(s["image"].iloc[0], s["image"].iloc[-1] + 1)
        if len(complet) < DUREE_MIN_S * FPS:
            continue
        s = s.set_index("image").reindex(complet)
        cols = ["tete_x", "tete_y", "corps_x", "corps_y", "queue_x", "queue_y", "horodatage_ms", "dist_voisin_px"]
        s[cols] = s[cols].interpolate(limit_direction="both")
        yield s


def indicateurs_segment(s, L):
    def lisse(v):
        return savgol_filter(v, 9, 2) if len(v) >= 9 else v
    h = np.c_[lisse(s["tete_x"].to_numpy()), lisse(s["tete_y"].to_numpy())]
    b = np.c_[lisse(s["corps_x"].to_numpy()), lisse(s["corps_y"].to_numpy())]
    q = np.c_[lisse(s["queue_x"].to_numpy()), lisse(s["queue_y"].to_numpy())]
    t = s["horodatage_ms"].to_numpy() / 1000.0
    dt = np.clip(np.diff(t), 0.02, 0.08)
    v = np.linalg.norm(np.diff(b, axis=0), axis=1) / dt / L                 # corps, longueurs de corps / s
    vh = np.diff(h, axis=0) / dt[:, None] / L                               # tete, comme l'article
    acc = np.linalg.norm(np.diff(vh, axis=0), axis=1) / dt[1:]
    theta = np.degrees(np.arctan2(h[:, 1] - b[:, 1], h[:, 0] - b[:, 0]))
    dtheta = (np.diff(theta) + 180) % 360 - 180
    a_hb, a_bq, a_qh = (np.linalg.norm(h - b, axis=1), np.linalg.norm(b - q, axis=1), np.linalg.norm(q - h, axis=1))
    aire = 0.5 * np.abs((b[:, 0] - h[:, 0]) * (q[:, 1] - h[:, 1]) - (q[:, 0] - h[:, 0]) * (b[:, 1] - h[:, 1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        courbure = 4 * aire / (a_hb * a_bq * a_qh) * L                       # courbure de Menger, sans unite
        flexion = ((q[:, 0] - h[:, 0]) * (b[:, 1] - h[:, 1]) - (q[:, 1] - h[:, 1]) * (b[:, 0] - h[:, 0])) / a_qh
    paroi = np.minimum.reduce([b[:, 0], LARGEUR - b[:, 0], b[:, 1], HAUTEUR - b[:, 1]]) / L
    battement = np.nan
    f = np.nan_to_num(flexion - np.nanmean(flexion))
    if len(f) >= 10 * FPS and np.nanmean(v) > SEUIL_IMMOBILE:
        fr, p = welch(f, fs=FPS, nperseg=min(len(f), 4 * FPS))
        m = (fr >= 0.5) & (fr <= 8)
        battement = fr[m][np.argmax(p[m])]
    return {
        "duree_s": len(s) / FPS,
        "vitesse_moy": np.mean(v), "vitesse_cv": np.std(v) / np.mean(v) if np.mean(v) > 0 else np.nan,
        "acceleration_moy": np.mean(acc), "angle_virage": np.mean(np.abs(dtheta)) / np.mean(dt),
        "courbure_moy": np.nanmean(courbure), "part_immobile": np.mean(v < SEUIL_IMMOBILE),
        "dist_paroi": np.mean(paroi), "dist_voisin": np.nanmean(s["dist_voisin_px"].to_numpy()) / L,
        "battement_hz": battement, "_v": v.astype(np.float32),
    }


def distances_voisins(d):
    """Distance au plus proche voisin (tous poissons detectes) et polarisation par groupe, image par image."""
    d = d.sort_values("image").reset_index(drop=True)
    nnd = np.full(len(d), np.nan)
    pol = []
    xy = d[["corps_x", "corps_y"]].to_numpy()
    cap = d[["tete_x", "tete_y"]].to_numpy() - xy
    cap = cap / np.maximum(np.linalg.norm(cap, axis=1, keepdims=True), 1e-6)
    grp = d["groupe"].to_numpy()
    bornes = np.r_[0, np.where(np.diff(d["image"].to_numpy()))[0] + 1, len(d)]
    for a, z in zip(bornes[:-1], bornes[1:]):
        p = xy[a:z]
        if len(p) > 1:
            dist = np.linalg.norm(p[:, None] - p[None], axis=2)
            np.fill_diagonal(dist, np.inf)
            nnd[a:z] = dist.min(1)
        for g in ("sain", "malade"):
            m = grp[a:z] == g
            if m.sum() >= 3:
                pol.append((g, np.linalg.norm(cap[a:z][m].mean(0)), m.sum()))
            elif m.sum() > 0:
                pol.append((g, np.nan, m.sum()))
    d["dist_voisin_px"] = nnd
    pol = pd.DataFrame(pol, columns=["groupe", "polarisation", "n"])
    return d, pol




# ---------------------------------------------------------------- 4. analyse principale : contraste par extrait
def contrastes_par_extrait(seg, ref_sd, duree_min_groupe, visibles=None):
    """Par extrait et indicateur : moyenne de chaque groupe ponderee par la duree de presence.
    Un poisson coupe en plusieurs pistes pese ainsi autant qu'un poisson suivi d'un seul tenant."""
    lignes = []
    for (fen, minutes), g in seg.groupby(["fenetre", "minutes_depuis_debut"]):
        for c in INDICATEURS:
            ligne = {"indicateur": c, "fenetre": fen, "minutes": minutes}
            for grp in ("sain", "malade"):
                x = g[(g["groupe"] == grp) & g[c].notna()]
                ligne[f"duree_{grp}_s"] = x["duree_s"].sum()
                ligne[f"n_pistes_{grp}"] = x["id"].nunique()
                ligne[f"moyenne_{grp}"] = np.average(x[c], weights=x["duree_s"]) if len(x) else np.nan
            ok = min(ligne["duree_sain_s"], ligne["duree_malade_s"]) >= duree_min_groupe
            if visibles is not None:
                ligne["poissons_visibles_malade"] = visibles.get(fen, {}).get("malade", np.nan)
                ok = ok and visibles.get(fen, {}).get("malade", 0) >= MIN_POISSONS_GROUPE \
                        and visibles.get(fen, {}).get("sain", 0) >= MIN_POISSONS_GROUPE
            ligne["ecart"] = ligne["moyenne_malade"] - ligne["moyenne_sain"] if ok else np.nan
            ligne["ecart_standardise"] = ligne["ecart"] / ref_sd[c] if ref_sd[c] > 0 else np.nan
            lignes.append(ligne)
    return pd.DataFrame(lignes)


def _rampe(t, t1, t2):
    return np.clip((t - t1) / (t2 - t1), 0, 1)


def _sse_rampe(t, Z, couples):
    """Somme des carres residuels du modele z = a + b * rampe(t ; t1, t2), pour chaque couple (t1, t2)
    et chaque ligne de Z. Rampe : 0 avant t1, montee lineaire de t1 a t2, 1 apres t2."""
    sse = np.empty((Z.shape[0], len(couples)))
    for k, (t1, t2) in enumerate(couples):
        X = np.c_[np.ones_like(t), _rampe(t, t1, t2)]
        R = Z - Z @ (X @ np.linalg.pinv(X)).T
        sse[:, k] = (R ** 2).sum(1)
    return sse


def _blocs(residus, n_tirages, longueur, rng):
    n = len(residus)
    n_blocs = int(np.ceil(n / longueur))
    debuts = rng.integers(0, n - longueur + 1, (n_tirages, n_blocs))
    idx = (debuts[:, :, None] + np.arange(longueur)[None, None, :]).reshape(n_tirages, -1)[:, :n]
    return residus[idx]


def regression_segmentee(t, z, rng):
    """Modele a trois phases : ecart stable, transition lineaire entre t1 et t2, nouveau plateau.
    t1 = debut de la divergence (la quantite d'interet pour une detection precoce). Le modele couvre
    aussi une divergence qui se poursuit jusqu'au bout (t2 = fin de l'enregistrement) et un saut brusque.
    t1, t2 estimes par recherche sur grille ; IC 95 % par bootstrap par blocs des residus ; p de la
    divergence par bootstrap sous l'hypothese nulle d'un ecart constant (erreurs autocorrelees :
    memes poissons d'un extrait a l'autre)."""
    ok = np.isfinite(z)
    t, z = t[ok], z[ok]
    res = {"n_extraits": len(t)}
    if len(t) < 10:
        return res
    grille = np.arange(t.min(), t.max() + 1e-9, PAS_GRILLE_H)
    couples = [(t1, t2) for t1 in grille if t1 <= t.max() - 0.5 for t2 in grille if t2 > t1]
    sse1 = _sse_rampe(t, z[None, :], couples)[0]
    k = int(np.argmin(sse1))
    t1, t2 = couples[k]
    X = np.c_[np.ones_like(t), _rampe(t, t1, t2)]
    a, b = np.linalg.lstsq(X, z, rcond=None)[0]
    ajuste = X @ np.array([a, b])
    residus = z - ajuste
    stat = ((z - z.mean()) ** 2).sum() / sse1[k]
    rho = np.corrcoef(residus[:-1], residus[1:])[0, 1]
    Zb = ajuste[None, :] + _blocs(residus, N_BOOT_SEG, LONGUEUR_BLOC, rng)
    kb = np.argmin(_sse_rampe(t, Zb, couples), axis=1)
    t1_b = np.array([couples[j][0] for j in kb])
    Z0 = z.mean() + _blocs(z - z.mean(), N_BOOT_SEG, LONGUEUR_BLOC, rng)
    s0 = ((Z0 - Z0.mean(1, keepdims=True)) ** 2).sum(1) / _sse_rampe(t, Z0, couples).min(1)
    res.update({"debut_divergence_h": t1, "debut_ic_bas_h": np.percentile(t1_b, 2.5),
                "debut_ic_haut_h": np.percentile(t1_b, 97.5), "fin_transition_h": t2,
                "ecart_initial": a, "amplitude": b, "ecart_final": a + b,
                "autocorrelation_residus": rho, "p_divergence": (np.sum(s0 >= stat) + 1) / (N_BOOT_SEG + 1),
                "_t": t, "_ajuste": ajuste})
    return res


def ecart_initial(seg, ref_sd, fenetres_debut, rng, n_boot=2000):
    """Ecart standardise malades - sains sur les premiers extraits (moyennes ponderees par la duree).
    IC 95 % par bootstrap des pistes (optimiste : plusieurs pistes peuvent appartenir au meme poisson)."""
    d = seg[seg["fenetre"].isin(fenetres_debut)]
    lignes = []
    for c in INDICATEURS:
        x = {g: d[(d["groupe"] == g) & d[c].notna()] for g in ("sain", "malade")}
        if min(len(x["sain"]), len(x["malade"])) < 3 or not ref_sd[c] > 0:
            continue
        def ecart(a, b):
            return (np.average(b[c], weights=b["duree_s"]) - np.average(a[c], weights=a["duree_s"])) / ref_sd[c]
        e0 = ecart(x["sain"], x["malade"])
        boot = [ecart(x["sain"].sample(frac=1, replace=True, random_state=int(rng.integers(1e9))),
                      x["malade"].sample(frac=1, replace=True, random_state=int(rng.integers(1e9))))
                for _ in range(n_boot)]
        lignes.append({"indicateur": c, "libelle": INDICATEURS[c][0],
                       "moyenne_sain": np.average(x["sain"][c], weights=x["sain"]["duree_s"]),
                       "moyenne_malade": np.average(x["malade"][c], weights=x["malade"]["duree_s"]),
                       "ecart_initial_et": e0, "ic_bas": np.percentile(boot, 2.5), "ic_haut": np.percentile(boot, 97.5),
                       "n_pistes_sain": len(x["sain"]), "n_pistes_malade": len(x["malade"])})
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------- 5. modele mixte sur tranches de 10 s (sensibilite)
def modeles_mixtes_tranches(tranches):
    lignes = []
    with open(SORTIE / "modeles_mixtes_10s.txt", "w", encoding="utf-8") as fo:
        fo.write("Modele : indicateur ~ groupe * temps_h + (1 | extrait) + (1 | extrait:piste), tranches de 10 s.\n"
                 "Les p-values sont OPTIMISTES : plusieurs pistes peuvent appartenir au meme poisson,\n"
                 "et l'identite des poissons (10 sains, 10 malades) ne peut pas etre modelisee.\n")
        for c in INDICATEURS:
            d = tranches[["fenetre", "piste", "groupe", "temps_h", c]].dropna().copy()
            if INDICATEURS[c][2]:
                d[c] = np.log(d[c].clip(lower=1e-3))
            try:
                m = smf.mixedlm(f"{c} ~ C(groupe, Treatment('sain')) * temps_h", d, groups=d["fenetre"],
                                re_formula="1", vc_formula={"piste": "0 + C(piste)"}).fit(reml=True)
                fo.write(f"\n===== {INDICATEURS[c][0]} ({'log, ' if INDICATEURS[c][2] else ''}{len(d)} tranches) =====\n")
                fo.write(m.summary().as_text() + "\n")
                if not m.converged:
                    fo.write("ATTENTION : l'ajustement n'a pas converge.\n")
                noms = list(m.params.index)
                eff_g = [n for n in noms if n.startswith("C(groupe") and ":" not in n][0]
                eff_i = [n for n in noms if n.startswith("C(groupe") and ":" in n][0]
                lignes.append({"indicateur": c, "libelle": INDICATEURS[c][0], "echelle_log": INDICATEURS[c][2],
                               "n_tranches": len(d), "effet_groupe_t0": m.params[eff_g], "p_groupe_t0": m.pvalues[eff_g],
                               "interaction_par_h": m.params[eff_i], "p_interaction": m.pvalues[eff_i],
                               "converge": m.converged})
            except Exception as ex:
                fo.write(f"\n===== {c} : echec de l'ajustement ({ex}) =====\n")
    pd.DataFrame(lignes).to_csv(SORTIE / "modeles_mixtes_10s.csv", index=False)


# ---------------------------------------------------------------- programme principal
def main():
    for w in (RuntimeWarning, ConvergenceWarning, FutureWarning, DeprecationWarning, UserWarning):
        warnings.filterwarnings("ignore", category=w)
    SORTIE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(GRAINE)
    fichiers = sorted(DOSSIER_TRAJ.glob("T*.csv.gz"))
    if not fichiers:
        raise SystemExit(f"Aucune trajectoire dans {DOSSIER_TRAJ} : decompresse trajectoires.zip dans ce dossier.")
    index = pd.read_csv(INDEX)[["fenetre", "minutes_depuis_debut"]]

    # 1. etat de sante des pistes, coupees aux echanges d'identite
    pistes, seuil = decouper_par_marque(fichiers)
    pistes.to_csv(SORTIE / "sous_pistes.csv", index=False)
    print(f"Seuil de marque : {seuil:.1f} (zone d'incertitude +/- {BANDE_MARQUE})")
    print(pistes["groupe"].value_counts().to_string())
    n_ech = pistes.groupby(["fenetre", "id"])["n_echanges_piste"].first().sum()
    print(f"Echanges d'identite sain <-> malade detectes et coupes : {n_ech}")

    # 2-3. segments, tranches de 10 s, indicateurs
    segs, tranches, v_segs, v_tranches, effectifs, qualite = [], [], [], [], [], []
    n_tranche = TRANCHE_S * FPS
    for f in fichiers:
        d = pd.read_csv(f)
        fen = d["fenetre"].iloc[0]
        d = appliquer_sous_pistes(d, pistes)
        ok = (d[["tete_c", "corps_c", "queue_c"]] > 0.5).all(axis=1)
        L = np.percentile(np.hypot(d.loc[ok, "tete_x"] - d.loc[ok, "queue_x"], d.loc[ok, "tete_y"] - d.loc[ok, "queue_y"]), 75)
        d, _ = distances_voisins(d)
        n_img = d["image"].nunique()
        eff = d.groupby("groupe").size() / max(n_img, 1)
        effectifs.append({"fenetre": fen, "longueur_corps_px": L,
                          **{f"n_{g}": eff.get(g, 0.0) for g in ["sain", "malade", "indetermine", "exclu"]}})
        for (i, grp), g in d.groupby(["sous_id", "groupe"]):
            if grp not in ("sain", "malade"):
                continue
            duree_piste = len(g) / FPS
            for s in segments(g):
                ind = indicateurs_segment(s, L)
                v_segs.append(ind.pop("_v"))
                segs.append({"fenetre": fen, "id": i, "groupe": grp, "duree_piste_s": duree_piste, **ind})
                for k in range(0, len(s) - n_tranche + 1, n_tranche):
                    it = indicateurs_segment(s.iloc[k:k + n_tranche], L)
                    v_tranches.append(it.pop("_v"))
                    tranches.append({"fenetre": fen, "piste": f"{fen}_{i}", "groupe": grp, **it})
        pf = pistes[pistes["fenetre"] == fen]
        pid = pf.groupby("id").first()
        duree = pid["n_detections_piste"] / FPS
        qualite.append({"fenetre": fen, "n_pistes": len(pid), "duree_mediane_s": duree.median(),
                        "part_detections_pistes_30s_plus": pid.loc[duree >= 30, "n_detections_piste"].sum()
                                                           / pid["n_detections_piste"].sum(),
                        "echanges_identite_detectes": int(pid["n_echanges_piste"].sum()),
                        "n_sous_pistes": len(pf),
                        "part_detections_etiquetees": float(d["groupe"].isin(["sain", "malade"]).mean())})
        print(f"[{fen}] longueur de corps {L:.0f} px, {len(pid)} pistes, {int(pid['n_echanges_piste'].sum())} echanges coupes")

    seg = pd.DataFrame(segs).merge(index, on="fenetre", how="left")
    tr = pd.DataFrame(tranches).merge(index, on="fenetre", how="left")
    # Nage rapide : au-dela du 90e percentile de la vitesse instantanee des poissons sains
    seuil_rapide = np.percentile(np.concatenate([v for v, g in zip(v_segs, seg["groupe"]) if g == "sain"]), 90)
    seg["part_rapide"] = [np.mean(v > seuil_rapide) for v in v_segs]
    tr["part_rapide"] = [np.mean(v > seuil_rapide) for v in v_tranches]
    for t in (seg, tr):
        t["temps_h"] = t["minutes_depuis_debut"] / 60

    # Valeurs par piste (descriptif) et qualite
    cols = list(INDICATEURS)
    def moyenne_ponderee(g):
        w = g["duree_s"].to_numpy()
        return pd.Series({c: np.average(g[c].dropna(), weights=w[g[c].notna().to_numpy()])
                          if g[c].notna().any() else np.nan for c in cols} | {"duree_s": w.sum()})
    par_piste = seg.groupby(["fenetre", "minutes_depuis_debut", "temps_h", "id", "groupe"]).apply(moyenne_ponderee).reset_index()
    par_piste.to_csv(SORTIE / "indicateurs_par_piste.csv", index=False)
    pd.DataFrame(qualite).merge(pd.DataFrame(effectifs), on="fenetre").merge(index, on="fenetre") \
        .to_csv(SORTIE / "qualite_suivi.csv", index=False)
    print(f"\nNage rapide : > {seuil_rapide:.2f} longueurs de corps / s ; {len(par_piste)} pistes "
          f"({(par_piste['groupe'] == 'sain').sum()} saines, {(par_piste['groupe'] == 'malade').sum()} malades), "
          f"{len(tr)} tranches de {TRANCHE_S} s")

    # 4. ANALYSE PRINCIPALE : ecart malades - sains par extrait, puis regression segmentee
    #    Echelle de reference : ecart-type des valeurs par piste chez les sains (toutes periodes)
    ref_sd = {c: par_piste.loc[par_piste["groupe"] == "sain", c].std() for c in cols}
    visibles = {e["fenetre"]: {"sain": e["n_sain"], "malade": e["n_malade"]} for e in effectifs}

    # 4a. Ecart des les premieres minutes (0 a 30 min)
    premiers = sorted(seg["fenetre"].unique(), key=lambda f: int(f[1:]))[:FENETRES_DEBUT]
    debut = ecart_initial(seg, ref_sd, premiers, rng)
    debut.to_csv(SORTIE / "ecart_initial.csv", index=False)
    resultats, courbes = [], {}
    for nom, sous in (("toutes les pistes", seg), (f"pistes >= {PISTE_LONGUE_S} s", seg[seg["duree_piste_s"] >= PISTE_LONGUE_S])):
        if sous.empty:
            print(f"Analyse '{nom}' : aucune piste, ignoree")
            continue
        ct = contrastes_par_extrait(sous, ref_sd, DUREE_MIN_GROUPE_S, visibles)
        if nom == "toutes les pistes":
            ct.to_csv(SORTIE / "ecart_par_extrait.csv", index=False)
        for c in cols:
            e = ct[ct["indicateur"] == c].sort_values("minutes")
            r = regression_segmentee(e["minutes"].to_numpy() / 60, e["ecart_standardise"].to_numpy(dtype=float), rng)
            if nom == "toutes les pistes":
                courbes[c] = (e, r)
            resultats.append({"analyse": nom, "indicateur": c, "libelle": INDICATEURS[c][0],
                              **{k: v for k, v in r.items() if not k.startswith("_")}})
    seg_res = pd.DataFrame(resultats)
    if "p_divergence" in seg_res:
        seg_res["divergence_detectee"] = seg_res["p_divergence"] < 0.05
    seg_res.to_csv(SORTIE / "divergence_segmentee.csv", index=False)

    # 5. Sensibilite : modele mixte sur tranches de 10 s
    modeles_mixtes_tranches(tr)

    # 6. Descriptif : tailles d'effet par extrait avec la piste comme unite
    effets = []
    for (fen, minutes), g in par_piste.groupby(["fenetre", "minutes_depuis_debut"]):
        for c in cols:
            a = g.loc[g["groupe"] == "malade", c].dropna().to_numpy()
            b = g.loc[g["groupe"] == "sain", c].dropna().to_numpy()
            if len(a) < 3 or len(b) < 3:
                continue
            lo, hi = bootstrap_g(a, b, rng)
            effets.append({"indicateur": c, "fenetre": fen, "minutes": minutes, "n_pistes_malade": len(a),
                           "n_pistes_sain": len(b), "hedges_g": hedges_g(a, b), "ic_bas": lo, "ic_haut": hi,
                           "p_mann_whitney": mannwhitneyu(a, b).pvalue})
    effets = pd.DataFrame(effets).sort_values(["indicateur", "minutes"])
    effets["p_bh"] = multipletests(effets["p_mann_whitney"], method="fdr_bh")[1]
    effets.to_csv(SORTIE / "effets_pistes_descriptif.csv", index=False)

    figures(pistes, seuil, pd.DataFrame(effectifs).merge(index, on="fenetre"), par_piste, courbes, seg_res, effets)

    r = seg_res[seg_res["analyse"] == "toutes les pistes"].copy()
    lg = seg_res[seg_res["analyse"] != "toutes les pistes"].set_index("indicateur")
    r["debut_pistes_longues_h"] = (r["indicateur"].map(lg["debut_divergence_h"])
                                   if "debut_divergence_h" in lg.columns else np.nan)
    r = r.sort_values("p_divergence")
    print(f"\nEcart malades - sains sur les {FENETRES_DEBUT} premiers extraits ({', '.join(premiers)}), "
          "en ecarts-types des sains [IC 95 %, bootstrap des pistes] :")
    print(debut[["libelle", "moyenne_sain", "moyenne_malade", "ecart_initial_et", "ic_bas", "ic_haut"]]
          .round(2).to_string(index=False))
    print(f"\nEvolution de l'ecart au cours du temps (modele a trois phases ; extraits ou chaque groupe compte "
          f">= {MIN_POISSONS_GROUPE} poissons visibles ; heures depuis le debut ; amplitude en ecarts-types des sains) :")
    print(r[["libelle", "debut_divergence_h", "debut_ic_bas_h", "debut_ic_haut_h", "fin_transition_h", "amplitude",
             "p_divergence", "debut_pistes_longues_h"]].round(3).to_string(index=False))
    print(f"\nResultats dans {SORTIE}")


def figures(pistes, seuil, effectifs, par_piste, courbes, seg_res, effets):
    plt.rcParams.update({"font.size": 9, "text.color": ENCRE, "axes.labelcolor": ENCRE_2,
                         "figure.facecolor": FOND, "savefig.facecolor": FOND})
    cols = list(INDICATEURS)
    n = len(cols)
    lignes_fig = (n + 1) // 2

    # Fig. 1 : marque
    fig, ax = plt.subplots(figsize=(7, 3.2))
    style_axe(ax)
    v = pistes.loc[pistes["n_marque"] >= MIN_DETECTIONS_MARQUE, "contraste_median"]
    ax.hist(v, bins=60, color="#8a8984", edgecolor=FOND, linewidth=0.5)
    ax.axvspan(seuil - BANDE_MARQUE, seuil + BANDE_MARQUE, color=GRILLE, alpha=0.8, zorder=0)
    ax.axvline(seuil, color=ENCRE, linewidth=1)
    ymax = ax.get_ylim()[1]
    ax.text(seuil - BANDE_MARQUE - 3, ymax * 0.92, "sans marque\n(malades)", ha="right", va="top", fontsize=8, color=ENCRE_2)
    ax.text(seuil + BANDE_MARQUE + 3, ymax * 0.92, "marque blanche\n(sains)", ha="left", va="top", fontsize=8, color=ENCRE_2)
    ax.set_xlabel("Contraste median de la marque par piste (max du disque - fond)")
    ax.set_ylabel("Nombre de pistes")
    ax.set_title("Attribution de l'etat de sante par la marque blanche", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(SORTIE / "fig1_marque.png", dpi=150); plt.close(fig)

    # Fig. 2 : poissons detectes par groupe
    fig, ax = plt.subplots(figsize=(7, 3.2))
    style_axe(ax)
    e = effectifs.sort_values("minutes_depuis_debut")
    t = e["minutes_depuis_debut"] / 60
    for g in ("sain", "malade"):
        ax.plot(t, e[f"n_{g}"], color=COULEURS[g], linewidth=2, marker="o", markersize=4, label=g)
        ax.text(t.iloc[-1] + 0.08, e[f"n_{g}"].iloc[-1], g, color=ENCRE_2, va="center", fontsize=8)
    ax.set_xlabel("Heures depuis le debut de l'enregistrement (≈ injection)")
    ax.set_ylabel("Poissons detectes par image")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Poissons visibles et etiquetes, par groupe", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(SORTIE / "fig2_effectifs.png", dpi=150); plt.close(fig)

    # Fig. 3 : moyenne de chaque groupe par extrait (ponderee par la duree), et dispersion des pistes
    fig, axes = plt.subplots(lignes_fig, 2, figsize=(10, 2.3 * lignes_fig), sharex=True)
    for ax, c in zip(axes.flat, cols):
        style_axe(ax)
        e, _ = courbes[c]
        for g in ("sain", "malade"):
            q = par_piste[par_piste["groupe"] == g].groupby("temps_h")[c].quantile([0.25, 0.75]).unstack()
            ax.fill_between(q.index, q[0.25], q[0.75], color=COULEURS[g], alpha=0.12, linewidth=0)
            ax.plot(e["minutes"] / 60, e[f"moyenne_{g}"], color=COULEURS[g], linewidth=2, label=g)
        ax.set_title(f"{INDICATEURS[c][0]} ({INDICATEURS[c][1]})", loc="left", fontsize=8.5)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Heures depuis le debut de l'enregistrement")
    fig.suptitle("Trait : moyenne du groupe ponderee par la duree de presence ; bande : ecart interquartile des pistes",
                 fontsize=8.5, color=ENCRE_2, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(SORTIE / "fig3_indicateurs.png", dpi=150); plt.close(fig)

    # Fig. 4 (principale) : ecart standardise malades - sains et regression segmentee
    fig, axes = plt.subplots(lignes_fig, 2, figsize=(10, 2.3 * lignes_fig), sharex=True)
    for ax, c in zip(axes.flat, cols):
        style_axe(ax)
        e, r = courbes[c]
        ax.axhline(0, color=ENCRE_2, linewidth=0.8)
        ax.plot(e["minutes"] / 60, e["ecart_standardise"], "o", color=COULEURS["malade"], markersize=4,
                markeredgecolor=FOND, markeredgewidth=0.8)
        titre = INDICATEURS[c][0]
        if "debut_divergence_h" in r and r["p_divergence"] < 0.05:
            ax.axvspan(r["debut_ic_bas_h"], r["debut_ic_haut_h"], color=GRILLE, alpha=0.9, zorder=0)
            ax.plot(r["_t"], r["_ajuste"], color=ENCRE, linewidth=1.5)
            ax.axvline(r["debut_divergence_h"], color=ENCRE, linewidth=1, linestyle="--")
            titre += f"  (p = {r['p_divergence']:.3f})"
        elif "debut_divergence_h" in r:
            titre += f"  (pas de divergence detectee, p = {r['p_divergence']:.2f})"
        ax.set_title(titre, loc="left", fontsize=8.5)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Ecart (ET des sains)")
    for ax in axes[-1]:
        ax.set_xlabel("Heures depuis le debut de l'enregistrement")
    fig.suptitle("Ecart malades - sains par extrait, en ecarts-types des poissons sains. Trait : modele a trois phases ; "
                 "tirets : debut de divergence ; zone grise : son IC 95 %", fontsize=8.5, color=ENCRE_2, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(SORTIE / "fig4_divergence.png", dpi=150); plt.close(fig)

    # Fig. 5 (descriptive) : tailles d'effet par extrait, piste = unite
    fig, axes = plt.subplots(lignes_fig, 2, figsize=(10, 2.3 * lignes_fig), sharex=True)
    for ax, c in zip(axes.flat, cols):
        style_axe(ax)
        e = effets[effets["indicateur"] == c]
        t = e["minutes"] / 60
        ax.axhline(0, color=ENCRE_2, linewidth=0.8)
        ax.fill_between(t, e["ic_bas"], e["ic_haut"], color=COULEURS["malade"], alpha=0.15, linewidth=0)
        ax.plot(t, e["hedges_g"], color=COULEURS["malade"], linewidth=2)
        vals = e[["ic_bas", "ic_haut", "hedges_g"]].to_numpy(dtype=float)
        if np.isfinite(vals).any():
            lo, hi = np.nanmin(vals), np.nanmax(vals)
            marge = 0.1 * (max(hi, 0) - min(lo, 0) + 1e-9)
            ax.set_ylim(min(lo, 0) - marge, max(hi, 0) + marge)
        ax.set_title(INDICATEURS[c][0], loc="left", fontsize=8.5)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Hedges g")
    for ax in axes[-1]:
        ax.set_xlabel("Heures depuis le debut de l'enregistrement")
    fig.suptitle("Descriptif : Hedges g (malades - sains) par extrait, la piste comme unite (IC optimistes, "
                 "plusieurs pistes par poisson)", fontsize=8.5, color=ENCRE_2, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(SORTIE / "fig5_effets_pistes.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
