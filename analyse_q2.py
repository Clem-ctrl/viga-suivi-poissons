"""
Q2 : que gagne-t-on a suivre chaque poisson par IA, par rapport a des mesures plus simples ?

Rappel de la Q1. Les 10 poissons injectes (Aeromonas hydrophila, voie intraperitoneale) se comportent
autrement des les premieres minutes. L'injection, une eventuelle anesthesie (non decrite dans l'article) et
l'infection ne peuvent pas etre separees, et il n'y a ni enregistrement avant l'injection ni bassin temoin.
Les groupes s'appellent donc ici "injectes" et "non injectes", et la question devient : quelle methode repere
un poisson au comportement anormal, au bout de combien de temps d'observation, et a partir de combien de
poissons atteints dans un bassin ?

Methodes comparees
  N0             mesure globale : activite moyenne du bassin (vitesse moyenne de tous les poissons presents).
                 Approche, a partir des trajectoires, la mesure d'activite par pixels d'un systeme classique
                 (verifier_n0_pixels.py compare les deux sur le vrai bassin). Niveau bassin seulement.
  N1 vitesse     une mesure simple par poisson, comme une sortie de ZebraLab : vitesse moyenne (distance
  N1 inactivite  parcourue / duree) ou part du temps inactif (< 0,2 longueur de corps / s).
                 ATTENTION : N1 est calcule sur les pistes de l'IA. ZebraLab suit un animal par puits et ne sait
                 pas suivre 20 poissons dans un meme bassin : N1 est donc une version AVANTAGEE de la methode
                 classique. Si N2 fait mieux que N1, il fait a fortiori mieux que la vraie methode classique.
  N2             score d'anomalie : distance de Mahalanobis a la norme des poissons non injectes, sur 7
                 indicateurs fixes a l'avance (activite, inactivite, regularite de la nage, rotation et courbure
                 du corps issues des points cles, distance a la paroi, distance au voisin). Appris sur les seuls
                 non injectes : les injectes ne servent qu'a l'evaluation.
  N2 supervise   plafond OPTIMISTE : analyse discriminante lineaire entrainee avec les injectes. Les memes 20
                 poissons sont presents dans toutes les periodes : elle peut apprendre ces poissons-la.
  N2 relatif     (bassin seulement) chaque poisson est compare aux autres poissons du meme bassin au meme
                 moment : ce qui change pour tout le bassin a la fois (heure, perturbation) s'annule.
  N0, N1, N2 et N2 relatif sont bilateraux (anomalie dans un sens ou dans l'autre) : on ne suppose pas a
  l'avance le sens de l'effet d'une maladie.

Periode : extraits de DEBUT_MIN a FIN_MIN minutes. Avant, la marque blanche fait des erreurs d'etiquetage
(Q1, fig. 2) ; apres, il reste trop peu d'injectes, et ce sont les survivants.

Validation croisee par blocs de temps. Les extraits sont groupes en blocs de 30 min. Pour chaque bloc, normes et
seuils sont appris sur les autres blocs, puis appliques au bloc. Les taux de fausses alertes OBSERVES sont
donnes a cote des taux vises : ils montrent si un seuil appris a un moment tient a un autre.
IC 95 % : bootstrap des extraits. Les memes 20 poissons sont presents partout : les resultats valent pour ce
bassin et ces poissons.

Niveau 1, alerte par poisson. Une observation = un poisson suivi sans interruption pendant d secondes
(d = 10 s, 30 s, 1 min, 3 min ; creneaux fixes dans chaque extrait). Seuil = 95e (ou 99e) centile des scores des
non injectes. Mesures : AUROC, part des injectes detectes (sensibilite), fausses alertes observees, et part du
temps de presence des poissons utilisable pour une observation de cette duree (plus d est long, plus il faut
une piste longue sans echange d'identite).

Niveau 2, alerte pour le bassin. Bassins virtuels de TAILLE_BASSIN poissons dont 0 a K_MAX injectes, composes
de poissons observes AU MEME MOMENT dans le meme extrait (deux pistes simultanees sont deux poissons differents,
observes dans les memes conditions). Seuls les creneaux ou toutes les compositions sont possibles sont gardes :
chaque nombre d'injectes est evalue sur les memes moments. Statistique du bassin : N0 = activite moyenne ;
N1, N2 = score du poisson le plus anormal ; N2 relatif = plus grand ecart d'un poisson a ses voisins de bassin.
Seuil : 95e centile sur des bassins virtuels sans injecte des autres blocs.
Limites : les bassins virtuels ignorent l'influence des poissons les uns sur les autres ; N0 est approche par
les trajectoires.

Entrees : data/trajectoires/T*.csv.gz, data/fenetres/index_fenetres.csv, et analyse_q1.py dans le meme dossier
(ses fonctions sont reutilisees ; inutile de relancer la Q1).
Sorties (results/q2/) :
  observations.csv            une ligne par poisson observe : indicateurs et scores hors bloc de chaque methode
  alerte_poisson.csv          par methode et duree : AUROC, sensibilite et fausses alertes observees [IC 95 %]
  alerte_poisson_ecarts.csv   AUROC de N2 moins celle du meilleur N1, par duree [IC 95 %]
  alerte_bassin.csv           par methode, duree et nombre d'injectes : taux d'alerte [IC 95 %]
  fig1_alerte_poisson.png, fig2_alerte_bassin.png
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import analyse_q1 as q1

RACINE = Path(__file__).resolve().parent
DOSSIER_TRAJ = RACINE / "data" / "trajectoires"
INDEX = RACINE / "data" / "fenetres" / "index_fenetres.csv"
SORTIE = RACINE / "results" / "q2"

FPS = q1.FPS
DEBUT_MIN, FIN_MIN = 50, 300      # periode analysee (minutes depuis le debut de l'enregistrement)
DUREES_S = (10, 30, 60, 180)      # durees d'observation d'un poisson
DUREES_BASSIN_S = (30, 60)        # durees d'observation pour l'alerte bassin
TAILLE_BASSIN = 8                 # poissons par bassin virtuel (environ 10 non injectes visibles a la fois)
K_MAX = 3                         # injectes par bassin virtuel : 0 a K_MAX
TIRAGES_BASSIN = 30               # bassins tires par creneau et par composition
EXTRAITS_PAR_BLOC = 3             # 3 extraits = 30 min par bloc de validation croisee
FAUSSES_ALERTES = (0.05, 0.01)    # taux de fausses alertes vises (niveau poisson) ; le premier sert au bassin
ECARTES_NORME = 0.05              # part des observations les plus atypiques ecartees pour estimer la norme
N_BOOT = 1000
GRAINE = 42

INDICATEURS_N2 = ["vitesse_moy", "part_immobile", "vitesse_cv", "angle_virage", "courbure_moy",
                  "dist_paroi", "dist_voisin"]
EN_LOG = {"vitesse_moy", "vitesse_cv", "angle_virage", "dist_paroi", "dist_voisin"}

# libelle, couleur, marqueur, trait (couleurs : palette categorielle de reference, creneaux 3 a 7 ;
# les creneaux 1-2, bleu et orange, restent reserves aux groupes comme en Q1)
METHODES = {
    "N0": ("N0 activite globale du bassin", "#8a8984", "s", "-"),
    "N1_vitesse": ("N1 vitesse moyenne", "#eda100", "^", "-"),
    "N1_inactivite": ("N1 part du temps inactif", "#e87ba4", "D", "-"),
    "N2": ("N2 score multi-indicateurs", "#6250d6", "o", "-"),
    "N2_supervise": ("N2 supervise (plafond optimiste)", "#6250d6", "o", "--"),
    "N2_relatif": ("N2 relatif aux voisins de bassin", "#1baf7a", "P", "-"),
}
METHODES_POISSON = ["N1_vitesse", "N1_inactivite", "N2", "N2_supervise"]
METHODES_BASSIN = ["N0", "N1_vitesse", "N1_inactivite", "N2", "N2_relatif"]
GROUPES = {"sain": "non injectes", "malade": "injectes"}
LIBELLES_DUREE = {10: "10 s", 30: "30 s", 60: "1 min", 120: "2 min", 180: "3 min"}


# ---------------------------------------------------------------- 1. observations : un poisson suivi d secondes
def extraire_observations(fichiers, pistes, index):
    """Une observation = une sous-piste etiquetee (Q1 : coupee aux echanges d'identite) presente sur tout un
    creneau de d secondes, sans trou de plus de TROU_MAX images. Creneaux fixes dans chaque extrait."""
    cols = ["tete_x", "tete_y", "corps_x", "corps_y", "queue_x", "queue_y", "horodatage_ms", "dist_voisin_px"]
    tm = q1.TROU_MAX
    obs, presence = [], []
    for f in fichiers:
        fen = f.name.split(".")[0]
        info = index.loc[fen]
        d = q1.appliquer_sous_pistes(pd.read_csv(f), pistes)
        ok = (d[["tete_c", "corps_c", "queue_c"]] > 0.5).all(axis=1)
        L = np.percentile(np.hypot(d.loc[ok, "tete_x"] - d.loc[ok, "queue_x"],
                                   d.loc[ok, "tete_y"] - d.loc[ok, "queue_y"]), 75)
        d, _ = q1.distances_voisins(d)
        d = d[d["groupe"].isin(GROUPES)]
        presence.append({"fenetre": fen, **{g: int((d["groupe"] == g).sum()) for g in GROUPES}})
        n_img = int(info["n_images"])
        avant = len(obs)
        for (sid, grp), g in d.groupby(["sous_id", "groupe"]):
            g = g.drop_duplicates("image").sort_values("image")
            img = g["image"].to_numpy()
            for duree in DUREES_S:
                n = duree * FPS
                for c in range(n_img // n):
                    a, b = c * n, c * n + n - 1
                    i0, i1 = np.searchsorted(img, a), np.searchsorted(img, b, side="right")
                    if i1 - i0 < 2:
                        continue
                    x = img[i0:i1]
                    if x[0] > a + tm or x[-1] < b - tm or np.diff(x).max() > tm + 1:
                        continue
                    s = g.iloc[i0:i1].set_index("image").reindex(np.arange(a, b + 1))
                    s[cols] = s[cols].interpolate(limit_direction="both")
                    ind = q1.indicateurs_segment(s, L)
                    obs.append({"fenetre": fen, "minutes": info["minutes_depuis_debut"], "duree_s": duree,
                                "creneau": c, "sous_id": sid, "groupe": grp, **{k: ind[k] for k in INDICATEURS_N2}})
        print(f"[{fen}] {len(obs) - avant} observations")
    return pd.DataFrame(obs), pd.DataFrame(presence)


# ---------------------------------------------------------------- 2. normes et scores
def matrice(df):
    return np.column_stack([np.log(np.clip(df[c].to_numpy(float), 1e-3, None)) if c in EN_LOG
                            else df[c].to_numpy(float) for c in INDICATEURS_N2])


def echelle_robuste(M):
    med = np.median(M, axis=0)
    mad = 1.4826 * np.median(np.abs(M - med), axis=0)
    sd = M.std(axis=0)
    return med, np.where(mad > 1e-6, mad, np.where(sd > 1e-6, sd, 1.0))


def mahalanobis2(Z, mu, Ci):
    D = Z - mu
    return np.einsum("ij,jk,ik->i", D, Ci, D)


def norme(Z):
    """Centre et inverse de la covariance des lignes de Z, en ecartant les ECARTES_NORME les plus atypiques
    (trois passes) : quelques poissons mal etiquetes ou pistes aberrantes ne doivent pas elargir la norme."""
    p = Z.shape[1]
    garde = np.ones(len(Z), bool)
    for _ in range(3):
        mu = Z[garde].mean(0)
        Ci = np.linalg.inv(np.atleast_2d(np.cov(Z[garde], rowvar=False)) + 1e-3 * np.eye(p))
        d2 = mahalanobis2(Z, mu, Ci)
        garde = d2 <= np.quantile(d2, 1 - ECARTES_NORME)
    return mu, Ci


def scores_pli(cal, toutes):
    """Scores de chaque methode pour toutes les observations, normes apprises sur les observations 'cal'."""
    y = (cal["groupe"] == "malade").to_numpy()
    Mc, Mt = matrice(cal), matrice(toutes)
    med, ech = echelle_robuste(Mc[~y])
    Zc, Zt = (Mc - med) / ech, (Mt - med) / ech
    s = {"N1_vitesse": np.abs(Zt[:, INDICATEURS_N2.index("vitesse_moy")]),
         "N1_inactivite": np.abs(Zt[:, INDICATEURS_N2.index("part_immobile")])}
    mu, Ci = norme(Zc[~y])
    s["N2"] = np.sqrt(mahalanobis2(Zt, mu, Ci))
    # plafond optimiste : analyse discriminante lineaire, entrainee avec les injectes
    n0, n1 = (~y).sum(), y.sum()
    C = (np.cov(Zc[~y], rowvar=False) * (n0 - 1) + np.cov(Zc[y], rowvar=False) * (n1 - 1)) / (n0 + n1 - 2)
    w = np.linalg.solve(C + 1e-3 * np.eye(Zc.shape[1]), Zc[y].mean(0) - Zc[~y].mean(0))
    s["N2_supervise"] = Zt @ w
    return s, Zt


# ---------------------------------------------------------------- 3. bassins virtuels
def bassins_pli(o, s, Z, cal, rng):
    """Bassins virtuels d'un pli. o : observations d'une duree ; s, Z : scores et indicateurs standardises du
    pli ; cal : masque des observations des blocs d'apprentissage. Renvoie les alertes des bassins du bloc teste."""
    N = TAILLE_BASSIN
    vit = o["vitesse_moy"].to_numpy(float)
    sain = (o["groupe"] == "sain").to_numpy()
    creneaux = o.groupby(["fenetre", "creneau"]).indices
    m_cal, m_test, meta = [], [], []
    for (fen, c), pos in creneaux.items():
        ps, pm = pos[sain[pos]], pos[~sain[pos]]
        if len(ps) < N:
            continue
        if cal[pos[0]]:
            m_cal += [rng.choice(ps, N, replace=False) for _ in range(TIRAGES_BASSIN)]
        elif len(pm) >= K_MAX:
            for k in range(K_MAX + 1):
                for _ in range(TIRAGES_BASSIN):
                    m_test.append(np.r_[rng.choice(pm, k, replace=False), rng.choice(ps, N - k, replace=False)])
                    meta.append((fen, c, k))
    if not m_cal or not m_test:
        return pd.DataFrame()
    m_cal, m_test = np.array(m_cal), np.array(m_test)

    def statistiques(M):
        st = {"N0": vit[M].mean(1)}
        for m in ("N1_vitesse", "N1_inactivite", "N2"):
            st[m] = s[m][M].max(1)
        ecarts = Z[M] - np.median(Z[M], axis=1, keepdims=True)       # chaque poisson moins la mediane du bassin
        return st, ecarts

    st_cal, ec_cal = statistiques(m_cal)
    st_test, ec_test = statistiques(m_test)
    mu, Ci = norme(ec_cal.reshape(-1, Z.shape[1]))
    rel = lambda ec: np.sqrt(mahalanobis2(ec.reshape(-1, Z.shape[1]), mu, Ci)).reshape(ec.shape[:2]).max(1)
    st_cal["N2_relatif"], st_test["N2_relatif"] = rel(ec_cal), rel(ec_test)
    # N0 bilateral : ecart a la mediane des bassins sans injecte
    centre = np.median(st_cal["N0"])
    st_cal["N0"], st_test["N0"] = np.abs(st_cal["N0"] - centre), np.abs(st_test["N0"] - centre)
    sortie = pd.DataFrame(meta, columns=["fenetre", "creneau", "k"])
    for m in METHODES_BASSIN:
        sortie[m] = st_test[m] > np.quantile(st_cal[m], 1 - FAUSSES_ALERTES[0])
    return sortie


# ---------------------------------------------------------------- 4. mesures et bootstrap des extraits
def auroc(y, s):
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def mesures_poisson(a):
    y = a["y"]
    r = {}
    for m in METHODES_POISSON:
        r[("auroc", m)] = auroc(y, a[f"score_{m}"])
        for fa in FAUSSES_ALERTES:
            p = round(fa * 100)
            r[(f"sensibilite_{p}", m)] = a[f"alerte{p}_{m}"][y].mean() if y.any() else np.nan
            r[(f"fausses_alertes_{p}", m)] = a[f"alerte{p}_{m}"][~y].mean() if (~y).any() else np.nan
    return r


def bootstrap(df, colonnes, fonction, rng):
    """Valeur et IC 95 % de fonction(tableaux) en reechantillonnant les extraits (avec remise)."""
    par_fen = {f: {c: g[c].to_numpy() for c in colonnes} for f, g in df.groupby("fenetre")}
    fens = list(par_fen)
    def reunir(liste):
        return {c: np.concatenate([par_fen[f][c] for f in liste]) for c in colonnes}
    point = fonction(reunir(fens))
    tirages = [fonction(reunir(rng.choice(fens, len(fens), replace=True))) for _ in range(N_BOOT)]
    ic = {k: np.nanpercentile([t[k] for t in tirages], [2.5, 97.5]) for k in point}
    return point, ic, tirages


# ---------------------------------------------------------------- programme principal
def main():
    for w in (RuntimeWarning, FutureWarning, DeprecationWarning, UserWarning):
        warnings.filterwarnings("ignore", category=w)
    SORTIE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(GRAINE)
    tous = sorted(DOSSIER_TRAJ.glob("T*.csv.gz"))
    if not tous:
        raise SystemExit(f"Aucune trajectoire dans {DOSSIER_TRAJ}")
    index = pd.read_csv(INDEX).set_index("fenetre")

    # Etat (marque blanche) et coupure des pistes aux echanges d'identite, comme en Q1 (seuil sur tous les extraits)
    pistes, seuil = q1.decouper_par_marque(tous)
    print(f"Seuil de marque : {seuil:.1f}")
    periode = [f for f in tous if DEBUT_MIN <= round(index.loc[f.name.split(".")[0], "minutes_depuis_debut"]) <= FIN_MIN]
    print(f"Periode analysee : {periode[0].name.split('.')[0]} a {periode[-1].name.split('.')[0]} "
          f"({len(periode)} extraits)\n")

    obs, presence = extraire_observations(periode, pistes, index)
    n_avant = len(obs)
    obs = obs.dropna(subset=INDICATEURS_N2).reset_index(drop=True)
    print(f"\n{n_avant - len(obs)} observations ecartees (indicateur manquant) sur {n_avant}")
    ordre = sorted(obs["fenetre"].unique(), key=lambda f: index.loc[f, "minutes_depuis_debut"])
    obs["bloc"] = obs["fenetre"].map({f: i // EXTRAITS_PAR_BLOC for i, f in enumerate(ordre)})
    obs["y"] = obs["groupe"] == "malade"
    blocs = sorted(obs["bloc"].unique())

    # Validation croisee par blocs : scores hors bloc (niveau poisson) et bassins virtuels
    morceaux, bassins = [], []
    for duree in DUREES_S:
        o = obs[obs["duree_s"] == duree].reset_index(drop=True)
        hors_bloc = pd.DataFrame(index=o.index)
        for b in blocs:
            test = (o["bloc"] == b).to_numpy()
            cal = ~test
            s, Z = scores_pli(o[cal], o)
            sain_cal = cal & ~o["y"].to_numpy()
            for m in METHODES_POISSON:
                hors_bloc.loc[test, f"score_{m}"] = s[m][test]
                for fa in FAUSSES_ALERTES:
                    hors_bloc.loc[test, f"alerte{round(fa * 100)}_{m}"] = s[m][test] > np.quantile(s[m][sain_cal], 1 - fa)
            if duree in DUREES_BASSIN_S:
                bt = bassins_pli(o, s, Z, cal, rng)
                if len(bt):
                    bassins.append(bt.assign(duree_s=duree, bloc=b))
        morceaux.append(pd.concat([o, hors_bloc], axis=1))
    oof = pd.concat(morceaux, ignore_index=True)
    for c in oof.columns:
        if c.startswith("alerte"):
            oof[c] = oof[c].astype(bool)
    oof.drop(columns=["y"]).assign(groupe=oof["groupe"].map(GROUPES)).to_csv(SORTIE / "observations.csv", index=False)

    # Niveau 1 : alerte par poisson
    presence_tot = presence[list(GROUPES)].sum()
    lignes, ecarts, couverture = [], [], {}
    colonnes = ["y"] + [c for c in oof.columns if c.startswith(("score_", "alerte"))]
    for duree in DUREES_S:
        o = oof[oof["duree_s"] == duree]
        point, ic, tirages = bootstrap(o, colonnes, mesures_poisson, rng)
        for g in GROUPES:
            couverture[(duree, g)] = (o["groupe"] == g).sum() * duree * FPS / presence_tot[g]
        for m in METHODES_POISSON:
            ligne = {"methode": m, "libelle": METHODES[m][0], "duree_s": duree,
                     "n_non_injectes": int((~o["y"]).sum()), "n_injectes": int(o["y"].sum()),
                     "n_extraits": o["fenetre"].nunique(),
                     "temps_utilisable_non_injectes": couverture[(duree, "sain")],
                     "temps_utilisable_injectes": couverture[(duree, "malade")]}
            for mesure in ["auroc"] + [f"{x}_{round(fa * 100)}" for fa in FAUSSES_ALERTES
                                       for x in ("sensibilite", "fausses_alertes")]:
                ligne[mesure] = point[(mesure, m)]
                ligne[f"{mesure}_ic_bas"], ligne[f"{mesure}_ic_haut"] = ic[(mesure, m)]
            lignes.append(ligne)
        # N2 contre le meilleur N1 (choisi sur ces memes donnees : comparaison favorable a N1)
        meilleur = max(("N1_vitesse", "N1_inactivite"), key=lambda m: point[("auroc", m)])
        diff = [t[("auroc", "N2")] - t[("auroc", meilleur)] for t in tirages]
        ecarts.append({"duree_s": duree, "meilleur_N1": meilleur,
                       "auroc_N2_moins_N1": point[("auroc", "N2")] - point[("auroc", meilleur)],
                       "ic_bas": np.nanpercentile(diff, 2.5), "ic_haut": np.nanpercentile(diff, 97.5),
                       "auroc_supervise_moins_N2": point[("auroc", "N2_supervise")] - point[("auroc", "N2")]})
    poisson = pd.DataFrame(lignes)
    poisson.to_csv(SORTIE / "alerte_poisson.csv", index=False)
    ecarts = pd.DataFrame(ecarts)
    ecarts.to_csv(SORTIE / "alerte_poisson_ecarts.csv", index=False)

    # Niveau 2 : alerte pour le bassin
    bassins = pd.concat(bassins, ignore_index=True)
    lignes = []
    for (duree, k), g in bassins.groupby(["duree_s", "k"]):
        point, ic, _ = bootstrap(g, METHODES_BASSIN, lambda a: {m: a[m].mean() for m in METHODES_BASSIN}, rng)
        for m in METHODES_BASSIN:
            lignes.append({"methode": m, "libelle": METHODES[m][0], "duree_s": duree, "taille_bassin": TAILLE_BASSIN,
                           "injectes": k, "taux_alerte": point[m], "ic_bas": ic[m][0], "ic_haut": ic[m][1],
                           "n_bassins": len(g), "n_creneaux": g[["fenetre", "creneau"]].drop_duplicates().shape[0],
                           "n_extraits": g["fenetre"].nunique()})
    bassin = pd.DataFrame(lignes)
    bassin.to_csv(SORTIE / "alerte_bassin.csv", index=False)

    figures(poisson, bassin)
    resume(poisson, ecarts, bassin, obs)


def resume(poisson, ecarts, bassin, obs):
    print("\nObservations retenues (un poisson suivi sans interruption) :")
    for duree, g in obs.groupby("duree_s"):
        p = poisson[poisson["duree_s"] == duree].iloc[0]
        print(f"  {LIBELLES_DUREE[duree]:>5s} : {int(p['n_non_injectes'])} non injectes, {int(p['n_injectes'])} injectes ; "
              f"temps de presence utilisable {p['temps_utilisable_non_injectes']:.0%} / {p['temps_utilisable_injectes']:.0%}")
    print("\nAlerte par poisson : AUROC [IC 95 %] ; injectes detectes / fausses alertes observees (5 % visees)")
    for m in METHODES_POISSON:
        print(f"  {METHODES[m][0]}")
        for _, r in poisson[poisson["methode"] == m].iterrows():
            print(f"    {LIBELLES_DUREE[r['duree_s']]:>5s} : AUROC {r['auroc']:.2f} [{r['auroc_ic_bas']:.2f}, "
                  f"{r['auroc_ic_haut']:.2f}] ; {r['sensibilite_5']:.0%} detectes / {r['fausses_alertes_5']:.1%} fausses alertes")
    print("\nAUROC de N2 moins celle du meilleur N1 :")
    for _, r in ecarts.iterrows():
        print(f"  {LIBELLES_DUREE[r['duree_s']]:>5s} : {r['auroc_N2_moins_N1']:+.3f} [{r['ic_bas']:+.3f}, {r['ic_haut']:+.3f}] "
              f"(meilleur N1 : {r['meilleur_N1']}) ; plafond supervise - N2 : {r['auroc_supervise_moins_N2']:+.3f}")
    for duree in DUREES_BASSIN_S:
        b = bassin[bassin["duree_s"] == duree]
        if b.empty:
            continue
        print(f"\nAlerte pour le bassin (bassins virtuels de {TAILLE_BASSIN} poissons, {LIBELLES_DUREE[duree]} "
              f"d'observation, {b['n_creneaux'].iloc[0]} creneaux dans {b['n_extraits'].iloc[0]} extraits) : taux d'alerte")
        t = b.pivot(index="libelle", columns="injectes", values="taux_alerte")
        t.columns = [f"{k} injecte(s)" for k in t.columns]
        print(t.loc[[METHODES[m][0] for m in METHODES_BASSIN]].apply(lambda c: c.map(lambda v: f"{v:.0%}")).to_string())
    print(f"\nResultats dans {SORTIE}")


def figures(poisson, bassin):
    plt.rcParams.update({"font.size": 9, "text.color": q1.ENCRE, "axes.labelcolor": q1.ENCRE_2,
                         "figure.facecolor": q1.FOND, "savefig.facecolor": q1.FOND})

    def courbe(ax, x, y, bas, haut, m, decalage):
        lib, coul, marq, trait = METHODES[m]
        x = np.asarray(x, float) + decalage
        creux = trait == "--"
        eb = ax.errorbar(x, y, yerr=[np.asarray(y) - np.asarray(bas), np.asarray(haut) - np.asarray(y)], color=coul,
                         linestyle=trait, linewidth=2, marker=marq, markersize=6,
                         markeredgecolor=coul if creux else q1.FOND, markeredgewidth=1.5 if creux else 1,
                         markerfacecolor=q1.FOND if creux else coul, elinewidth=1, capsize=0, label=lib)
        for barres in eb[2]:
            barres.set_linestyle("-")

    # Fig. 1 : alerte par poisson
    durees = list(DUREES_S)
    xs = np.arange(len(durees))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6))
    dec = dict(zip(METHODES_POISSON, np.linspace(-0.12, 0.12, len(METHODES_POISSON))))
    panneaux = [("auroc", "AUROC (0,5 = hasard)", "Separation injectes / non injectes"),
                ("sensibilite_5", "Part des injectes detectes", "Injectes detectes, seuil a 5 % de fausses alertes"),
                ("fausses_alertes_5", "Part des non injectes en alerte", "Fausses alertes observees (5 % visees)")]
    for ax, (col, ylab, titre) in zip(axes.flat, panneaux):
        q1.style_axe(ax)
        for m in METHODES_POISSON:
            p = poisson[poisson["methode"] == m].set_index("duree_s").loc[durees]
            courbe(ax, xs, p[col], p[f"{col}_ic_bas"], p[f"{col}_ic_haut"], m, dec[m])
        ax.set_title(titre, loc="left", fontsize=9)
        ax.set_ylabel(ylab)
    axes[0, 0].axhline(0.5, color=q1.ENCRE_2, linewidth=0.8)
    axes[0, 0].set_ylim(0.45, 1.0)
    axes[0, 1].set_ylim(0, 1)
    axes[1, 0].axhline(FAUSSES_ALERTES[0], color=q1.ENCRE_2, linewidth=1, linestyle=":")
    axes[1, 0].set_ylim(bottom=0)
    ax = axes[1, 1]
    q1.style_axe(ax)
    p = poisson[poisson["methode"] == METHODES_POISSON[0]].set_index("duree_s").loc[durees]
    for g, col in (("sain", "temps_utilisable_non_injectes"), ("malade", "temps_utilisable_injectes")):
        ax.plot(xs, p[col], color=q1.COULEURS[g], linewidth=2, marker="o", markersize=6,
                markeredgecolor=q1.FOND, label=GROUPES[g])
        ax.text(xs[-1] + 0.12, p[col].iloc[-1], GROUPES[g], color=q1.ENCRE_2, va="center", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.3, len(durees) - 0.4)
    ax.set_ylabel("Part du temps de presence")
    ax.set_title("Temps de presence utilisable pour une observation de cette duree", loc="left", fontsize=9)
    for ax in axes.flat:
        ax.set_xticks(xs, [LIBELLES_DUREE[d] for d in durees])
    for ax in axes[1]:
        ax.set_xlabel("Duree d'observation d'un poisson")
    poignees, noms = axes[0, 0].get_legend_handles_labels()
    fig.legend(poignees, noms, loc="upper left", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.01, 0.995))
    fig.text(0.01, 0.008, f"Alerte par poisson, extraits de {DEBUT_MIN} a {FIN_MIN} min. Normes et seuils appris sur "
             "les autres blocs de 30 min. Barres : IC 95 %, bootstrap des extraits.", fontsize=8, color=q1.ENCRE_2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(SORTIE / "fig1_alerte_poisson.png", dpi=150)
    plt.close(fig)

    # Fig. 2 : alerte pour le bassin
    durees_b = sorted(bassin["duree_s"].unique())
    fig, axes = plt.subplots(1, len(durees_b), figsize=(5 * len(durees_b), 3.8), sharey=True, squeeze=False)
    dec = dict(zip(METHODES_BASSIN, np.linspace(-0.14, 0.14, len(METHODES_BASSIN))))
    for ax, duree in zip(axes[0], durees_b):
        q1.style_axe(ax)
        b = bassin[bassin["duree_s"] == duree]
        ax.axhline(FAUSSES_ALERTES[0], color=q1.ENCRE_2, linewidth=1, linestyle=":")
        for m in METHODES_BASSIN:
            p = b[b["methode"] == m].sort_values("injectes")
            courbe(ax, p["injectes"], p["taux_alerte"], p["ic_bas"], p["ic_haut"], m, dec[m])
        ax.set_xticks(range(K_MAX + 1))
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"Poissons injectes dans un bassin virtuel de {TAILLE_BASSIN}")
        ax.set_title(f"Observation de {LIBELLES_DUREE[duree]} ({b['n_creneaux'].iloc[0]} creneaux, "
                     f"{b['n_extraits'].iloc[0]} extraits)", loc="left", fontsize=9)
    axes[0, 0].set_ylabel("Part des bassins en alerte")
    poignees, noms = axes[0, 0].get_legend_handles_labels()
    fig.legend(poignees, noms, loc="upper left", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.01, 0.995))
    fig.text(0.01, 0.01, "Alerte pour le bassin. Seuils regles pour 5 % de fausses alertes (pointilles) sur des bassins "
             "sans injecte d'autres blocs de 30 min. Barres : IC 95 %, bootstrap des extraits.",
             fontsize=8, color=q1.ENCRE_2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.88))
    fig.savefig(SORTIE / "fig2_alerte_bassin.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
