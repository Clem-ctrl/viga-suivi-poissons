"""
Convertit les fichiers .video de la camera Huawei (dataset FHA-MTT) en .mp4.

Format observe : suite de paquets [en-tete de 43 octets][donnees].
  - octet 0      : 0x02
  - octet 1      : 0x02 = video (HEVC Annex B), 0x01 = audio (ignore)
  - octets 10-17 : horodatage en millisecondes depuis 1970 (UTC), little-endian
  - octets 39-42 : longueur des donnees, little-endian
Chaque paquet video contient une image. La video sort sans reencodage (-c copy).

Usage (dans le dossier video_data) :
    python convertir_videos.py            -> convertit tous les .video du dossier
    python convertir_videos.py X.video    -> convertit un seul fichier
Sorties : X.mp4 et X_images.csv (numero d'image, horodatage ms, heure locale Chine).
Necessite FFmpeg accessible dans le PATH.
"""
import csv
import datetime as dt
import os
import shutil
import struct
import subprocess
import sys

HEADER = 43
FPS = 25
TZ_CHINE = dt.timezone(dt.timedelta(hours=8))


def paquets(f):
    """Lit le fichier paquet par paquet ; renvoie (type, horodatage_ms, donnees)."""
    position = 0
    while True:
        h = f.read(HEADER)
        if len(h) < HEADER:
            return
        if h[0] != 0x02 or h[1] not in (0x01, 0x02):
            raise ValueError(f"En-tete inattendu a l'octet {position} : {h[:4].hex(' ')}")
        horodatage = struct.unpack("<Q", h[10:18])[0]
        longueur = struct.unpack("<I", h[39:43])[0]
        donnees = f.read(longueur)
        if len(donnees) < longueur:
            return  # fin de fichier tronquee
        position += HEADER + longueur
        yield h[1], horodatage, donnees


def trouver_ffmpeg():
    """FFmpeg dans le PATH, sinon dans le dossier d'installation de winget."""
    chemin = shutil.which("ffmpeg")
    if chemin:
        return chemin
    racine = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet")
    for dossier, _, fichiers in os.walk(racine):
        if "ffmpeg.exe" in fichiers:
            return os.path.join(dossier, "ffmpeg.exe")
    return None


FFMPEG = None


def convertir(chemin_video):
    base = os.path.splitext(chemin_video)[0]
    sortie_mp4 = base + ".mp4"
    sortie_csv = base + "_images.csv"
    if os.path.exists(sortie_mp4):
        print(f"[deja fait] {sortie_mp4}")
        return
    print(f"[conversion] {os.path.basename(chemin_video)} -> {os.path.basename(sortie_mp4)}")
    ffmpeg = subprocess.Popen(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "hevc", "-r", str(FPS), "-i", "pipe:0",
         "-c", "copy", "-tag:v", "hvc1", sortie_mp4],
        stdin=subprocess.PIPE,
    )
    n = 0
    taille = os.path.getsize(chemin_video)
    with open(chemin_video, "rb") as f, open(sortie_csv, "w", newline="") as fc:
        w = csv.writer(fc)
        w.writerow(["image", "horodatage_ms", "heure_chine"])
        for type_paquet, horodatage, donnees in paquets(f):
            if type_paquet != 0x02:
                continue
            ffmpeg.stdin.write(donnees)
            heure = dt.datetime.fromtimestamp(horodatage / 1000, TZ_CHINE)
            w.writerow([n, horodatage, heure.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]])
            n += 1
            if n % 15000 == 0:
                print(f"   {n} images ({100 * f.tell() / taille:.0f} %)")
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"   termine : {n} images, soit {n / FPS / 60:.1f} min")


def main():
    global FFMPEG
    FFMPEG = trouver_ffmpeg()
    if FFMPEG is None:
        sys.exit("FFmpeg introuvable. Installe-le (winget install -e --id Gyan.FFmpeg).")
    print(f"FFmpeg utilise : {FFMPEG}")
    if len(sys.argv) > 1:
        fichiers = sys.argv[1:]
    else:
        dossier = os.path.dirname(os.path.abspath(__file__))
        fichiers = sorted(os.path.join(dossier, x) for x in os.listdir(dossier) if x.endswith(".video"))
    for chemin in fichiers:
        convertir(chemin)


if __name__ == "__main__":
    main()
