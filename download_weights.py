"""
download_weights.py
====================
Télécharge automatiquement tous les poids de modèles nécessaires
à l'agent GUI OmniParser dans la bonne arborescence.

Structure cible :
    weights/
    ├── icon_caption_florence/   ← Florence-2 (microsoft/Florence-2-base-ft)
    ├── icon_detect/             ← YOLO OmniParser (model.pt)
    └── EasyOCR/                 ← Modèles EasyOCR (craft, recognition)

Usage :
    python download_weights.py
    python download_weights.py --dir /chemin/vers/COMPUTER_USE_LOCAL_AI_AGENT_V6_OMNIPARSER
"""

import os
import sys
import io
import argparse
import urllib.request
import shutil
from pathlib import Path

# ── Force stdout UTF-8 (evite UnicodeEncodeError sur Windows cp1252) ─────────
# DOIT etre fait avant tout print()
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True
    )
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True
    )

# ── Couleurs ANSI ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

SEP = "-" * 60   # ASCII pur — compatible cp1252 et UTF-8

def ok(msg):    print(f"{GREEN}  [OK]  {msg}{RESET}", flush=True)
def info(msg):  print(f"{CYAN}  [ . ] {msg}{RESET}", flush=True)
def warn(msg):  print(f"{YELLOW}  [!]   {msg}{RESET}", flush=True)
def err(msg):   print(f"{RED}  [ERR] {msg}{RESET}", flush=True)
def title(msg): print(f"\n{BOLD}{CYAN}{SEP}\n  {msg}\n{SEP}{RESET}", flush=True)


# ── Barre de progression ──────────────────────────────────────────────────────

def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct     = min(100, downloaded * 100 // total_size)
        filled  = pct // 5
        bar     = "#" * filled + "-" * (20 - filled)   # ASCII pur
        mb_done  = downloaded  / 1_048_576
        mb_total = total_size  / 1_048_576
        print(
            f"\r    [{bar}] {pct:3d}%  {mb_done:.1f}/{mb_total:.1f} MB",
            end="", flush=True
        )
    else:
        mb_done = downloaded / 1_048_576
        print(f"\r    {mb_done:.1f} MB telecharges...", end="", flush=True)


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Telecharge un fichier avec barre de progression. Retourne True si OK."""
    if dest.exists():
        ok(f"Deja present : {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    info(f"Telechargement : {desc or dest.name}")
    print(f"    URL : {url}", flush=True)
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
        print()  # saut de ligne apres la barre
        ok(f"Telecharge -> {dest}")
        return True
    except Exception as e:
        print()
        err(f"Echec : {e}")
        if dest.exists():
            dest.unlink()
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  1. FLORENCE-2  (microsoft/Florence-2-base-ft via HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────

FLORENCE_REPO   = "microsoft/Florence-2-base-ft"
FLORENCE_BRANCH = "main"

# Fichiers indispensables du modèle Florence-2
FLORENCE_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "pytorch_model.bin",        # poids principaux (~900 MB)
    "model.safetensors",        # alternative safetensors (peut remplacer .bin)
]

# Fichiers optionnels (présents selon la version du repo)
FLORENCE_OPTIONAL = {
    "pytorch_model.bin",
    "model.safetensors",
}


def _hf_raw_url(repo: str, branch: str, filename: str) -> str:
    return (
        f"https://huggingface.co/{repo}/resolve/{branch}/{filename}"
    )


def download_florence(weights_dir: Path) -> None:
    title("1/3 - Florence-2  (microsoft/Florence-2-base-ft)")
    dest_dir = weights_dir / "icon_caption_florence"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Methode recommandee : huggingface_hub snapshot_download
    try:
        from huggingface_hub import snapshot_download
        info("huggingface_hub detecte -- snapshot_download...")
        snapshot_download(
            repo_id        = FLORENCE_REPO,
            local_dir      = str(dest_dir),
            ignore_patterns= ["*.gguf", "flax_model*", "tf_model*", "rust_model*"],
        )
        ok(f"Florence-2 telecharge dans {dest_dir}")
        return
    except ImportError:
        warn("huggingface_hub non installe -- telechargement fichier par fichier.")
        warn("Pour un telechargement plus rapide : pip install huggingface_hub")

    # Fallback : telechargement fichier par fichier
    missing_critical   = False
    weights_downloaded = False

    for fname in FLORENCE_FILES:
        dest_file = dest_dir / fname
        url = _hf_raw_url(FLORENCE_REPO, FLORENCE_BRANCH, fname)

        if fname in FLORENCE_OPTIONAL:
            if dest_file.exists():
                ok(f"Deja present : {fname}")
                if "model" in fname:
                    weights_downloaded = True
                continue
            ok_dl = download_file(url, dest_file, fname)
            if ok_dl and "model" in fname:
                weights_downloaded = True
        else:
            ok_dl = download_file(url, dest_file, fname)
            if not ok_dl:
                missing_critical = True

    if missing_critical:
        err("Certains fichiers Florence-2 n'ont pas pu etre telecharges.")
        err("Essayez manuellement :")
        err(f"  huggingface-cli download {FLORENCE_REPO} --local-dir {dest_dir}")
    elif not weights_downloaded:
        warn("Aucun poids (pytorch_model.bin / model.safetensors) telecharge.")
        warn("Essayez : pip install huggingface_hub && python download_weights.py")
    else:
        ok("Florence-2 OK")


# ─────────────────────────────────────────────────────────────────────────────
#  2. YOLO OmniParser  (icon_detect/model.pt)
# ─────────────────────────────────────────────────────────────────────────────

# Repo officiel OmniParser Microsoft sur HuggingFace
OMNIPARSER_REPO   = "microsoft/OmniParser-v2.0"
OMNIPARSER_BRANCH = "main"

# Chemin du modèle YOLO dans le repo HF
YOLO_HF_PATH = "icon_detect/model.pt"


def download_yolo(weights_dir: Path) -> None:
    title("2/3 - YOLO OmniParser  (icon_detect/model.pt)")
    dest_dir  = weights_dir / "icon_detect"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / "model.pt"

    try:
        from huggingface_hub import hf_hub_download
        info("Telechargement via huggingface_hub...")
        downloaded = hf_hub_download(
            repo_id  = OMNIPARSER_REPO,
            filename = YOLO_HF_PATH,
            local_dir= str(weights_dir),
        )
        src = Path(downloaded)
        if src.resolve() != dest_file.resolve():
            shutil.copy2(src, dest_file)
        ok(f"YOLO telecharge -> {dest_file}")
        return
    except ImportError:
        pass
    except Exception as e:
        warn(f"huggingface_hub a echoue ({e}) -- fallback URL directe.")

    # Fallback URL directe HF
    url = _hf_raw_url(OMNIPARSER_REPO, OMNIPARSER_BRANCH, YOLO_HF_PATH)
    ok_dl = download_file(url, dest_file, "model.pt (YOLO OmniParser ~6 MB)")
    if not ok_dl:
        err("Echec du telechargement YOLO.")
        err("Telechargez manuellement :")
        err(f"  huggingface-cli download {OMNIPARSER_REPO} {YOLO_HF_PATH} --local-dir {weights_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  3. EasyOCR  (craft_mlt_25k + english_g2)
# ─────────────────────────────────────────────────────────────────────────────

EASYOCR_MODEL_DIR_ENV = "EASYOCR_MODULE_PATH"

EASYOCR_FILES = {
    "craft_mlt_25k.pth": (
        "https://github.com/JaidedAI/EasyOCR/releases/download/"
        "pre-v1.1.6/craft_mlt_25k.zip"
    ),
    "english_g2.pth": (
        "https://github.com/JaidedAI/EasyOCR/releases/download/"
        "v1.3/english_g2.zip"
    ),
}


def _unzip_to(zip_path: Path, dest_dir: Path) -> None:
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    zip_path.unlink()


def download_easyocr(weights_dir: Path) -> None:
    title("3/3 - EasyOCR  (craft_mlt_25k + english_g2)")
    dest_dir = weights_dir / "EasyOCR"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Methode recommandee : laisser EasyOCR se telecharger lui-meme
    try:
        import easyocr
        info("Pre-chargement EasyOCR (telecharge craft + english_g2 si absent)...")
        info(f"Dossier cible EasyOCR : {dest_dir}")
        os.environ[EASYOCR_MODEL_DIR_ENV] = str(dest_dir)
        reader = easyocr.Reader(
            ['en'],
            model_storage_directory=str(dest_dir),
            download_enabled=True,
            gpu=False,
            verbose=False,
        )
        del reader
        ok(f"EasyOCR telecharge dans {dest_dir}")
        return
    except ImportError:
        warn("easyocr non installe -- telechargement manuel des fichiers .zip.")
        warn("Installez d'abord : pip install easyocr")

    # Fallback : telechargement manuel des .zip puis extraction
    for model_name, zip_url in EASYOCR_FILES.items():
        dest_file = dest_dir / model_name
        if dest_file.exists():
            ok(f"Deja present : {model_name}")
            continue
        zip_dest = dest_dir / (model_name + ".zip")
        ok_dl = download_file(zip_url, zip_dest, model_name)
        if ok_dl:
            info(f"Extraction de {zip_dest.name}...")
            try:
                _unzip_to(zip_dest, dest_dir)
                ok(f"Extrait -> {dest_file}")
            except Exception as e:
                err(f"Extraction echouee : {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  Vérification finale de l'arborescence
# ─────────────────────────────────────────────────────────────────────────────

def verify_structure(weights_dir: Path) -> None:
    title("Verification de la structure weights/")

    checks = {
        "icon_caption_florence/config.json":          "Florence-2 config",
        "icon_caption_florence/tokenizer.json":       "Florence-2 tokenizer",
        "icon_detect/model.pt":                       "YOLO model.pt",
    }
    all_ok = True
    for rel_path, label in checks.items():
        full = weights_dir / rel_path
        if full.exists():
            ok(f"{label:35s}  ({rel_path})")
        else:
            warn(f"{label:35s}  MANQUANT  ({rel_path})")
            all_ok = False

    # Florence poids (au moins l'un des deux)
    florence_dir = weights_dir / "icon_caption_florence"
    has_weights = (
        (florence_dir / "pytorch_model.bin").exists()
        or (florence_dir / "model.safetensors").exists()
    )
    if has_weights:
        ok(f"{'Florence-2 poids (.bin/.safetensors)':35s}")
    else:
        warn("Florence-2 poids (.bin/.safetensors)   MANQUANTS")
        all_ok = False

    # EasyOCR
    easyocr_dir = weights_dir / "EasyOCR"
    if easyocr_dir.exists() and any(easyocr_dir.iterdir()):
        ok(f"{'EasyOCR models':35s}  ({easyocr_dir})")
    else:
        warn(f"{'EasyOCR models':35s}  MANQUANTS  ({easyocr_dir})")
        all_ok = False

    print()
    if all_ok:
        print(f"{GREEN}{BOLD}  [OK] Tous les poids sont presents. Pret a demarrer !{RESET}", flush=True)
    else:
        print(f"{YELLOW}{BOLD}  [!]  Certains fichiers manquent -- voir les avertissements ci-dessus.{RESET}", flush=True)

    print(f"\n  Arborescence finale :\n")
    _print_tree(weights_dir)


def _print_tree(root: Path, prefix: str = "  ", max_depth: int = 3, depth: int = 0) -> None:
    if depth > max_depth:
        return
    try:
        entries = sorted(root.iterdir())
    except PermissionError:
        return
    for i, entry in enumerate(entries):
        connector = "+-- " if i < len(entries) - 1 else "\\-- "
        size_str = ""
        if entry.is_file():
            sz = entry.stat().st_size
            if sz >= 1_048_576:
                size_str = f"  ({sz/1_048_576:.0f} MB)"
            elif sz >= 1024:
                size_str = f"  ({sz/1024:.0f} KB)"
        print(f"{prefix}{connector}{entry.name}{size_str}", flush=True)
        if entry.is_dir() and depth < max_depth:
            extension = "|   " if i < len(entries) - 1 else "    "
            _print_tree(entry, prefix + extension, max_depth, depth + 1)


# ─────────────────────────────────────────────────────────────────────────────
#  Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Télécharge les poids OmniParser (Florence-2, YOLO, EasyOCR)"
    )
    parser.add_argument(
        "--dir", "-d",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Chemin racine du projet (contenant le dossier weights/). "
             "Par défaut : répertoire courant du script."
    )
    parser.add_argument(
        "--skip-florence", action="store_true", help="Ne pas télécharger Florence-2"
    )
    parser.add_argument(
        "--skip-yolo",     action="store_true", help="Ne pas télécharger YOLO"
    )
    parser.add_argument(
        "--skip-easyocr",  action="store_true", help="Ne pas télécharger EasyOCR"
    )
    args = parser.parse_args()

    root_dir    = Path(args.dir).resolve()
    weights_dir = root_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{BOLD}{'='*60}", flush=True)
    print(f"  Telechargement des poids -- OmniParser GUI Agent")
    print(f"  Dossier cible : {weights_dir}")
    print(f"{'='*60}{RESET}", flush=True)

    if not args.skip_florence:
        download_florence(weights_dir)
    else:
        info("Florence-2 ignore (--skip-florence)")

    if not args.skip_yolo:
        download_yolo(weights_dir)
    else:
        info("YOLO ignore (--skip-yolo)")

    if not args.skip_easyocr:
        download_easyocr(weights_dir)
    else:
        info("EasyOCR ignore (--skip-easyocr)")

    verify_structure(weights_dir)


if __name__ == "__main__":
    main()