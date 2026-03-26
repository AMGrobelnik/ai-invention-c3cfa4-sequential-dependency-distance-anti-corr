#!/usr/bin/env python3
"""Compile a merged typological classification table for all UD treebanks.

Merges data from 4 sources:
1. WALS CLDF (Features 49A case-marking, 81A word order)
2. Glottolog CLDF (language family classification)
3. UD GitHub READMEs (genre/modality metadata)
4. HuggingFace commul/universal_dependencies (Case feature proportions)

Output: data_out.json with one row per UD treebank.
"""

import gc
import io
import json
import math
import os
import re
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from datasets import get_dataset_config_names, load_dataset
from loguru import logger

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware detection (cgroup-aware)
# ---------------------------------------------------------------------------
def _detect_cpus() -> int:
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return math.ceil(q / p)
    except (FileNotFoundError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1


def _container_ram_gb() -> Optional[float]:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or 29.0
RAM_BUDGET = int(TOTAL_RAM_GB * 0.6 * 1e9)  # 60% of container RAM

# Set memory limit
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (7200, 7200))  # 2h CPU time

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget={RAM_BUDGET/1e9:.1f} GB")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent
TEMP_DIR = WORKSPACE / "temp" / "datasets"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

WALS_BASE = "https://raw.githubusercontent.com/cldf-datasets/wals/master/cldf"
GLOTTOLOG_URL = "https://raw.githubusercontent.com/glottolog/glottolog-cldf/master/cldf/languages.csv"

# UD Genre fallback
UD_GENRE_FALLBACK_URL = "https://raw.githubusercontent.com/personads/ud-genre/main/ud28/meta.json"

# ---------------------------------------------------------------------------
# STEP 1: Download & process WALS data
# ---------------------------------------------------------------------------
def download_csv(url: str, name: str) -> pd.DataFrame:
    """Download a CSV from URL and return as DataFrame."""
    logger.info(f"Downloading {name} from {url}")
    try:
        resp = requests.get(url, timeout=60, headers=HEADERS)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
        logger.info(f"  {name}: {len(df)} rows, columns={list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Failed to download {name}: {e}")
        raise


def build_wals_lookups() -> tuple[dict, dict]:
    """Build WALS case (49A) and word-order (81A) lookup dicts keyed by ISO and Glottocode."""
    values_df = download_csv(f"{WALS_BASE}/values.csv", "WALS values")
    languages_df = download_csv(f"{WALS_BASE}/languages.csv", "WALS languages")
    codes_df = download_csv(f"{WALS_BASE}/codes.csv", "WALS codes")

    # Save raw CSVs
    values_df.to_csv(TEMP_DIR / "wals_values.csv", index=False)
    languages_df.to_csv(TEMP_DIR / "wals_languages.csv", index=False)
    codes_df.to_csv(TEMP_DIR / "wals_codes.csv", index=False)
    logger.info("Saved WALS CSVs to temp/datasets/")

    # Filter for 49A (case) and 81A (word order)
    case_vals = values_df[values_df["Parameter_ID"] == "49A"].copy()
    wo_vals = values_df[values_df["Parameter_ID"] == "81A"].copy()
    logger.info(f"WALS 49A (case): {len(case_vals)} entries")
    logger.info(f"WALS 81A (word order): {len(wo_vals)} entries")

    # Join with codes for human-readable labels
    codes_49a = codes_df[codes_df["Parameter_ID"] == "49A"][["ID", "Name", "Number"]].rename(
        columns={"ID": "Code_ID", "Name": "case_label", "Number": "case_number"}
    )
    codes_81a = codes_df[codes_df["Parameter_ID"] == "81A"][["ID", "Name", "Number"]].rename(
        columns={"ID": "Code_ID", "Name": "wo_label", "Number": "wo_number"}
    )

    case_vals = case_vals.merge(codes_49a, on="Code_ID", how="left")
    wo_vals = wo_vals.merge(codes_81a, on="Code_ID", how="left")

    # Join with languages for ISO and Glottocode
    lang_cols = ["ID", "Name", "ISO639P3code", "Glottocode", "Family", "Genus", "Macroarea"]
    # Filter to only existing columns
    lang_cols = [c for c in lang_cols if c in languages_df.columns]
    langs = languages_df[lang_cols].rename(columns={"ID": "Language_ID", "Name": "wals_lang_name"})

    case_vals = case_vals.merge(langs, on="Language_ID", how="left")
    wo_vals = wo_vals.merge(langs, on="Language_ID", how="left")

    # Build lookup dicts keyed by ISO639P3code AND Glottocode
    wals_case = {}
    wals_wo = {}

    # Track multiple matches
    iso_case_seen = {}
    iso_wo_seen = {}

    for _, row in case_vals.iterrows():
        iso = str(row.get("ISO639P3code", "")).strip()
        gc_code = str(row.get("Glottocode", "")).strip()
        entry = {
            "wals_language_id": row["Language_ID"],
            "wals_case_category": int(row["case_number"]) if pd.notna(row.get("case_number")) else None,
            "wals_case_label": row.get("case_label", ""),
            "wals_family": row.get("Family", ""),
            "wals_genus": row.get("Genus", ""),
            "wals_macroarea": row.get("Macroarea", ""),
        }
        if iso and iso != "nan":
            if iso in iso_case_seen:
                iso_case_seen[iso] += 1
            else:
                iso_case_seen[iso] = 1
                wals_case[f"iso:{iso}"] = entry
        if gc_code and gc_code != "nan":
            wals_case[f"gc:{gc_code}"] = entry

    for _, row in wo_vals.iterrows():
        iso = str(row.get("ISO639P3code", "")).strip()
        gc_code = str(row.get("Glottocode", "")).strip()
        entry = {
            "wals_language_id": row["Language_ID"],
            "wals_word_order": int(row["wo_number"]) if pd.notna(row.get("wo_number")) else None,
            "wals_word_order_label": row.get("wo_label", ""),
            "wals_family": row.get("Family", ""),
            "wals_genus": row.get("Genus", ""),
            "wals_macroarea": row.get("Macroarea", ""),
        }
        if iso and iso != "nan":
            if iso in iso_wo_seen:
                iso_wo_seen[iso] += 1
            else:
                iso_wo_seen[iso] = 1
                wals_wo[f"iso:{iso}"] = entry
        if gc_code and gc_code != "nan":
            wals_wo[f"gc:{gc_code}"] = entry

    multi_case = {k: v for k, v in iso_case_seen.items() if v > 1}
    multi_wo = {k: v for k, v in iso_wo_seen.items() if v > 1}
    logger.info(f"WALS case lookup: {len(wals_case)} entries, {len(multi_case)} ISOs with multiple matches")
    logger.info(f"WALS word-order lookup: {len(wals_wo)} entries, {len(multi_wo)} ISOs with multiple matches")

    return wals_case, wals_wo, multi_case, multi_wo


# ---------------------------------------------------------------------------
# STEP 2: Download & process Glottolog data
# ---------------------------------------------------------------------------
def build_glottolog_lookup() -> dict:
    """Build Glottolog family lookup keyed by ISO and Glottocode."""
    glot_df = download_csv(GLOTTOLOG_URL, "Glottolog languages")
    glot_df.to_csv(TEMP_DIR / "glottolog_languages.csv", index=False)

    # Separate families and languages
    families = glot_df[glot_df["Level"] == "family"][["Glottocode", "Name"]].copy()
    family_lookup = dict(zip(families["Glottocode"], families["Name"]))
    logger.info(f"Glottolog families: {len(family_lookup)}")

    languages = glot_df[glot_df["Level"] == "language"].copy()
    logger.info(f"Glottolog languages: {len(languages)}")

    glottolog = {}
    for _, row in languages.iterrows():
        iso = str(row.get("ISO639P3code", "")).strip()
        gc_code = str(row.get("Glottocode", "")).strip()
        fam_id = str(row.get("Family_ID", "")).strip()
        is_isolate = str(row.get("Is_Isolate", "")).strip().lower() == "true"

        if is_isolate:
            family_name = f"{row['Name']} (isolate)"
        elif fam_id and fam_id != "nan" and fam_id in family_lookup:
            family_name = family_lookup[fam_id]
        else:
            family_name = ""

        entry = {
            "glottocode": gc_code,
            "family_name": family_name,
            "macroarea": row.get("Macroarea", ""),
            "is_isolate": is_isolate,
        }

        if iso and iso != "nan":
            glottolog[f"iso:{iso}"] = entry
        if gc_code and gc_code != "nan":
            glottolog[f"gc:{gc_code}"] = entry

    logger.info(f"Glottolog lookup: {len(glottolog)} entries")
    return glottolog


# ---------------------------------------------------------------------------
# STEP 3: Enumerate UD treebanks + extract genre/modality
# ---------------------------------------------------------------------------

# UD language code to language name mapping (comprehensive)
UD_LANG_MAP = {
    "af": "Afrikaans", "akk": "Akkadian", "am": "Amharic", "grc": "Ancient_Greek",
    "apu": "Apurina", "ar": "Arabic", "hy": "Armenian", "aii": "Assyrian",
    "bm": "Bambara", "eu": "Basque", "be": "Belarusian", "bho": "Bhojpuri",
    "br": "Breton", "bg": "Bulgarian", "bxr": "Buryat", "yue": "Cantonese",
    "ca": "Catalan", "ceb": "Cebuano", "zh": "Chinese", "lzh": "Classical_Chinese",
    "cop": "Coptic", "hr": "Croatian", "cs": "Czech", "da": "Danish",
    "nl": "Dutch", "en": "English", "myv": "Erzya", "et": "Estonian",
    "fo": "Faroese", "fi": "Finnish", "fr": "French", "gl": "Galician",
    "de": "German", "got": "Gothic", "el": "Greek", "gub": "Guajajara",
    "he": "Hebrew", "hi": "Hindi", "hit": "Hittite", "hu": "Hungarian",
    "is": "Icelandic", "id": "Indonesian", "ga": "Irish", "it": "Italian",
    "ja": "Japanese", "jv": "Javanese", "krl": "Karelian", "kk": "Kazakh",
    "kfm": "Khunsari", "ko": "Korean", "kmr": "Kurmanji", "la": "Latin",
    "lv": "Latvian", "lt": "Lithuanian", "olo": "Livvi", "mk": "Macedonian",
    "mt": "Maltese", "gv": "Manx", "mr": "Marathi", "gun": "Mbyá_Guarani",
    "mdf": "Moksha", "mn": "Mongolian", "pcm": "Naija", "sme": "North_Sami",
    "nb": "Norwegian", "nn": "Norwegian", "no": "Norwegian",
    "cu": "Old_Church_Slavonic", "fro": "Old_French", "orv": "Old_East_Slavic",
    "otk": "Old_Turkish", "fa": "Persian", "pl": "Polish", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "sa": "Sanskrit", "gd": "Scottish_Gaelic",
    "sr": "Serbian", "sms": "Skolt_Sami", "sk": "Slovak", "sl": "Slovenian",
    "soj": "Soi", "hsb": "Upper_Sorbian", "es": "Spanish", "sv": "Swedish",
    "gsw": "Swiss_German", "tl": "Tagalog", "ta": "Tamil", "te": "Telugu",
    "th": "Thai", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu",
    "ug": "Uyghur", "vi": "Vietnamese", "wbp": "Warlpiri", "cy": "Welsh",
    "wo": "Wolof", "yo": "Yoruba", "ess": "Yupik",
    "ab": "Abkhaz", "abq": "Abaza", "aqz": "Akuntsu", "sq": "Albanian",
    "bej": "Beja", "qfn": "Frisian_Dutch", "qhe": "Hindi_English",
    "qtd": "Turkish_German", "swl": "Swedish_Sign_Language",
    "koi": "Komi_Permyak", "kpv": "Komi_Zyrian", "mpu": "Makurap",
    "nds": "Low_Saxon", "nyq": "Nayini", "olo": "Livvi",
    "sms": "Skolt_Sami", "tpn": "Tupinamba", "urb": "Kaapor",
    "xav": "Xavante", "xnr": "Kangri",
}

# ISO 639-3 codes for UD language codes
UD_ISO_MAP = {
    "af": "afr", "akk": "akk", "am": "amh", "grc": "grc", "apu": "apu",
    "ar": "ara", "hy": "hye", "aii": "aii", "bm": "bam", "eu": "eus",
    "be": "bel", "bho": "bho", "br": "bre", "bg": "bul", "bxr": "bxr",
    "yue": "yue", "ca": "cat", "ceb": "ceb", "zh": "zho", "lzh": "lzh",
    "cop": "cop", "hr": "hrv", "cs": "ces", "da": "dan", "nl": "nld",
    "en": "eng", "myv": "myv", "et": "est", "fo": "fao", "fi": "fin",
    "fr": "fra", "gl": "glg", "de": "deu", "got": "got", "el": "ell",
    "gub": "gub", "he": "heb", "hi": "hin", "hit": "hit", "hu": "hun",
    "is": "isl", "id": "ind", "ga": "gle", "it": "ita", "ja": "jpn",
    "jv": "jav", "krl": "krl", "kk": "kaz", "kfm": "kfm", "ko": "kor",
    "kmr": "kmr", "la": "lat", "lv": "lav", "lt": "lit", "olo": "olo",
    "mk": "mkd", "mt": "mlt", "gv": "glv", "mr": "mar", "gun": "gun",
    "mdf": "mdf", "mn": "khk", "pcm": "pcm", "sme": "sme",
    "nb": "nob", "nn": "nno", "no": "nor",
    "cu": "chu", "fro": "fro", "orv": "orv", "otk": "otk",
    "fa": "fas", "pl": "pol", "pt": "por", "ro": "ron", "ru": "rus",
    "sa": "san", "gd": "gla", "sr": "srp", "sms": "sms", "sk": "slk",
    "sl": "slv", "soj": "soj", "hsb": "hsb", "es": "spa", "sv": "swe",
    "gsw": "gsw", "tl": "tgl", "ta": "tam", "te": "tel", "th": "tha",
    "tr": "tur", "uk": "ukr", "ur": "urd", "ug": "uig", "vi": "vie",
    "wbp": "wbp", "cy": "cym", "wo": "wol", "yo": "yor", "ess": "ess",
    "ab": "abk", "abq": "abq", "aqz": "aqz", "sq": "sqi",
    "bej": "bej", "koi": "koi", "kpv": "kpv", "mpu": "mpu",
    "nds": "nds", "nyq": "nyq", "swl": "swl",
    "tpn": "tpn", "urb": "urb", "xav": "xav", "xnr": "xnr",
    # Non-standard q-prefix codes
    "qfn": "qfn", "qhe": "qhe", "qtd": "qtd",
}

# Macrolanguage to individual language fallbacks (for Glottolog matching)
MACRO_TO_INDIVIDUAL = {
    "ara": "arb",  # Arabic → Standard Arabic
    "zho": "cmn",  # Chinese → Mandarin Chinese
    "fas": "pes",  # Persian → Iranian Persian
    "nor": "nob",  # Norwegian → Norwegian Bokmaal
    "msa": "zsm",  # Malay → Standard Malay
    "est": "ekk",  # Estonian → Standard Estonian
    "lav": "lvs",  # Latvian → Standard Latvian
    "sqi": "als",  # Albanian → Tosk Albanian
}


def get_iso3_for_ud_code(ud_code: str) -> tuple[str, bool]:
    """Get ISO 639-3 code for a UD language code. Returns (iso3, nonstandard)."""
    if ud_code in UD_ISO_MAP:
        iso = UD_ISO_MAP[ud_code]
        nonstandard = iso.startswith("q")
        return iso, nonstandard

    # If code is already 3 chars, it might be ISO 639-3 directly
    if len(ud_code) == 3:
        return ud_code, ud_code.startswith("q")

    # Try iso639-lang library
    try:
        from iso639 import Lang
        iso3 = Lang(ud_code).pt3
        return iso3, False
    except Exception:
        pass

    return ud_code, True


def get_ud_language_name(ud_code: str) -> str:
    """Get human-readable language name for a UD language code."""
    if ud_code in UD_LANG_MAP:
        return UD_LANG_MAP[ud_code].replace("_", " ")
    try:
        from iso639 import Lang
        return Lang(ud_code).name
    except Exception:
        return ud_code


def parse_config_to_lang_treebank(config: str) -> tuple[str, str]:
    """Split UD config name into language code and treebank name."""
    parts = config.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return parts[0], ""


# Known treebank name capitalizations that differ from simple capitalize/upper
TB_NAME_MAP = {
    "afribooms": "AfriBooms", "pisandub": "PISANDUB", "riao": "RIAO",
    "tudet": "TuDeT", "staf": "STAF", "tsa": "TSA", "divital": "Divital",
    "uzh": "UZH", "atb": "ATB", "abnc": "ABNC", "ewt": "EWT", "gum": "GUM",
    "pud": "PUD", "gsd": "GSD", "hdt": "HDT", "tdt": "TDT", "itt": "ITT",
    "ittb": "ITTB", "llct": "LLCT", "proiel": "PROIEL", "padt": "PADT",
    "partut": "ParTUT", "lines": "LinES", "talbanken": "Talbanken",
    "sst": "SST", "nsc": "NSC", "nynorsklia": "NynorskLIA",
    "rhapsodie": "Rhapsodie", "gumreddit": "GUMReddit", "atis": "ATIS",
    "gentle": "GENTLE", "littleprince": "LittlePrince", "eslspok": "ESLSpok",
    "ctetex": "CTETEX", "childes": "CHILDES", "ddt": "DDT", "lfg": "LFG",
    "bdt": "BDT", "pdb": "PDB", "cat": "CAT", "fic": "FicTree",
    "fictree": "FicTree", "cltt": "CLTT", "artt": "ArTT", "spmrl": "SPMRL",
    "htb": "HTB", "ktb": "KTB", "isdt": "ISDT", "vit": "VIT",
    "twittiro": "TWITTIRO", "postwita": "PoSTWITA", "marktest": "MarkTest",
    "penn": "Penn", "oldcio": "OldCIO", "kaist": "Kaist", "gkc": "GKC",
    "ikdp": "IKDP", "lvtb": "LVTB", "alksnis": "ALKSNIS",
    "hse": "HSE", "syntagrus": "SynTagRus", "taiga": "Taiga",
    "ufal": "UFAL", "snk": "SNK", "set": "SET", "bokmaal": "Bokmaal",
    "nynorsk": "Nynorsk", "fame": "FAME", "lassysmall": "LassySmall",
    "alpino": "Alpino", "ancora": "AnCora", "pdt": "PDT", "cac": "CAC",
    "vsrp": "VSRP", "msr": "MSR", "odt": "ODT", "ofn": "OFN",
    "torot": "TOROT", "rnc": "RNC", "poetry": "Poetry",
    "imst": "IMST", "penn": "Penn", "boun": "BOUN", "tourism": "Tourism",
    "kenet": "Kenet", "iu": "IU", "clasp": "CLASP",
}


def fetch_readme_genre(lang_name: str, treebank_name: str, session: requests.Session) -> tuple[Optional[str], Optional[str]]:
    """Fetch genre metadata from UD GitHub README."""
    if not treebank_name:
        return None, "unavailable"

    # Build candidate treebank name variants
    tb_variants = set()
    # Use known mapping first
    if treebank_name.lower() in TB_NAME_MAP:
        tb_variants.add(TB_NAME_MAP[treebank_name.lower()])
    tb_variants.add(treebank_name[0].upper() + treebank_name[1:])  # Afribooms
    tb_variants.add(treebank_name.upper())  # AFRIBOOMS
    tb_variants.add(treebank_name.capitalize())  # Afribooms
    # If short (<=5 chars), very likely an acronym: EWT, GUM, PUD, etc.
    if len(treebank_name) <= 5:
        tb_variants.add(treebank_name.upper())

    # Try master, main, dev branches AND both README.md and README.txt
    urls_to_try = []
    for tb in tb_variants:
        for branch in ["master", "dev", "main"]:
            for readme_name in ["README.md", "README.txt"]:
                urls_to_try.append(
                    f"https://raw.githubusercontent.com/UniversalDependencies/UD_{lang_name}-{tb}/{branch}/{readme_name}"
                )

    for url in urls_to_try:
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200:
                text = resp.text
                # Parse genre from machine-readable metadata section
                genre_match = re.search(r'Genre:\s*(.+)', text)
                if genre_match:
                    genre_tags = genre_match.group(1).strip()
                    return genre_tags, "readme_genre"
        except Exception:
            pass

    return None, "unavailable"


def determine_modality(genre_tags: Optional[str]) -> Optional[str]:
    """Determine modality from genre tags."""
    if not genre_tags:
        return None
    tags = genre_tags.lower().split()
    if tags == ["spoken"]:
        return "spoken"
    elif "spoken" in tags:
        return "mixed"
    else:
        return "written"


def fetch_genre_fallback(session: requests.Session) -> dict:
    """Fetch pre-compiled genre metadata as fallback.

    Returns dict keyed by config name (e.g. 'af_afribooms') -> genre string.
    The source data is structured as Language -> treebanks -> Treebank -> files.
    We extract genre from filenames which follow the pattern: lang_tb-ud-split.conllu.
    """
    result = {}
    try:
        resp = session.get(UD_GENRE_FALLBACK_URL, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            # Navigate the nested structure
            for lang_name, lang_data in data.items():
                if not isinstance(lang_data, dict):
                    continue
                treebanks = lang_data.get("treebanks", {})
                for tb_name, tb_data in treebanks.items():
                    if not isinstance(tb_data, dict):
                        continue
                    files = tb_data.get("files", {})
                    # Extract config name from first filename
                    for fname in files:
                        # e.g. "af_afribooms-ud-dev.conllu" -> "af_afribooms"
                        config_match = re.match(r'([a-z]+_[a-z0-9]+)-ud-', fname)
                        if config_match:
                            config_key = config_match.group(1)
                            # Check if there are genre annotations in the data
                            file_data = files[fname]
                            if isinstance(file_data, dict) and "genre" in file_data:
                                genres = file_data["genre"]
                                if isinstance(genres, list):
                                    result[config_key] = " ".join(genres)
                                elif isinstance(genres, dict):
                                    result[config_key] = " ".join(genres.keys())
                                else:
                                    result[config_key] = str(genres)
                            break  # Only need one file per treebank
            logger.info(f"Parsed fallback genre data: {len(result)} treebank entries")
    except Exception as e:
        logger.warning(f"Failed to load genre fallback: {e}")
    return result


# ---------------------------------------------------------------------------
# STEP 4: Compute Case feature proportions from HuggingFace
# ---------------------------------------------------------------------------
def compute_case_proportion(config: str, max_retries: int = 2) -> dict:
    """Compute Case feature proportion for a single UD treebank config."""
    result = {"config": config, "total_tokens": 0, "case_tokens": 0, "proportion": None, "error": None}

    for attempt in range(max_retries):
        try:
            ds = load_dataset("commul/universal_dependencies", config)
            total = 0
            case_count = 0
            for split_name in ds:
                for sent in ds[split_name]:
                    feats = sent.get("feats", [])
                    if feats:
                        for feat in feats:
                            total += 1
                            if feat and "Case=" in str(feat):
                                case_count += 1
            result["total_tokens"] = total
            result["case_tokens"] = case_count
            result["proportion"] = case_count / total if total > 0 else 0.0
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                logger.debug(f"Retry {attempt+1} for {config}: {e}")
                time.sleep(2)
            else:
                result["error"] = str(e)[:200]
                logger.warning(f"Failed to load {config}: {str(e)[:100]}")
    return result


# ---------------------------------------------------------------------------
# STEP 5: Main merge logic
# ---------------------------------------------------------------------------
@logger.catch
def main(max_configs: Optional[int] = None, skip_case: bool = False, skip_readme: bool = False):
    """Main entry point. max_configs limits number of treebanks processed."""
    logger.info("=" * 60)
    logger.info("Starting UD Typological Classification Table compilation")
    logger.info("=" * 60)

    # STEP 1: WALS data
    logger.info("STEP 1: Building WALS lookups...")
    t0 = time.time()
    wals_case, wals_wo, multi_case, multi_wo = build_wals_lookups()
    logger.info(f"STEP 1 done in {time.time()-t0:.1f}s")

    # STEP 2: Glottolog data
    logger.info("STEP 2: Building Glottolog lookup...")
    t0 = time.time()
    glottolog = build_glottolog_lookup()
    logger.info(f"STEP 2 done in {time.time()-t0:.1f}s")

    # STEP 3: Enumerate UD treebanks
    logger.info("STEP 3: Enumerating UD treebanks...")
    t0 = time.time()

    try:
        configs = get_dataset_config_names("commul/universal_dependencies")
        logger.info(f"Found {len(configs)} configs from commul/universal_dependencies")
    except Exception as e:
        logger.warning(f"commul failed: {e}, trying fallback...")
        configs = get_dataset_config_names("universal-dependencies/universal_dependencies")
        logger.info(f"Found {len(configs)} configs from universal-dependencies/universal_dependencies")

    if max_configs:
        configs = configs[:max_configs]
        logger.info(f"Limited to {max_configs} configs for testing")

    logger.info(f"STEP 3a done in {time.time()-t0:.1f}s: {len(configs)} treebanks")

    # STEP 3b: Fetch genre/modality from GitHub READMEs
    logger.info("STEP 3b: Fetching genre/modality from UD GitHub READMEs...")
    t0 = time.time()
    genre_data = {}  # config -> (genre_tags, modality_source)

    if not skip_readme:
        # First try the fallback (faster)
        session = requests.Session()
        if HEADERS:
            session.headers.update(HEADERS)

        fallback_genres = fetch_genre_fallback(session)

        # Build a mapping from config names to their readme info
        configs_needing_fetch = []
        for config in configs:
            # Try fallback first (keyed by config name e.g. "af_afribooms")
            if fallback_genres and config in fallback_genres:
                genre_str = fallback_genres[config]
                genre_data[config] = (genre_str, "fallback_meta_json")
                continue
            configs_needing_fetch.append(config)

        logger.info(f"Genre from fallback: {len(genre_data)}, still need: {len(configs_needing_fetch)}")

        # Fetch remaining from GitHub READMEs with rate limiting
        def fetch_single_readme(config):
            lang_code, tb_name = parse_config_to_lang_treebank(config)
            lang_name = get_ud_language_name(lang_code)
            # Clean language name for URL (replace spaces with underscores)
            lang_name_url = lang_name.replace(" ", "_")
            genre_tags, source = fetch_readme_genre(lang_name_url, tb_name, session)
            return config, genre_tags, source

        # Use ThreadPool with rate limiting
        with ThreadPoolExecutor(max_workers=min(NUM_CPUS, 4)) as executor:
            futures = {}
            for i, config in enumerate(configs_needing_fetch):
                futures[executor.submit(fetch_single_readme, config)] = config

            done_count = 0
            for future in as_completed(futures):
                try:
                    cfg, tags, source = future.result()
                    genre_data[cfg] = (tags, source)
                    done_count += 1
                    if done_count % 50 == 0:
                        logger.info(f"  README fetch progress: {done_count}/{len(configs_needing_fetch)}")
                except Exception as e:
                    cfg = futures[future]
                    genre_data[cfg] = (None, "error")
                    logger.debug(f"  README fetch error for {cfg}: {e}")

    logger.info(f"STEP 3b done in {time.time()-t0:.1f}s: {len(genre_data)} genre entries")

    # STEP 4: Compute Case feature proportions
    logger.info("STEP 4: Computing Case feature proportions...")
    t0 = time.time()
    case_proportions = {}

    if not skip_case:
        # Process in batches with ThreadPoolExecutor
        num_workers = min(NUM_CPUS, 3)  # Conservative to avoid memory issues
        logger.info(f"Processing {len(configs)} configs with {num_workers} workers")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(compute_case_proportion, cfg): cfg for cfg in configs}
            done_count = 0
            for future in as_completed(futures):
                try:
                    result = future.result()
                    case_proportions[result["config"]] = result
                    done_count += 1
                    if done_count % 20 == 0:
                        logger.info(f"  Case proportion progress: {done_count}/{len(configs)} "
                                    f"({done_count/len(configs)*100:.0f}%)")
                        gc.collect()
                except Exception as e:
                    cfg = futures[future]
                    case_proportions[cfg] = {"config": cfg, "total_tokens": 0, "case_tokens": 0,
                                             "proportion": None, "error": str(e)[:200]}
                    done_count += 1

        logger.info(f"STEP 4 done in {time.time()-t0:.1f}s")
        successful = sum(1 for v in case_proportions.values() if v["proportion"] is not None)
        logger.info(f"  Successful: {successful}/{len(configs)}")
    else:
        logger.info("STEP 4 skipped (skip_case=True)")

    # STEP 5: Merge into final table
    logger.info("STEP 5: Merging into final table...")
    rows = []

    for config in configs:
        lang_code, tb_name = parse_config_to_lang_treebank(config)
        iso3, nonstandard = get_iso3_for_ud_code(lang_code)
        lang_name = get_ud_language_name(lang_code)

        # Build list of ISO codes to try (primary + macrolanguage fallback)
        iso_candidates = [iso3]
        if iso3 in MACRO_TO_INDIVIDUAL:
            iso_candidates.append(MACRO_TO_INDIVIDUAL[iso3])

        # Look up WALS case data
        case_entry = None
        case_quality = "none"
        for iso_try in iso_candidates:
            if f"iso:{iso_try}" in wals_case:
                case_entry = wals_case[f"iso:{iso_try}"]
                case_quality = "exact_iso"
                break
        if case_entry is None:
            # Try glottocode fallback via glottolog
            for iso_try in iso_candidates:
                gc_info = glottolog.get(f"iso:{iso_try}", {})
                gc_code = gc_info.get("glottocode", "")
                if gc_code and f"gc:{gc_code}" in wals_case:
                    case_entry = wals_case[f"gc:{gc_code}"]
                    case_quality = "glottocode_fallback"
                    break

        # Look up WALS word order data
        wo_entry = None
        wo_quality = "none"
        for iso_try in iso_candidates:
            if f"iso:{iso_try}" in wals_wo:
                wo_entry = wals_wo[f"iso:{iso_try}"]
                wo_quality = "exact_iso"
                break
        if wo_entry is None:
            for iso_try in iso_candidates:
                gc_info = glottolog.get(f"iso:{iso_try}", {})
                gc_code = gc_info.get("glottocode", "")
                if gc_code and f"gc:{gc_code}" in wals_wo:
                    wo_entry = wals_wo[f"gc:{gc_code}"]
                    wo_quality = "glottocode_fallback"
                    break

        # Look up family: WALS first, then Glottolog fallback
        family = ""
        genus = ""
        macroarea = ""
        is_isolate = False
        family_quality = "none"

        # WALS family (from either case or wo entry)
        wals_fam = (case_entry or wo_entry or {}).get("wals_family", "")
        wals_gen = (case_entry or wo_entry or {}).get("wals_genus", "")
        wals_macro = (case_entry or wo_entry or {}).get("wals_macroarea", "")

        if wals_fam:
            family = wals_fam
            genus = wals_gen
            macroarea = wals_macro
            family_quality = "exact_iso" if (case_quality == "exact_iso" or wo_quality == "exact_iso") else "glottocode_fallback"
        else:
            # Glottolog fallback (try primary ISO, then macrolanguage individual)
            gc_info = {}
            for iso_try in iso_candidates:
                gc_info = glottolog.get(f"iso:{iso_try}", {})
                if gc_info:
                    break
            if gc_info:
                family = gc_info.get("family_name", "")
                macroarea = gc_info.get("macroarea", "")
                is_isolate = gc_info.get("is_isolate", False)
                family_quality = "exact_iso" if family else "none"

        # Glottocode (try primary ISO, then macrolanguage individual)
        gc_info = {}
        for iso_try in iso_candidates:
            gc_info = glottolog.get(f"iso:{iso_try}", {})
            if gc_info:
                break
        glottocode = gc_info.get("glottocode", "") if gc_info else ""

        # Genre/modality
        genre_tags_raw, modality_source = genre_data.get(config, (None, "unavailable"))
        genre_tags = genre_tags_raw if genre_tags_raw else None
        modality = determine_modality(genre_tags)
        if modality_source == "unavailable":
            modality = None

        # Case proportion
        case_prop = case_proportions.get(config, {})
        ud_case_proportion = case_prop.get("proportion")
        ud_total_tokens = case_prop.get("total_tokens", 0)
        ud_case_tokens = case_prop.get("case_tokens", 0)

        # WALS language ID
        wals_lang_id = (case_entry or wo_entry or {}).get("wals_language_id", "")

        # Build output format
        # Build the "output" summary string
        parts = [lang_name]
        if wo_entry:
            parts.append(wo_entry.get("wals_word_order_label", ""))
        if case_entry:
            parts.append(case_entry.get("wals_case_label", ""))
        if family:
            parts.append(family)
        if modality:
            parts.append(modality)
        output_str = " | ".join([p for p in parts if p])

        row = {
            "input": config,
            "output": output_str,
            "metadata_fold": "all",
            "metadata_treebank_id": config,
            "metadata_language_name": lang_name,
            "metadata_iso_639_3": iso3,
            "metadata_glottocode": glottocode if glottocode else None,
            "metadata_wals_language_id": wals_lang_id if wals_lang_id else None,
            "metadata_wals_case_category": case_entry["wals_case_category"] if case_entry else None,
            "metadata_wals_case_label": case_entry["wals_case_label"] if case_entry else None,
            "metadata_wals_word_order": wo_entry["wals_word_order"] if wo_entry else None,
            "metadata_wals_word_order_label": wo_entry["wals_word_order_label"] if wo_entry else None,
            "metadata_language_family": family if family else None,
            "metadata_language_genus": genus if genus else None,
            "metadata_macroarea": macroarea if macroarea else None,
            "metadata_is_isolate": is_isolate,
            "metadata_modality": modality,
            "metadata_genre_tags": genre_tags,
            "metadata_ud_case_proportion": round(ud_case_proportion, 6) if ud_case_proportion is not None else None,
            "metadata_ud_total_tokens": ud_total_tokens,
            "metadata_ud_case_tokens": ud_case_tokens,
            "metadata_mapping_quality_case": case_quality,
            "metadata_mapping_quality_wordorder": wo_quality,
            "metadata_mapping_quality_family": family_quality,
            "metadata_modality_source": modality_source if modality_source else "unavailable",
            "metadata_has_wals_case": case_entry is not None,
            "metadata_has_wals_wordorder": wo_entry is not None,
            "metadata_has_family": bool(family),
            "metadata_has_modality": modality is not None,
            "metadata_nonstandard_code": nonstandard,
            "metadata_multiple_wals_matches": iso3 in multi_case or iso3 in multi_wo,
        }
        rows.append(row)

    logger.info(f"STEP 5 done: {len(rows)} rows")

    # STEP 6: Coverage statistics
    logger.info("=" * 60)
    logger.info("STEP 6: Coverage Statistics")
    logger.info("=" * 60)
    logger.info(f"Total treebanks: {len(rows)}")

    has_case = sum(1 for r in rows if r["metadata_has_wals_case"])
    has_wo = sum(1 for r in rows if r["metadata_has_wals_wordorder"])
    has_fam = sum(1 for r in rows if r["metadata_has_family"])
    has_mod = sum(1 for r in rows if r["metadata_has_modality"])
    logger.info(f"WALS 49A (case):     {has_case}/{len(rows)} ({100*has_case/len(rows):.1f}%)")
    logger.info(f"WALS 81A (word order): {has_wo}/{len(rows)} ({100*has_wo/len(rows):.1f}%)")
    logger.info(f"Language family:     {has_fam}/{len(rows)} ({100*has_fam/len(rows):.1f}%)")
    logger.info(f"Modality:            {has_mod}/{len(rows)} ({100*has_mod/len(rows):.1f}%)")

    # Modality breakdown
    mod_counts = {}
    for r in rows:
        m = r["metadata_modality"] or "unknown"
        mod_counts[m] = mod_counts.get(m, 0) + 1
    logger.info(f"Modality breakdown: {mod_counts}")

    # Word order breakdown
    wo_counts = {}
    for r in rows:
        wo = r["metadata_wals_word_order_label"] or "unknown"
        wo_counts[wo] = wo_counts.get(wo, 0) + 1
    logger.info(f"Word order breakdown: {wo_counts}")

    # Case proportion vs WALS category correlation (sanity check)
    if not skip_case:
        case_pairs = [(r["metadata_wals_case_category"], r["metadata_ud_case_proportion"])
                      for r in rows
                      if r["metadata_wals_case_category"] is not None and r["metadata_ud_case_proportion"] is not None]
        if len(case_pairs) > 5:
            import numpy as np
            xs = np.array([p[0] for p in case_pairs])
            ys = np.array([p[1] for p in case_pairs])
            if np.std(xs) > 0 and np.std(ys) > 0:
                corr = np.corrcoef(xs, ys)[0, 1]
                logger.info(f"Pearson r (WALS case ordinal vs UD case proportion): {corr:.3f} (n={len(case_pairs)})")

    # STEP 7: Write output in schema-compliant format
    logger.info("STEP 7: Writing data_out.json...")
    output_data = {
        "metadata": {
            "description": "Typological Classification Table for UD Treebanks (WALS + Glottolog + UD Metadata)",
            "sources": ["WALS CLDF (Features 49A, 81A)", "Glottolog CLDF", "UD GitHub READMEs", "commul/universal_dependencies (HuggingFace)"],
            "total_treebanks": len(rows),
            "coverage": {
                "wals_case_49A": f"{has_case}/{len(rows)} ({100*has_case/len(rows):.1f}%)",
                "wals_wordorder_81A": f"{has_wo}/{len(rows)} ({100*has_wo/len(rows):.1f}%)",
                "language_family": f"{has_fam}/{len(rows)} ({100*has_fam/len(rows):.1f}%)",
                "modality": f"{has_mod}/{len(rows)} ({100*has_mod/len(rows):.1f}%)",
            },
        },
        "datasets": [
            {
                "dataset": "ud_typological_classification",
                "examples": rows,
            }
        ],
    }

    output_path = WORKSPACE / "data_out.json"
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    logger.info(f"Wrote {len(rows)} rows to {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-configs", type=int, default=None, help="Limit number of configs")
    parser.add_argument("--skip-case", action="store_true", help="Skip Case proportion computation")
    parser.add_argument("--skip-readme", action="store_true", help="Skip README genre fetching")
    args = parser.parse_args()
    main(max_configs=args.max_configs, skip_case=args.skip_case, skip_readme=args.skip_readme)
