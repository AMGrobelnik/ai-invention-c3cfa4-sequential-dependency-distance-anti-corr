#!/usr/bin/env python3
"""Standardize UD Typological Classification Table into exp_sel_data_out.json schema.

Loads pre-downloaded source CSVs from temp/datasets/ (WALS, Glottolog) and
pre-computed UD case proportions + genre metadata from data_out.json (produced
by collect_data.py + patch scripts).

Outputs full_data_out.json in exp_sel_data_out.json schema format.
"""

import json
import math
import os
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
def _container_ram_gb() -> float:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return 29.0

TOTAL_RAM_GB = _container_ram_gb()
RAM_BUDGET = int(TOTAL_RAM_GB * 0.5 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent
TEMP_DIR = WORKSPACE / "temp" / "datasets"
LOGS_DIR = WORKSPACE / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Macrolanguage → individual language fallbacks (for Glottolog matching)
MACRO_TO_INDIVIDUAL = {
    "ara": "arb", "zho": "cmn", "fas": "pes", "nor": "nob",
    "msa": "zsm", "est": "ekk", "lav": "lvs", "sqi": "als",
}

# UD language code → ISO 639-3
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
    "nb": "nob", "nn": "nno", "no": "nor", "cu": "chu", "fro": "fro",
    "orv": "orv", "otk": "otk", "fa": "fas", "pl": "pol", "pt": "por",
    "ro": "ron", "ru": "rus", "sa": "san", "gd": "gla", "sr": "srp",
    "sms": "sms", "sk": "slk", "sl": "slv", "soj": "soj", "hsb": "hsb",
    "es": "spa", "sv": "swe", "gsw": "gsw", "tl": "tgl", "ta": "tam",
    "te": "tel", "th": "tha", "tr": "tur", "uk": "ukr", "ur": "urd",
    "ug": "uig", "vi": "vie", "wbp": "wbp", "cy": "cym", "wo": "wol",
    "yo": "yor", "ess": "ess", "ab": "abk", "abq": "abq", "aqz": "aqz",
    "sq": "sqi", "bej": "bej", "koi": "koi", "kpv": "kpv", "mpu": "mpu",
    "nds": "nds", "nyq": "nyq", "swl": "swl", "tpn": "tpn", "urb": "urb",
    "xav": "xav", "xnr": "xnr", "qfn": "qfn", "qhe": "qhe", "qtd": "qtd",
    "eo": "epo", "hbo": "hbo", "bar": "bar", "bn": "ben", "sab": "sab",
    "bor": "bor", "cpg": "cpg", "ckb": "ckb", "ctn": "ctn", "ckt": "ckt",
    "xcl": "xcl", "egy": "egy", "ka": "kat", "aln": "aln", "gn": "grn",
    "gu": "guj", "gwi": "gwi", "ha": "hau", "hnj": "hnj", "quc": "quc",
    "mag": "mag", "myu": "myu", "nhi": "nhi", "qpm": "qpm",
    "ht": "hat", "azz": "azz", "arh": "arh", "arr": "arr", "naq": "naq",
    "ky": "kir", "ltg": "ltg", "lij": "lij", "lb": "ltz", "jaa": "jaa",
    "qaf": "qaf", "ml": "mal", "frm": "frm", "nmf": "nmf", "nap": "nap",
    "yrk": "yrk", "yrl": "yrl", "gya": "gya", "oc": "oci", "or": "ori",
    "ang": "ang", "sga": "sga", "pro": "pro", "ota": "ota", "ps": "pus",
    "pad": "pad", "pay": "pay", "xpg": "xpg", "wuu": "wuu", "scn": "scn",
    "sd": "snd", "si": "sin", "ajp": "ajp", "sdh": "sdh", "ssp": "ssp",
    "tt": "tat", "eme": "eme", "qte": "qte", "tn": "tsn", "qti": "qti",
    "xum": "xum", "uz": "uzb", "vep": "vep", "hyw": "hyw", "sjo": "sjo",
    "sah": "sah", "yi": "yid", "say": "say", "az": "aze",
}

# UD language code → human name
UD_LANG_MAP = {
    "af": "Afrikaans", "akk": "Akkadian", "am": "Amharic", "grc": "Ancient Greek",
    "apu": "Apurina", "ar": "Arabic", "hy": "Armenian", "aii": "Assyrian",
    "bm": "Bambara", "eu": "Basque", "be": "Belarusian", "bho": "Bhojpuri",
    "br": "Breton", "bg": "Bulgarian", "bxr": "Buryat", "yue": "Cantonese",
    "ca": "Catalan", "ceb": "Cebuano", "zh": "Chinese", "lzh": "Classical Chinese",
    "cop": "Coptic", "hr": "Croatian", "cs": "Czech", "da": "Danish",
    "nl": "Dutch", "en": "English", "myv": "Erzya", "et": "Estonian",
    "fo": "Faroese", "fi": "Finnish", "fr": "French", "gl": "Galician",
    "de": "German", "got": "Gothic", "el": "Greek", "gub": "Guajajara",
    "he": "Hebrew", "hi": "Hindi", "hit": "Hittite", "hu": "Hungarian",
    "is": "Icelandic", "id": "Indonesian", "ga": "Irish", "it": "Italian",
    "ja": "Japanese", "jv": "Javanese", "krl": "Karelian", "kk": "Kazakh",
    "kfm": "Khunsari", "ko": "Korean", "kmr": "Kurmanji", "la": "Latin",
    "lv": "Latvian", "lt": "Lithuanian", "olo": "Livvi", "mk": "Macedonian",
    "mt": "Maltese", "gv": "Manx", "mr": "Marathi", "gun": "Mbya Guarani",
    "mdf": "Moksha", "mn": "Mongolian", "pcm": "Naija", "sme": "North Sami",
    "nb": "Norwegian", "nn": "Norwegian", "no": "Norwegian",
    "cu": "Old Church Slavonic", "fro": "Old French", "orv": "Old East Slavic",
    "otk": "Old Turkish", "fa": "Persian", "pl": "Polish", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "sa": "Sanskrit", "gd": "Scottish Gaelic",
    "sr": "Serbian", "sms": "Skolt Sami", "sk": "Slovak", "sl": "Slovenian",
    "soj": "Soi", "hsb": "Upper Sorbian", "es": "Spanish", "sv": "Swedish",
    "gsw": "Swiss German", "tl": "Tagalog", "ta": "Tamil", "te": "Telugu",
    "th": "Thai", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu",
    "ug": "Uyghur", "vi": "Vietnamese", "wbp": "Warlpiri", "cy": "Welsh",
    "wo": "Wolof", "yo": "Yoruba", "ess": "Yupik", "ab": "Abkhaz",
    "abq": "Abaza", "aqz": "Akuntsu", "sq": "Albanian", "bej": "Beja",
    "eo": "Esperanto", "hbo": "Ancient Hebrew", "bar": "Bavarian",
    "bn": "Bengali", "sab": "Sabanes", "bor": "Bororo", "cpg": "Cappadocian",
    "ckb": "Central Kurdish", "ctn": "Chatino", "ckt": "Chukchi",
    "xcl": "Classical Armenian", "egy": "Egyptian", "ka": "Georgian",
    "aln": "Gheg", "gn": "Guarani", "gu": "Gujarati", "gwi": "Gwichin",
    "ha": "Hausa", "hnj": "Hmong Njua", "quc": "Kiche", "mag": "Magahi",
    "myu": "Munduruku", "nhi": "Nahuatl", "nyq": "Nayini", "qpm": "Pomak",
    "ht": "Haitian Creole", "azz": "Highland Puebla Nahuatl", "arh": "Arhuaco",
    "arr": "Karo", "naq": "Khoekhoe", "ky": "Kyrgyz", "ltg": "Latgalian",
    "lij": "Ligurian", "lb": "Luxembourgish", "jaa": "Jarawara",
    "ml": "Malayalam", "frm": "Middle French", "nmf": "Tangkhul Naga",
    "nap": "Neapolitan", "yrk": "Nenets", "yrl": "Nheengatu",
    "gya": "Northwest Gbaya", "oc": "Occitan", "or": "Odia",
    "ang": "Old English", "sga": "Old Irish", "pro": "Old Occitan",
    "ota": "Ottoman Turkish", "ps": "Pashto", "pad": "Paumari",
    "pay": "Pech", "xpg": "Phrygian", "wuu": "Shanghainese",
    "scn": "Sicilian", "sd": "Sindhi", "si": "Sinhala",
    "ajp": "South Levantine Arabic", "sdh": "Southern Kurdish",
    "ssp": "Spanish Sign Language", "tt": "Tatar", "eme": "Teko",
    "qte": "Tena Quichua", "tn": "Tswana", "qti": "Tsetsaut",
    "xum": "Umbrian", "uz": "Uzbek", "vep": "Veps",
    "hyw": "Western Armenian", "sjo": "Xibe", "sah": "Yakut",
    "yi": "Yiddish", "say": "Sayula Popoluca", "az": "Azerbaijani",
    "qfn": "Frisian Dutch", "qhe": "Hindi English", "qtd": "Turkish German",
    "swl": "Swedish Sign Language", "koi": "Komi Permyak",
    "kpv": "Komi Zyrian", "mpu": "Makurap", "nds": "Low Saxon",
    "tpn": "Tupinamba", "urb": "Kaapor", "xav": "Xavante", "xnr": "Kangri",
    "qaf": "Arabizi",
}


def build_wals_lookups() -> tuple[dict, dict, dict, dict]:
    """Build WALS case/word-order lookups from temp/datasets/ CSVs."""
    logger.info("Loading WALS CSVs from temp/datasets/")
    values = pd.read_csv(TEMP_DIR / "wals_values.csv", low_memory=False)
    langs = pd.read_csv(TEMP_DIR / "wals_languages.csv", low_memory=False)
    codes = pd.read_csv(TEMP_DIR / "wals_codes.csv", low_memory=False)

    # Feature 49A (case)
    case_vals = values[values["Parameter_ID"] == "49A"].copy()
    codes_49a = codes[codes["Parameter_ID"] == "49A"][["ID", "Name", "Number"]].rename(
        columns={"ID": "Code_ID", "Name": "case_label", "Number": "case_number"})
    case_vals = case_vals.merge(codes_49a, on="Code_ID", how="left")

    # Feature 81A (word order)
    wo_vals = values[values["Parameter_ID"] == "81A"].copy()
    codes_81a = codes[codes["Parameter_ID"] == "81A"][["ID", "Name", "Number"]].rename(
        columns={"ID": "Code_ID", "Name": "wo_label", "Number": "wo_number"})
    wo_vals = wo_vals.merge(codes_81a, on="Code_ID", how="left")

    # Join with languages
    lang_cols = [c for c in ["ID", "Name", "ISO639P3code", "Glottocode", "Family", "Genus", "Macroarea"]
                 if c in langs.columns]
    lang_df = langs[lang_cols].rename(columns={"ID": "Language_ID", "Name": "wals_lang_name"})
    case_vals = case_vals.merge(lang_df, on="Language_ID", how="left")
    wo_vals = wo_vals.merge(lang_df, on="Language_ID", how="left")

    # Build lookup dicts
    wals_case, wals_wo = {}, {}
    iso_case_seen, iso_wo_seen = {}, {}

    for _, row in case_vals.iterrows():
        iso = str(row.get("ISO639P3code", "")).strip()
        gc = str(row.get("Glottocode", "")).strip()
        entry = {
            "wals_language_id": row["Language_ID"],
            "wals_case_category": int(row["case_number"]) if pd.notna(row.get("case_number")) else None,
            "wals_case_label": row.get("case_label", ""),
            "wals_family": row.get("Family", ""), "wals_genus": row.get("Genus", ""),
            "wals_macroarea": row.get("Macroarea", ""),
        }
        if iso and iso != "nan":
            if iso not in iso_case_seen:
                iso_case_seen[iso] = 1
                wals_case[f"iso:{iso}"] = entry
            else:
                iso_case_seen[iso] += 1
        if gc and gc != "nan":
            wals_case[f"gc:{gc}"] = entry

    for _, row in wo_vals.iterrows():
        iso = str(row.get("ISO639P3code", "")).strip()
        gc = str(row.get("Glottocode", "")).strip()
        entry = {
            "wals_language_id": row["Language_ID"],
            "wals_word_order": int(row["wo_number"]) if pd.notna(row.get("wo_number")) else None,
            "wals_word_order_label": row.get("wo_label", ""),
            "wals_family": row.get("Family", ""), "wals_genus": row.get("Genus", ""),
            "wals_macroarea": row.get("Macroarea", ""),
        }
        if iso and iso != "nan":
            if iso not in iso_wo_seen:
                iso_wo_seen[iso] = 1
                wals_wo[f"iso:{iso}"] = entry
            else:
                iso_wo_seen[iso] += 1
        if gc and gc != "nan":
            wals_wo[f"gc:{gc}"] = entry

    multi_case = {k for k, v in iso_case_seen.items() if v > 1}
    multi_wo = {k for k, v in iso_wo_seen.items() if v > 1}
    logger.info(f"WALS case: {len(wals_case)} entries | word-order: {len(wals_wo)} entries")
    return wals_case, wals_wo, multi_case, multi_wo


def build_glottolog_lookup() -> dict:
    """Build Glottolog family lookup from temp/datasets/ CSV."""
    logger.info("Loading Glottolog CSV from temp/datasets/")
    glot = pd.read_csv(TEMP_DIR / "glottolog_languages.csv", low_memory=False)
    families = dict(zip(
        glot[glot["Level"] == "family"]["Glottocode"],
        glot[glot["Level"] == "family"]["Name"]))
    languages = glot[glot["Level"] == "language"]

    lookup = {}
    for _, row in languages.iterrows():
        iso = str(row.get("ISO639P3code", "")).strip()
        gc = str(row.get("Glottocode", "")).strip()
        fam_id = str(row.get("Family_ID", "")).strip()
        is_isolate = str(row.get("Is_Isolate", "")).strip().lower() == "true"
        family_name = (f"{row['Name']} (isolate)" if is_isolate
                       else families.get(fam_id, "") if fam_id and fam_id != "nan"
                       else "")
        entry = {"glottocode": gc, "family_name": family_name,
                 "macroarea": row.get("Macroarea", ""), "is_isolate": is_isolate}
        if iso and iso != "nan":
            lookup[f"iso:{iso}"] = entry
        if gc and gc != "nan":
            lookup[f"gc:{gc}"] = entry

    logger.info(f"Glottolog: {len(lookup)} entries")
    return lookup


def load_precomputed_hf_data() -> dict:
    """Load pre-computed case proportions and genre data from data_out.json."""
    src = WORKSPACE / "data_out.json"
    logger.info(f"Loading pre-computed HF data from {src}")
    data = json.loads(src.read_text())
    rows = data["datasets"][0]["examples"]
    cache = {}
    for r in rows:
        config = r["metadata_treebank_id"]
        cache[config] = {
            "ud_case_proportion": r.get("metadata_ud_case_proportion"),
            "ud_total_tokens": r.get("metadata_ud_total_tokens", 0),
            "ud_case_tokens": r.get("metadata_ud_case_tokens", 0),
            "genre_tags": r.get("metadata_genre_tags"),
            "modality": r.get("metadata_modality"),
            "modality_source": r.get("metadata_modality_source", "unavailable"),
        }
    logger.info(f"Loaded pre-computed data for {len(cache)} configs")
    return cache


def get_iso3(ud_code: str) -> tuple[str, bool]:
    if ud_code in UD_ISO_MAP:
        iso = UD_ISO_MAP[ud_code]
        return iso, iso.startswith("q")
    return ud_code, len(ud_code) != 3 or ud_code.startswith("q")


def get_lang_name(ud_code: str) -> str:
    return UD_LANG_MAP.get(ud_code, ud_code)


def determine_modality(genre_tags: str | None) -> str | None:
    if not genre_tags:
        return None
    tags = genre_tags.lower().split()
    if tags == ["spoken"]:
        return "spoken"
    if "spoken" in tags:
        return "mixed"
    return "written"


@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("Building UD Typological Classification Table")
    logger.info("=" * 60)

    # Load source data from temp/datasets/
    wals_case, wals_wo, multi_case, multi_wo = build_wals_lookups()
    glottolog = build_glottolog_lookup()
    hf_cache = load_precomputed_hf_data()

    configs = sorted(hf_cache.keys())
    logger.info(f"Processing {len(configs)} UD treebank configs")

    rows = []
    for row_idx, config in enumerate(configs):
        lang_code = config.split("_", 1)[0]
        iso3, nonstandard = get_iso3(lang_code)
        lang_name = get_lang_name(lang_code)

        # ISO candidates for macrolanguage fallback
        iso_candidates = [iso3]
        if iso3 in MACRO_TO_INDIVIDUAL:
            iso_candidates.append(MACRO_TO_INDIVIDUAL[iso3])

        # WALS case
        case_entry, case_quality = None, "none"
        for iso_try in iso_candidates:
            if f"iso:{iso_try}" in wals_case:
                case_entry = wals_case[f"iso:{iso_try}"]
                case_quality = "exact_iso"
                break
        if not case_entry:
            for iso_try in iso_candidates:
                gc_info = glottolog.get(f"iso:{iso_try}", {})
                gc = gc_info.get("glottocode", "")
                if gc and f"gc:{gc}" in wals_case:
                    case_entry = wals_case[f"gc:{gc}"]
                    case_quality = "glottocode_fallback"
                    break

        # WALS word order
        wo_entry, wo_quality = None, "none"
        for iso_try in iso_candidates:
            if f"iso:{iso_try}" in wals_wo:
                wo_entry = wals_wo[f"iso:{iso_try}"]
                wo_quality = "exact_iso"
                break
        if not wo_entry:
            for iso_try in iso_candidates:
                gc_info = glottolog.get(f"iso:{iso_try}", {})
                gc = gc_info.get("glottocode", "")
                if gc and f"gc:{gc}" in wals_wo:
                    wo_entry = wals_wo[f"gc:{gc}"]
                    wo_quality = "glottocode_fallback"
                    break

        # Family: WALS first, then Glottolog
        family, genus, macroarea, is_isolate, family_quality = "", "", "", False, "none"
        wals_fam = (case_entry or wo_entry or {}).get("wals_family", "")
        if wals_fam:
            family = wals_fam
            genus = (case_entry or wo_entry or {}).get("wals_genus", "")
            macroarea = (case_entry or wo_entry or {}).get("wals_macroarea", "")
            family_quality = "exact_iso" if "exact_iso" in (case_quality, wo_quality) else "glottocode_fallback"
        else:
            for iso_try in iso_candidates:
                gc_info = glottolog.get(f"iso:{iso_try}", {})
                if gc_info and gc_info.get("family_name"):
                    family = gc_info["family_name"]
                    macroarea = gc_info.get("macroarea", "")
                    is_isolate = gc_info.get("is_isolate", False)
                    family_quality = "exact_iso"
                    break

        # Glottocode
        glottocode = ""
        for iso_try in iso_candidates:
            gc_info = glottolog.get(f"iso:{iso_try}", {})
            if gc_info:
                glottocode = gc_info.get("glottocode", "")
                break

        # HF pre-computed data
        hf = hf_cache.get(config, {})
        genre_tags = hf.get("genre_tags")
        modality = hf.get("modality") or determine_modality(genre_tags)
        modality_source = hf.get("modality_source", "unavailable")
        ud_case_prop = hf.get("ud_case_proportion")
        ud_total = hf.get("ud_total_tokens", 0)
        ud_case_tok = hf.get("ud_case_tokens", 0)

        # Build output summary
        parts = [lang_name]
        if wo_entry:
            parts.append(wo_entry.get("wals_word_order_label", ""))
        if case_entry:
            parts.append(case_entry.get("wals_case_label", ""))
        if family:
            parts.append(family)
        if modality:
            parts.append(modality)
        output_str = " | ".join(p for p in parts if p)

        rows.append({
            "input": config,
            "output": output_str,
            "metadata_fold": "all",
            "metadata_row_index": row_idx,
            "metadata_treebank_id": config,
            "metadata_language_name": lang_name,
            "metadata_iso_639_3": iso3,
            "metadata_glottocode": glottocode or None,
            "metadata_wals_language_id": (case_entry or wo_entry or {}).get("wals_language_id") or None,
            "metadata_wals_case_category": case_entry["wals_case_category"] if case_entry else None,
            "metadata_wals_case_label": case_entry["wals_case_label"] if case_entry else None,
            "metadata_wals_word_order": wo_entry["wals_word_order"] if wo_entry else None,
            "metadata_wals_word_order_label": wo_entry["wals_word_order_label"] if wo_entry else None,
            "metadata_language_family": family or None,
            "metadata_language_genus": genus or None,
            "metadata_macroarea": macroarea or None,
            "metadata_is_isolate": is_isolate,
            "metadata_modality": modality,
            "metadata_genre_tags": genre_tags,
            "metadata_ud_case_proportion": round(ud_case_prop, 6) if ud_case_prop is not None else None,
            "metadata_ud_total_tokens": ud_total,
            "metadata_ud_case_tokens": ud_case_tok,
            "metadata_mapping_quality_case": case_quality,
            "metadata_mapping_quality_wordorder": wo_quality,
            "metadata_mapping_quality_family": family_quality,
            "metadata_modality_source": modality_source,
            "metadata_has_wals_case": case_entry is not None,
            "metadata_has_wals_wordorder": wo_entry is not None,
            "metadata_has_family": bool(family),
            "metadata_has_modality": modality is not None,
            "metadata_nonstandard_code": nonstandard,
            "metadata_multiple_wals_matches": iso3 in multi_case or iso3 in multi_wo,
        })

    # Coverage statistics
    n = len(rows)
    has_case = sum(1 for r in rows if r["metadata_has_wals_case"])
    has_wo = sum(1 for r in rows if r["metadata_has_wals_wordorder"])
    has_fam = sum(1 for r in rows if r["metadata_has_family"])
    has_mod = sum(1 for r in rows if r["metadata_has_modality"])
    logger.info(f"Total treebanks: {n}")
    logger.info(f"WALS case (49A):      {has_case}/{n} ({100*has_case/n:.1f}%)")
    logger.info(f"WALS word order (81A): {has_wo}/{n} ({100*has_wo/n:.1f}%)")
    logger.info(f"Language family:      {has_fam}/{n} ({100*has_fam/n:.1f}%)")
    logger.info(f"Modality:             {has_mod}/{n} ({100*has_mod/n:.1f}%)")

    # Pearson r sanity check
    pairs = [(r["metadata_wals_case_category"], r["metadata_ud_case_proportion"])
             for r in rows
             if r["metadata_wals_case_category"] is not None and r["metadata_ud_case_proportion"] is not None]
    if len(pairs) > 5:
        xs, ys = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        if np.std(xs) > 0 and np.std(ys) > 0:
            r_val = np.corrcoef(xs, ys)[0, 1]
            logger.info(f"Pearson r (WALS case vs UD case proportion): {r_val:.3f} (n={len(pairs)})")

    # Write output
    output = {
        "metadata": {
            "description": "Typological Classification Table for UD Treebanks (WALS + Glottolog + UD Metadata)",
            "sources": [
                "WALS CLDF (Features 49A, 81A)",
                "Glottolog CLDF",
                "UD GitHub READMEs",
                "commul/universal_dependencies (HuggingFace)",
            ],
            "total_treebanks": n,
            "coverage": {
                "wals_case_49A": f"{has_case}/{n} ({100*has_case/n:.1f}%)",
                "wals_wordorder_81A": f"{has_wo}/{n} ({100*has_wo/n:.1f}%)",
                "language_family": f"{has_fam}/{n} ({100*has_fam/n:.1f}%)",
                "modality": f"{has_mod}/{n} ({100*has_mod/n:.1f}%)",
            },
        },
        "datasets": [{
            "dataset": "ud_typological_classification",
            "examples": rows,
        }],
    }

    out_path = WORKSPACE / "full_data_out.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info(f"Wrote {n} rows to {out_path} ({out_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
