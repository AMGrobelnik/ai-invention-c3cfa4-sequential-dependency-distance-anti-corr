#!/usr/bin/env python3
"""Patch missing modality data by fetching README.txt from UD GitHub repos."""

import json
import re
import sys
import time
from pathlib import Path

import requests
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

WORKSPACE = Path(__file__).parent

# Treebank name capitalization map
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
    "imst": "IMST", "boun": "BOUN", "tourism": "Tourism",
    "kenet": "Kenet", "iu": "IU", "clasp": "CLASP",
    "btb": "BTB", "cairo": "Cairo", "prago": "Prago",
    "gnc": "GNC", "glc": "GLC", "gps": "GPS",
    "oldtudet": "OldTuDeT", "gujtb": "GujTB",
    "tuecl": "TueCL", "maibaam": "MaiBaam", "ptnk": "PTNK",
    "chibergis": "ChiBERGiS", "amgic": "AMGIC",
    "mukri": "Mukri", "ctntb": "CTNTB", "caval": "CAVAL",
    "ujaen": "UJaen",
}

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
    "xav": "Xavante", "xnr": "Kangri", "eo": "Esperanto",
    "hbo": "Ancient_Hebrew", "bar": "Bavarian", "bn": "Bengali",
    "sab": "Sabanes", "bor": "Bororo", "cpg": "Cappadocian",
    "ckb": "Central_Kurdish", "ctn": "Chatino", "ckt": "Chukchi",
    "xcl": "Classical_Armenian", "egy": "Egyptian", "ka": "Georgian",
    "aln": "Gheg", "gn": "Guarani", "gu": "Gujarati", "gwi": "Gwich_in",
    "ha": "Hausa", "hnj": "Hmong_Njua", "quc": "K_iche", "kk": "Kazakh",
    "kfm": "Khunsari", "kmr": "Kurmanji", "mag": "Magahi", "myu": "Munduruku",
    "nhi": "Nahuatl", "nyq": "Nayini", "sme": "North_Sami",
    "qpm": "Pomak", "pt": "Portuguese", "ro": "Romanian",
    "sa": "Sanskrit", "sms": "Skolt_Sami", "sl": "Slovenian",
    "soj": "Soi", "hsb": "Upper_Sorbian", "ta": "Tamil",
    "te": "Telugu", "th": "Thai", "tl": "Tagalog",
    "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu",
    "ug": "Uyghur", "vi": "Vietnamese", "wbp": "Warlpiri",
    "wo": "Wolof", "yo": "Yoruba",
}


def get_lang_name(ud_code: str) -> str:
    if ud_code in UD_LANG_MAP:
        return UD_LANG_MAP[ud_code].replace(" ", "_")
    return ud_code.capitalize()


def fetch_genre(config: str, session: requests.Session) -> tuple:
    """Try to fetch genre from UD GitHub README.md or README.txt."""
    parts = config.split("_", 1)
    lang_code = parts[0]
    tb_name = parts[1] if len(parts) > 1 else ""

    lang_name = get_lang_name(lang_code)

    # Build treebank name variants
    tb_variants = set()
    if tb_name.lower() in TB_NAME_MAP:
        tb_variants.add(TB_NAME_MAP[tb_name.lower()])
    tb_variants.add(tb_name[0].upper() + tb_name[1:] if tb_name else "")
    tb_variants.add(tb_name.upper())

    # Try each variant with README.txt first (since that's what was missing), then .md
    for tb in tb_variants:
        if not tb:
            continue
        for readme in ["README.txt", "README.md"]:
            for branch in ["master", "dev"]:
                url = f"https://raw.githubusercontent.com/UniversalDependencies/UD_{lang_name}-{tb}/{branch}/{readme}"
                try:
                    resp = session.get(url, timeout=10)
                    if resp.status_code == 200:
                        genre_match = re.search(r'Genre:\s*(.+)', resp.text)
                        if genre_match:
                            genre_tags = genre_match.group(1).strip()
                            return genre_tags, "readme_genre"
                except Exception:
                    pass

    return None, "unavailable"


def determine_modality(genre_tags):
    if not genre_tags:
        return None
    tags = genre_tags.lower().split()
    if tags == ["spoken"]:
        return "spoken"
    elif "spoken" in tags:
        return "mixed"
    else:
        return "written"


@logger.catch
def main():
    data_path = WORKSPACE / "data_out.json"
    data = json.loads(data_path.read_text())
    rows = data["datasets"][0]["examples"]

    # Find rows missing modality
    missing = [(i, r) for i, r in enumerate(rows) if not r["metadata_has_modality"]]
    logger.info(f"Found {len(missing)} treebanks missing modality")

    session = requests.Session()
    fixed = 0

    for idx, (i, row) in enumerate(missing):
        config = row["metadata_treebank_id"]
        genre_tags, source = fetch_genre(config, session)

        if genre_tags:
            modality = determine_modality(genre_tags)
            rows[i]["metadata_genre_tags"] = genre_tags
            rows[i]["metadata_modality"] = modality
            rows[i]["metadata_modality_source"] = source
            rows[i]["metadata_has_modality"] = True

            # Update output string
            parts = row["output"].split(" | ")
            if modality and modality not in parts:
                parts.append(modality)
                rows[i]["output"] = " | ".join(parts)

            fixed += 1
            logger.info(f"  [{idx+1}/{len(missing)}] {config}: {genre_tags} -> {modality}")
        else:
            logger.debug(f"  [{idx+1}/{len(missing)}] {config}: still unavailable")

        # Small delay to avoid rate limiting (every 5 requests)
        if (idx + 1) % 5 == 0:
            time.sleep(0.5)

    logger.info(f"Fixed {fixed}/{len(missing)} missing modality entries")

    # Recalculate coverage
    has_mod = sum(1 for r in rows if r["metadata_has_modality"])
    logger.info(f"New modality coverage: {has_mod}/{len(rows)} ({100*has_mod/len(rows):.1f}%)")

    # Update metadata
    data["metadata"]["coverage"]["modality"] = f"{has_mod}/{len(rows)} ({100*has_mod/len(rows):.1f}%)"

    # Write back
    data_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info(f"Updated {data_path}")


if __name__ == "__main__":
    main()
