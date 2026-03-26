#!/usr/bin/env python3
"""Second pass: fix remaining modality gaps using GitHub API to find exact repo names."""

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

# Extended language name mapping for remaining codes
EXTENDED_LANG_MAP = {
    "gsw": "Swiss_German", "az": "Azerbaijani", "sab": "Saban",
    "ctn": "Chatino", "gwi": "Gwichin", "ht": "Haitian_Creole",
    "azz": "Highland_Puebla_Nahuatl", "arh": "Arhuaco",
    "arr": "Karo", "naq": "Khoekhoe", "quc": "Kiche",
    "ky": "Kyrgyz", "ltg": "Latgalian", "lij": "Ligurian",
    "lb": "Luxembourgish", "jaa": "Jarawara", "qaf": "Arabizi",
    "ml": "Malayalam", "frm": "Middle_French", "nmf": "Tangkhul_Naga",
    "nap": "Neapolitan", "yrk": "Nenets", "yrl": "Nheengatu",
    "kmr": "Kurmanji", "gya": "Northwest_Gbaya", "oc": "Occitan",
    "or": "Odia", "ang": "Old_English", "sga": "Old_Irish",
    "pro": "Old_Occitan", "ota": "Ottoman_Turkish", "ps": "Pashto",
    "pad": "Paumari", "pay": "Pech", "xpg": "Phrygian",
    "wuu": "Shanghainese", "scn": "Sicilian", "sd": "Sindhi",
    "si": "Sinhala", "ajp": "South_Levantine_Arabic",
    "sdh": "Southern_Kurdish", "ssp": "Spanish_Sign_Language",
    "tt": "Tatar", "eme": "Teko", "qte": "Tena_Quichua",
    "tn": "Tswana", "qti": "Tsetsaut", "xum": "Umbrian",
    "uz": "Uzbek", "vep": "Veps", "hyw": "Western_Armenian",
    "nhi": "Western_Sierra_Puebla_Nahuatl", "sjo": "Xibe",
    "sah": "Yakut", "yi": "Yiddish", "say": "Sayula_Popoluca",
}

# Additional treebank name caps
EXTRA_TB_MAP = {
    "adolphe": "Adolphe", "autogramm": "Autogramm", "itml": "ITML",
    "chibergis": "ChiBERGiS", "kdt": "KDT", "ktmu": "KTMU",
    "tuecl": "TueCL", "glt": "GLT", "luxbank": "LuxBank",
    "jarawara": "Jarawara", "arabizi": "Arabizi", "altm": "ALTM",
    "profiterole": "Profiterole", "suansu": "SuanSu", "rb": "RB",
    "tundra": "Tundra", "complin": "CompLin", "kurmanji": "Kurmanji",
    "ttb": "TTB", "odtb": "ODTB", "dipsgg": "DipSGG",
    "dipwbg": "DipWBG", "corag": "CoRaG", "dudu": "Dudu",
    "sikaram": "Sikaram", "kul": "KUL", "shud": "SHUD",
    "stb": "STB", "isra": "ISRA", "madar": "MADAR",
    "garrusi": "Garrusi", "lse": "LSE", "nmctt": "NMCTT",
    "tect": "TECT", "popapolelo": "PopaPolelo", "butr": "BUTR",
    "ikuvina": "Ikuvina", "ut": "UT", "uzudt": "UZUDT",
    "vwt": "VWT", "armtdp": "ArmTDP", "xdt": "XDT",
    "yktdt": "YKTDT", "yitb": "YITB",
}


def fetch_genre(lang_name: str, tb_name: str, session: requests.Session) -> tuple:
    """Try to fetch genre from UD GitHub README."""
    tb_variants = set()
    tb_low = tb_name.lower()
    if tb_low in EXTRA_TB_MAP:
        tb_variants.add(EXTRA_TB_MAP[tb_low])
    tb_variants.add(tb_name[0].upper() + tb_name[1:] if tb_name else "")
    tb_variants.add(tb_name.upper())

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
                            return genre_match.group(1).strip(), "readme_genre"
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
    return "written"


@logger.catch
def main():
    data_path = WORKSPACE / "data_out.json"
    data = json.loads(data_path.read_text())
    rows = data["datasets"][0]["examples"]

    missing = [(i, r) for i, r in enumerate(rows) if not r["metadata_has_modality"]]
    logger.info(f"Found {len(missing)} treebanks still missing modality")

    session = requests.Session()
    fixed = 0

    for idx, (i, row) in enumerate(missing):
        config = row["metadata_treebank_id"]
        parts = config.split("_", 1)
        lang_code = parts[0]
        tb_name = parts[1] if len(parts) > 1 else ""

        lang_name = EXTENDED_LANG_MAP.get(lang_code, lang_code.capitalize())

        genre_tags, source = fetch_genre(lang_name, tb_name, session)

        if genre_tags:
            modality = determine_modality(genre_tags)
            rows[i]["metadata_genre_tags"] = genre_tags
            rows[i]["metadata_modality"] = modality
            rows[i]["metadata_modality_source"] = source
            rows[i]["metadata_has_modality"] = True

            parts_out = row["output"].split(" | ")
            if modality and modality not in parts_out:
                parts_out.append(modality)
                rows[i]["output"] = " | ".join(parts_out)

            fixed += 1
            logger.info(f"  [{idx+1}/{len(missing)}] {config}: {genre_tags} -> {modality}")

        if (idx + 1) % 5 == 0:
            time.sleep(0.3)

    logger.info(f"Fixed {fixed}/{len(missing)} more missing modality entries")

    has_mod = sum(1 for r in rows if r["metadata_has_modality"])
    logger.info(f"Final modality coverage: {has_mod}/{len(rows)} ({100*has_mod/len(rows):.1f}%)")

    data["metadata"]["coverage"]["modality"] = f"{has_mod}/{len(rows)} ({100*has_mod/len(rows):.1f}%)"
    data_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info(f"Updated {data_path}")


if __name__ == "__main__":
    main()
