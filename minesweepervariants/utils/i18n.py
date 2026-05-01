#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import gettext
import locale
import struct
import ast
from pathlib import Path


DOMAIN = "minesweepervariants"
LOCALE_DIR = Path(__file__).resolve().parents[1] / "locale"


def _split_language(language: str | None) -> list[str]:
    if not language:
        return []
    languages = [language]
    if "_" in language:
        languages.append(language.split("_", 1)[0])
    return languages


def _po_path(language: str) -> Path:
    return LOCALE_DIR / language / "LC_MESSAGES" / f"{DOMAIN}.po"


def _mo_path(language: str) -> Path:
    return LOCALE_DIR / language / "LC_MESSAGES" / f"{DOMAIN}.mo"


def _parse_po_string(value: str) -> str:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        return ast.literal_eval(value)
    return value


def _compile_po_to_mo(po_path: Path, mo_path: Path):
    if not po_path.exists():
        return False

    catalog: dict[str, str] = {}
    msgid = None
    msgstr = None
    state = None

    def flush_entry():
        nonlocal msgid, msgstr
        if msgid is not None and msgstr is not None:
            catalog[msgid] = msgstr
        msgid = None
        msgstr = None

    for raw_line in po_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            if not line:
                flush_entry()
            continue
        if line.startswith("msgid "):
            flush_entry()
            msgid = _parse_po_string(line[6:])
            state = "msgid"
            continue
        if line.startswith("msgstr "):
            msgstr = _parse_po_string(line[7:])
            state = "msgstr"
            continue
        if line.startswith('"'):
            chunk = _parse_po_string(line)
            if state == "msgid" and msgid is not None:
                msgid += chunk
            elif state == "msgstr" and msgstr is not None:
                msgstr += chunk

    flush_entry()

    items = sorted(catalog.items())
    ids = []
    strs = []
    for msgid_text, msgstr_text in items:
        ids.append(msgid_text.encode("utf-8") + b"\0")
        strs.append(msgstr_text.encode("utf-8") + b"\0")

    ids_blob = b"".join(ids)
    strs_blob = b"".join(strs)
    count = len(items)
    keystart = 7 * 4 + 16 * count
    valuestart = keystart + len(ids_blob)

    koffsets = []
    voffsets = []
    pointer = 0
    for data in ids:
        koffsets.append((len(data) - 1, keystart + pointer))
        pointer += len(data)
    pointer = 0
    for data in strs:
        voffsets.append((len(data) - 1, valuestart + pointer))
        pointer += len(data)

    mo_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        struct.pack("Iiiiiii", 0x950412de, 0, count, 7 * 4, 7 * 4 + count * 8, 0, 0)
    ]
    for length, offset in koffsets:
        payload.append(struct.pack("II", length, offset))
    for length, offset in voffsets:
        payload.append(struct.pack("II", length, offset))
    payload.extend([ids_blob, strs_blob])
    mo_path.write_bytes(b"".join(payload))
    return True


def _ensure_catalog(language: str):
    po_path = _po_path(language)
    mo_path = _mo_path(language)
    if not mo_path.exists() or (po_path.exists() and po_path.stat().st_mtime > mo_path.stat().st_mtime):
        _compile_po_to_mo(po_path, mo_path)


def _load_translation(languages: list[str]):
    active_languages = []
    for language in languages:
        _ensure_catalog(language)
        active_languages.append(language)
    if not active_languages:
        return gettext.translation(DOMAIN, localedir=str(LOCALE_DIR), fallback=True)
    return gettext.translation(DOMAIN, localedir=str(LOCALE_DIR), languages=active_languages, fallback=True)


def init_gettext(language: str | None = None):
    if language:
        try:
            locale.setlocale(locale.LC_ALL, language)
        except locale.Error:
            pass

    current_locale = locale.getlocale()[0]
    requested_languages = _split_language(language)
    fallback_languages = ["en_US", "en"] if language else []

    translation = _load_translation(requested_languages or _split_language(current_locale) or fallback_languages)
    if type(translation) is gettext.NullTranslations and language:
        translation = _load_translation(["en_US", "en"])

    translation.install()
    return translation.gettext
