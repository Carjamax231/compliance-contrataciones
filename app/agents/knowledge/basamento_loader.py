"""Índice y recuperación de extractos del basamento legal (MD locales)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.agents.knowledge.storage import leer_texto, listar_basamento_rel_paths

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BASAMENTO_DIR = _REPO_ROOT / "basamento-legal"
_LIMITE_DEFAULT = 4_000
_CHUNK_CHARS = 1_200
_EXTRACTO_MAX_CHARS = 900
_MAX_ARTICULOS = 4

# Solo cabeceras de artículo (inicio de línea), no menciones en el cuerpo.
_ART_HEADER_RE = re.compile(
    r"(?m)^(?:#{1,4}\s+)?\*{0,2}\s*Art(?:[íi]culo|\.)\s+"
    r"(?P<num>\d+[°º]?(?:\.\d+)*)\b[^\n]*",
    re.IGNORECASE,
)

_INST = r"LCP|RLCP|LOPA|LOCGR|LOJCA|SUNAI|LCC|LOPJ|LOAF|LOAP|LCCP"
# Citas: Art. 19 LOPA | artículos 101 y 102 LCP | 91.1 LOCGR | art. 98 del RLCP
_CITA_RE = re.compile(
    rf"(?:art(?:[íi]culo)?s?\.?\s+)"
    rf"(?P<nums>\d{{1,3}}(?:\.\d+)?(?:\s*(?:[,y]|y)\s*\d{{1,3}}(?:\.\d+)?)*)"
    rf"(?:\s+(?:de(?:l)?|de\s+la|de\s+las))?"
    rf"\s+"
    rf"(?P<inst>{_INST})",
    re.IGNORECASE,
)
_CITA_RE_BARE = re.compile(
    rf"\b(?P<nums>\d{{1,3}}(?:\.\d+)?)\s+(?P<inst>{_INST})\b",
    re.IGNORECASE,
)
_CITA_RE_REV = re.compile(
    rf"(?P<inst>{_INST})"
    rf"\s*(?:art(?:[íi]culo)?s?\.?\s*)?(?P<nums>\d{{1,3}}(?:\.\d+)?(?:\s*(?:[,y]|y)\s*\d{{1,3}}(?:\.\d+)?)*)",
    re.IGNORECASE,
)

_INSTRUMENTOS = (
    "LCP",
    "RLCP",
    "LOPA",
    "LOCGR",
    "LOJCA",
    "SUNAI",
    "LCC",
    "LOPJ",
    "LOAF",
    "LOAP",
    "LCCP",
)

# filename (lower) substring → código. Más específico primero (RLCP antes que LCP).
_FILE_ALIAS: tuple[tuple[str, str], ...] = (
    ("reglamento de la ley de contrataciones", "RLCP"),
    ("procedimientos administrativos", "LOPA"),
    ("l.o.p.a", "LOPA"),
    ("lopa", "LOPA"),
    ("contraloría general", "LOCGR"),
    ("contraloria general", "LOCGR"),
    ("control fiscal", "LOCGR"),
    ("jurisdiccion.contenciosa", "LOJCA"),
    ("jurisdicción contenciosa", "LOJCA"),
    ("normas_de_control_interno", "SUNAI"),
    ("control interno aplicables", "SUNAI"),
    ("contra la corrupción", "LCC"),
    ("contra la corrupcion", "LCC"),
    ("precios.justos", "LOPJ"),
    ("precios justos", "LOPJ"),
    ("administracion_financiera", "LOAF"),
    ("administración financiera", "LOAF"),
    ("administracion_publica", "LOAP"),
    ("administración pública", "LOAP"),
    ("guerra_economica", "LCCP"),
    ("guerra economica", "LCCP"),
    ("legislacion.contrataciones.publicas", "LCP"),
    ("contrataciones.publicas", "LCP"),
    ("contrataciones públicas", "LCP"),
)


@dataclass(frozen=True)
class Extracto:
    instrumento: str
    articulo: str | None
    fuente: str
    texto: str


def _norm_num(raw: str) -> str:
    t = (raw or "").strip().replace("°", "").replace("º", "")
    return t


def _instrumento_de_path(path: Path) -> str:
    name = path.name.lower()
    stem = path.stem.lower()
    blob = f"{name} {stem}"
    for needle, code in _FILE_ALIAS:
        if needle in blob:
            return code
    return path.stem[:40]


def _split_articulos(markdown: str, fuente: str, instrumento: str) -> list[Extracto]:
    matches = list(_ART_HEADER_RE.finditer(markdown))
    if not matches:
        return []
    out: list[Extracto] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        bloque = markdown[start:end].strip()
        if len(bloque) < 20:
            continue
        num = _norm_num(m.group("num"))
        # Recorte por artículo para no inflar el índice
        if len(bloque) > 4_000:
            bloque = bloque[:4_000].rstrip() + "…"
        out.append(
            Extracto(
                instrumento=instrumento,
                articulo=num,
                fuente=fuente,
                texto=bloque,
            )
        )
    return out


def _chunks_libres(markdown: str, fuente: str, instrumento: str) -> list[Extracto]:
    partes: list[str] = []
    por_h = re.split(r"\n(?=#{1,3}\s)", markdown)
    for p in por_h:
        p = p.strip()
        if len(p) < 80:
            continue
        if len(p) <= _CHUNK_CHARS:
            partes.append(p)
            continue
        for i in range(0, len(p), _CHUNK_CHARS):
            partes.append(p[i : i + _CHUNK_CHARS])
    out: list[Extracto] = []
    for i, t in enumerate(partes, start=1):
        out.append(
            Extracto(
                instrumento=instrumento,
                articulo=None,
                fuente=f"{fuente}#c{i}",
                texto=t.strip(),
            )
        )
    return out


@lru_cache(maxsize=1)
def _indice() -> tuple[Extracto, ...]:
    rels = listar_basamento_rel_paths(_BASAMENTO_DIR)
    if not rels:
        logger.warning("No hay MD de basamento (local/GCS)")
        return tuple()
    items: list[Extracto] = []
    for rel in rels:
        path = _BASAMENTO_DIR / rel
        text = leer_texto(local_path=path, gcs_blob=rel, kind="basamento")
        if not text:
            continue
        inst = _instrumento_de_path(path)
        arts = _split_articulos(text, rel, inst)
        if arts:
            items.extend(arts)
        else:
            items.extend(_chunks_libres(text, rel, inst))
    logger.info("Basamento legal: %s extractos indexados", len(items))
    return tuple(items)


def catalogo_fuentes() -> str:
    idx = _indice()
    if not idx:
        return "(KB normativa vacía)"
    insts = sorted({ex.instrumento for ex in idx if ex.instrumento in _INSTRUMENTOS})
    n_md = len({ex.fuente.split("#", 1)[0] for ex in idx})
    extra = "doctrina-administrativa, sentencias"
    inst_txt = ", ".join(insts) if insts else extra
    return f"Fuentes en KB: {n_md} archivos MD ({inst_txt}; {extra})."


def extraer_citas(texto: str) -> list[tuple[str, str]]:
    """Devuelve pares (instrumento, numero_articulo)."""
    blob = texto or ""
    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for rx in (_CITA_RE, _CITA_RE_REV, _CITA_RE_BARE):
        for m in rx.finditer(blob):
            inst = m.group("inst").upper()
            nums_raw = m.group("nums") or ""
            for n in re.findall(r"\d+(?:\.\d+)?", nums_raw):
                key = (inst, n)
                if key not in seen:
                    seen.add(key)
                    found.append(key)
    return found


def _busca_articulo(inst: str, num: str) -> Extracto | None:
    idx = _indice()
    inst_u = inst.upper()
    num_n = _norm_num(num)
    for ex in idx:
        if ex.instrumento.upper() == inst_u and ex.articulo == num_n:
            return ex
    # 91.1 → 91
    if "." in num_n:
        padre = num_n.split(".", 1)[0]
        for ex in idx:
            if ex.instrumento.upper() == inst_u and ex.articulo == padre:
                return ex
    return None


def _score_keyword(ex: Extracto, tokens: list[str]) -> int:
    blob = (ex.texto + " " + ex.fuente + " " + ex.instrumento).lower()
    return sum(1 for t in tokens if t in blob)


def retrieve_extractos(
    query: str,
    *,
    articulos: list[tuple[str, str]] | None = None,
    limite_chars: int = _LIMITE_DEFAULT,
    permitir_keywords: bool = False,
) -> str:
    """Arma un bloque de citas textuales acotado para el prompt.

    Por defecto SOLO artículos citados (Art. 19 LOPA). El fallback por
    keywords inflaba el prompt y dejaba al modelo pensando.
    """
    idx = _indice()
    if not idx:
        return (
            "BASAMENTO LEGAL RECUPERADO: (vacío — no hay MD en basamento-legal/). "
            "No inventes artículos."
        )

    citas = list(articulos or [])
    citas.extend(extraer_citas(query or ""))
    seen_c: set[tuple[str, str]] = set()
    uniq: list[tuple[str, str]] = []
    for inst, num in citas:
        k = (inst.upper(), _norm_num(num))
        if k not in seen_c:
            seen_c.add(k)
            uniq.append(k)
    uniq = uniq[:_MAX_ARTICULOS]

    picked: list[Extracto] = []
    used_ids: set[int] = set()
    faltantes: list[str] = []
    for inst, num in uniq:
        hit = _busca_articulo(inst, num)
        if hit is not None and id(hit) not in used_ids:
            picked.append(hit)
            used_ids.add(id(hit))
        elif hit is None:
            faltantes.append(f"{inst} art. {num}")

    if permitir_keywords and not uniq:
        tokens = [
            t
            for t in re.findall(r"[a-záéíóúñ]{4,}", (query or "").lower())
            if t
            not in {
                "este",
                "esta",
                "para",
                "como",
                "debe",
                "sobre",
                "cuando",
                "donde",
                "cual",
                "artículo",
                "articulo",
                "según",
                "segun",
                "dice",
                "texto",
                "fundamento",
                "pendiente",
                "revision",
                "revisión",
                "juridica",
                "jurídica",
                "informe",
            }
        ]
        if tokens:
            ranked = sorted(idx, key=lambda ex: _score_keyword(ex, tokens), reverse=True)
            for ex in ranked:
                if id(ex) in used_ids:
                    continue
                if _score_keyword(ex, tokens) < 2:
                    break
                picked.append(ex)
                used_ids.add(id(ex))
                if len(picked) >= 2:
                    break

    lineas = [
        "BASAMENTO LEGAL RECUPERADO (citar SOLO lo que esté aquí o en el "
        "cuestionario; si falta el artículo, marca fundamento pendiente; "
        "no inventes numeración):",
        catalogo_fuentes(),
        "",
    ]
    usados = 0
    for ex in picked:
        etiq = f"[{ex.instrumento}"
        if ex.articulo:
            etiq += f" art. {ex.articulo}"
        etiq += f" — {ex.fuente}]"
        cuerpo = ex.texto.strip()
        if len(cuerpo) > _EXTRACTO_MAX_CHARS:
            cuerpo = cuerpo[:_EXTRACTO_MAX_CHARS].rstrip() + "…"
        bloque = f"{etiq}\n{cuerpo}\n"
        if usados + len(bloque) > limite_chars:
            break
        lineas.append(bloque)
        usados += len(bloque)
    if faltantes:
        lineas.append(
            "No están en la KB (fundamento pendiente, no inventar): "
            + "; ".join(faltantes)
            + "."
        )
    if usados == 0 and not faltantes:
        lineas.append(
            "(No se localizó un artículo exacto para esta consulta; "
            "usa el cuestionario si trae fundamento y no fabriques citas.)"
        )
    return "\n".join(lineas)


def bloque_basamento(*textos: str, limite_chars: int = _LIMITE_DEFAULT) -> str:
    """Atajo: junta textos (mensaje, fundamentos) y recupera extractos citados."""
    query = "\n".join(t for t in textos if t)
    return retrieve_extractos(query, limite_chars=limite_chars)
