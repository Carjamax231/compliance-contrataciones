"""Carga cuestionarios oficiales desde archivos .md por modalidad × tipo de documento."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.agents.knowledge.storage import leer_texto
from app.models.schemas import Modalidad, TipoDocumento

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CUESTIONARIOS_DIR = _REPO_ROOT / "cuestionarios"

# Códigos de rúbrica CA… (Apertura Única) y CD… (Contratación Directa).
# Variantes: **CAAUAP.1**¿ | CDAAP. 1¿ | **CDAIOC.2** ¿ | CDACTO.4 (BIENES) ¿
_ITEM_RE = re.compile(
    r"(?:\*\*)?"
    r"(?P<code>(?:CA|CD)[A-ZÁÉÍÓÚÑ]{2,}[A-Z0-9]*)"
    r"\.?\s*"
    r"(?P<num_in>\d+(?:\.\d+)?)?"
    r"\s*:?\s*"
    r"(?:\\?[.\-])*\s*"
    r"(?P<pre_close>[^*\n]*?)"
    r"(?:\*\*)?"
    r"\s*"
    r"(?:\\?[.\-])*\s*"
    r"(?P<num_out>\d+(?:\.\d+)?)?"
    r"\s*:?\s*"
    r"(?:\\?[.\-])*\s*"
    r"(?P<body>[^\n]*\?)",
    re.MULTILINE,
)

_CRIT_RE = re.compile(
    r"(?:RANGO\s*S?\s*DE\s+CRITICIDAD|SIRITICIDAD|CRITICIDAD|"
    r"Ponderaci[oó]n)\s*:\s*(?P<crit>[^\n*]+)",
    re.IGNORECASE,
)
_ACCION_RE = re.compile(
    r"ACCI[ÓO]N\s+LEGAL\s*:\s*(?P<val>.+?)"
    r"(?=\n\s*(?:\*\*)?ADVERTENCIA|\*\*\s*ADVERTENCIA|\n\s*\*\*[A-Z]|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_ADV_RE = re.compile(
    r"ADVERTENCIA(?:\s+(?:PARA\s+LA\s+|A\s+LA\s+)?GERENCIA)?\s*:\s*(?P<val>.+?)"
    r"(?=\n\s*(?:\*\*)?(?:CA|CD)[A-ZÁÉÍÓÚÑ]{2,}|\n\s*\*\*[A-ZÁÉÍÓÚÑ]{3,}|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_ART_RE = re.compile(
    r"(?:\*\*)?(?:Art[ií]culos?|Base\s+Legal|ART[ÍI]CULOS)\*?\*?\s*:?\s*"
    r"(?P<val>.+?)"
    r"(?=\n\s*(?:\*\*)?(?:RANGO|CRITICIDAD|Ponderaci[oó]n|Respuesta))",
    re.IGNORECASE | re.DOTALL,
)
# Formato detallado: texto íntegro de artículos bajo el bloque de fundamento.
_FUND_DETALLADO_RE = re.compile(
    r"FUNDAMENTO\s+LEGAL\s+(?:VIOLENTADO|APLICABLE)"
    r"(?:\s+EN\s+CASO\s+DE\s+NO)?\s*:?\s*"
    r"(?P<val>.+)",
    re.IGNORECASE | re.DOTALL,
)
_TIPO_PREF_RE = re.compile(
    r"^\(\s*(BIENES|OBRAS|SERVICIOS|URGENCIA)\s*\)\s*",
    re.IGNORECASE,
)

# Tope de fundamento en prompt compacto (antes hasta 1800 → hinchaba lotes).
_FUNDAMENTO_COMPACTO_MAX = 280


@dataclass(frozen=True)
class ItemCuestionario:
    codigo: str
    texto: str
    fundamento_legal: str | None = None
    rango_criticidad: str | None = None
    accion_legal: str | None = None
    advertencia_gerencia: str | None = None


@dataclass(frozen=True)
class CuestionarioDocumento:
    modalidad: Modalidad
    tipo_documento: TipoDocumento
    path: Path
    markdown: str
    items: tuple[ItemCuestionario, ...]


def ruta_cuestionario(modalidad: Modalidad, tipo: TipoDocumento) -> Path:
    return _CUESTIONARIOS_DIR / modalidad.value / f"{tipo.value}.md"


def existe_cuestionario(modalidad: Modalidad, tipo: TipoDocumento) -> bool:
    return cargar_cuestionario(modalidad, tipo) is not None


def _limpiar(txt: str) -> str:
    t = re.sub(r"\s+", " ", txt or "").strip()
    return t.strip("*").strip().rstrip("\\").strip()


def _extraer_pregunta(pre_close: str, body: str) -> str | None:
    """Une fragmentos del encabezado y normaliza la pregunta (añade ¿ si falta)."""
    raw = _limpiar(f"{pre_close or ''} {body or ''}")
    raw = re.sub(r"\*+", "", raw).strip(" .-:")
    raw = _limpiar(raw)
    if not raw or "?" not in raw:
        return None
    pref = ""
    pm = _TIPO_PREF_RE.match(raw)
    if pm:
        pref = f"({pm.group(1).upper()}) "
        raw = _limpiar(raw[pm.end() :])
    # Tomar desde el primer ¿ si existe (permite preámbulo antes)
    if "¿" in raw:
        raw = raw[raw.index("¿") :]
    else:
        raw = "¿" + raw
    # Cerrar en el último ? de la línea (por si hay ? intermedios raros)
    if not raw.endswith("?"):
        raw = raw[: raw.rindex("?") + 1]
    texto = f"{pref}{raw}" if pref else raw
    return texto if len(raw) > 2 else None


def _parse_items(markdown: str) -> list[ItemCuestionario]:
    matches = list(_ITEM_RE.finditer(markdown))
    items: list[ItemCuestionario] = []
    seen: set[str] = set()

    for i, m in enumerate(matches):
        code = _limpiar(m.group("code"))
        num = m.group("num_in") or m.group("num_out")
        body = m.group("body") or ""
        pre_close = m.group("pre_close") or ""

        # **CAAUA**. 1 ¿… → el número quedó en el body
        if not num:
            bm = re.match(
                r"(\d+(?:\.\d+)?)\s*(?:\\?[.\-])*\s*(.*)$",
                body.strip(),
            )
            if bm:
                num = bm.group(1)
                body = bm.group(2)
        if not num:
            continue

        codigo = f"{code}.{num}"
        pregunta = _extraer_pregunta(pre_close, body)
        if not pregunta:
            continue

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        bloque = markdown[start:end]

        fund = None
        # Preferir bloque detallado (texto íntegro de artículos); si no, cita corta.
        fund_m = _FUND_DETALLADO_RE.search(bloque)
        if fund_m:
            fund = _limpiar(fund_m.group("val"))
        else:
            art_m = _ART_RE.search(bloque)
            if art_m:
                fund = _limpiar(art_m.group("val"))

        crit = None
        crit_m = _CRIT_RE.search(bloque)
        if crit_m:
            crit = _limpiar(crit_m.group("crit"))

        accion = None
        acc_m = _ACCION_RE.search(bloque)
        if acc_m:
            accion = _limpiar(acc_m.group("val"))

        adv = None
        adv_m = _ADV_RE.search(bloque)
        if adv_m:
            adv = _limpiar(adv_m.group("val"))

        item = ItemCuestionario(
            codigo=codigo,
            texto=pregunta,
            fundamento_legal=fund or None,
            rango_criticidad=crit or None,
            accion_legal=accion or None,
            advertencia_gerencia=adv or None,
        )
        if codigo in seen:
            # Duplicado: conservar el que traiga más metadatos (p. ej. CDACTO.19).
            prev_i = next(i for i, it in enumerate(items) if it.codigo == codigo)
            prev = items[prev_i]
            prev_score = sum(
                bool(x)
                for x in (
                    prev.fundamento_legal,
                    prev.rango_criticidad,
                    prev.accion_legal,
                )
            )
            new_score = sum(
                bool(x)
                for x in (
                    item.fundamento_legal,
                    item.rango_criticidad,
                    item.accion_legal,
                )
            )
            if new_score > prev_score:
                items[prev_i] = item
            continue
        seen.add(codigo)
        items.append(item)
    return items


@lru_cache(maxsize=128)
def cargar_cuestionario(
    modalidad: Modalidad,
    tipo: TipoDocumento,
) -> CuestionarioDocumento | None:
    path = ruta_cuestionario(modalidad, tipo)
    blob = f"{modalidad.value}/{tipo.value}.md"
    markdown = leer_texto(local_path=path, gcs_blob=blob, kind="cuestionario")
    if markdown is None:
        return None

    items = _parse_items(markdown)
    if not items:
        logger.warning("Cuestionario sin ítems parseados: %s", blob)

    return CuestionarioDocumento(
        modalidad=modalidad,
        tipo_documento=tipo,
        path=path,
        markdown=markdown,
        items=tuple(items),
    )


def textos_preguntas(cuest: CuestionarioDocumento) -> list[str]:
    return [it.texto for it in cuest.items]


def alcance_tipo_contrato(texto: str) -> frozenset[str] | None:
    """Tipos de contrato a los que aplica la pregunta.

    None = aplica a BIENES, OBRAS y SERVICIOS.
    frozenset({'BIENES'}) = solo ese tipo (el resto se excluye del pipeline).
    """
    raw = (texto or "").strip()
    if not raw:
        return None

    pref = _TIPO_PREF_RE.match(raw)
    if pref:
        g = pref.group(1).upper()
        if g in {"BIENES", "OBRAS", "SERVICIOS"}:
            return frozenset({g})

    tl = raw.lower()
    # Preguntas que aplican a obra O servicio (no bienes).
    if re.search(
        r"obra o se prestar[ií]a el servicio|obra o(?:\s+se)?\s+servicio|"
        r"ejecutar[ií]a la obra o|"
        r"\(servicios\s*/\s*obras\)|servicios\s*/\s*obras|"
        r"estructura de costos|an[aá]lisis de precios unitarios|\bapu\b",
        tl,
    ):
        return frozenset({"OBRAS", "SERVICIOS"})

    flags: set[str] = set()
    if re.search(
        r"\(bienes\)|adquisici[oó]n de bienes|bienes a adquirir|"
        r"caracter[ií]sticas de los bienes|entrega de los bienes|"
        r"entrega de bienes|para (la )?adquisici[oó]n de bienes|"
        r"forma de entrega de los bienes|plazos para la entrega de bienes|"
        r"condiciones para la entrega de bienes|"
        r"los bienes una vez recibidos|"
        r"especificaciones t[eé]cnicas de los bienes|"
        r"contrataci[oó]n de bienes|"
        r"cantidades del bien|"
        r"bienes de gran importancia|"
        r"20\.?000\s*ucau|art\.?\s*77\.1\b|"
        r"no podr[aá] ser menor de 7 d[ií]as h[aá]biles",
        tl,
    ):
        flags.add("BIENES")
    if re.search(
        r"\(servicios\)|prestaci[oó]n de servicios|servicios a prestar|"
        r"caracter[ií]sticas de los servicios|para (la )?prestaci[oó]n|"
        r"forma de prestaci[oó]n del servicio|"
        r"plazos para la prestaci[oó]n del servicio|"
        r"condiciones para la prestaci[oó]n del servicio|"
        r"los servicios una vez recibidos|"
        r"especificaciones t[eé]cnicas de los servicios|"
        r"contrataci[oó]n de servicios|"
        r"alcance del servicio|"
        r"30\.?000\s*ucau|art\.?\s*77\.2\b|"
        r"no podr[aá] ser menor de 9 d[ií]as h[aá]biles",
        tl,
    ):
        flags.add("SERVICIOS")
    if re.search(
        r"\(obras\)|ejecuci[oó]n de (la )?obra|obras a ejecutar|"
        r"caracter[ií]sticas de las obras|listas de cantidades|"
        r"forma de ejecuci[oó]n de las obras|"
        r"plazos para la ejecuci[oó]n de la obra|"
        r"condiciones para la ejecuci[oó]n de la obra|"
        r"las obras una vez recibidas|"
        r"obra a ejecutar|especificaciones t[eé]cnicas de la obra|"
        r"ejecuci[oó]n de una obra|"
        r"proyecto como la obra|proyecto y la (ejecuci[oó]n de la )?obra|"
        r"anteproyecto|cantidades de obra|"
        r"50\.?000\s*ucau|art\.?\s*77\.3\b|"
        r"no podr[aá] ser menor de 11 d[ií]as h[aá]biles",
        tl,
    ):
        flags.add("OBRAS")

    if len(flags) == 1:
        return frozenset(flags)
    if flags == {"OBRAS", "SERVICIOS"}:
        return frozenset(flags)
    # Si menciona varios tipos de forma excluyente (bienes vs servicios vs obras
    # en ítems hermanos), no mezclar: None = transversal.
    return None


def item_aplica_a_tipo(texto: str, tipo_contratacion: str) -> bool:
    """True si la pregunta debe evaluarse con el tipo de contrato de la sesión."""
    tipo = (tipo_contratacion or "").strip().upper()
    if not tipo:
        return True
    alcance = alcance_tipo_contrato(texto)
    if alcance is None:
        return True
    return tipo in alcance


def particionar_items_por_tipo(
    items: list[ItemCuestionario] | tuple[ItemCuestionario, ...],
    tipo_contratacion: str,
) -> tuple[list[ItemCuestionario], list[ItemCuestionario]]:
    """Separa ítems a evaluar con LLM vs auto-na por tipo de contrato."""
    aplicables: list[ItemCuestionario] = []
    na_auto: list[ItemCuestionario] = []
    for it in items:
        if item_aplica_a_tipo(it.texto, tipo_contratacion):
            aplicables.append(it)
        else:
            na_auto.append(it)
    return aplicables, na_auto


def formato_cuestionario_compacto(
    cuest: CuestionarioDocumento,
    *,
    items: list[ItemCuestionario] | tuple[ItemCuestionario, ...] | None = None,
) -> str:
    """Versión liviana para el LLM (sin acción/advertencia largas)."""
    usados = tuple(items) if items is not None else cuest.items
    lineas: list[str] = [
        f"Cuestionario oficial — {cuest.tipo_documento.value} "
        f"({len(usados)} ítems a evaluar). Responde TODOS los códigos listados."
    ]
    for it in usados:
        lineas.append(f"- {it.codigo}: {it.texto}")
        if it.rango_criticidad:
            lineas.append(f"  Criticidad: {it.rango_criticidad}")
        if it.fundamento_legal:
            fund = it.fundamento_legal.strip()
            if len(fund) > _FUNDAMENTO_COMPACTO_MAX:
                fund = fund[:_FUNDAMENTO_COMPACTO_MAX].rstrip() + "…"
            lineas.append(f"  Fundamento: {fund}")
    return "\n".join(lineas)
