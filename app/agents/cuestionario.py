"""Helpers de cuestionario de seguimiento e informes."""

from __future__ import annotations

import logging
import re

from app.models.schemas import (
    DocumentoAnalizado,
    HallazgoAnalista,
    PreguntaSeguimiento,
    SesionCompliance,
)

logger = logging.getLogger(__name__)

_QA_SECTION = "## Rúbrica del documento (Analista)"
_QA_SECTION_ALT = "## Respuestas de seguimiento / cuestionario"
_QA_SECTION_ALT2 = "## Respuestas de seguimiento"

# Estados que cuentan como hallazgo en el informe de usuario / JSON Nest.
_ESTADOS_HALLAZGO = frozenset({"no", "parcial", "no_consta"})


def pregunta_pendiente(doc: DocumentoAnalizado) -> PreguntaSeguimiento | None:
    """Ya no hay Q&A de usuario: la rúbrica la responde el Analista al subir."""
    return None


def progreso_cuestionario(doc: DocumentoAnalizado) -> tuple[int, int]:
    total = len(doc.preguntas_seguimiento)
    hechas = sum(1 for p in doc.preguntas_seguimiento if p.respondida)
    return hechas, total


def documento_cuestionario_activo(
    sesion: SesionCompliance,
) -> DocumentoAnalizado | None:
    """Desactivado: el chat no atrapa respuestas de cuestionario de usuario."""
    return None


def _norm_estado(raw: str) -> str:
    t = raw.strip().lower().replace(" ", "_")
    if t in {"n/a", "n.a.", "na", "no_aplica", "noaplica"}:
        return "na"
    if t in {"si", "sí", "yes"}:
        return "si"
    if t in {"no"}:
        return "no"
    if t in {"parcial"}:
        return "parcial"
    if t in {"no_consta", "noconsta"}:
        return "no_consta"
    return "no_consta"


def aplicar_rubrica_agente(
    preguntas: list[PreguntaSeguimiento],
    rubrica_raw: object,
) -> None:
    """Rellena cada ítem con estado/respuesta/ref del Analista.

    Empareja solo por código oficial o id de pregunta (qN). Sin fallback
    posicional. No sobrescribe campos del cuestionario MD.
    """
    by_id: dict[str, dict] = {}
    items: list = []
    if isinstance(rubrica_raw, list):
        items = rubrica_raw
    elif isinstance(rubrica_raw, dict):
        items = rubrica_raw.get("items") or rubrica_raw.get("preguntas") or []
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in ("id", "codigo_pregunta"):
            pid = str(item.get(key) or "").strip().lower()
            if pid:
                by_id[pid] = item

    for p in preguntas:
        item = by_id.get(p.id.lower()) if p.id else None
        if item is None and p.codigo_pregunta:
            item = by_id.get(p.codigo_pregunta.lower())
        if not item:
            p.respondida = True
            p.respondida_por = "agente"
            p.estado = "no_consta"
            p.respuesta = "Sin valoración explícita del modelo sobre este ítem."
            continue
        estado = _norm_estado(str(item.get("estado") or ""))
        resp = str(item.get("respuesta") or item.get("explicacion") or "").strip()
        ref = item.get("ref")
        p.estado = estado  # type: ignore[assignment]
        p.respuesta = resp or f"Estado: {estado}"
        p.ref = str(ref).strip() if ref not in (None, "") else None
        # Conservar codigo/fundamento/criticidad/acción/advertencia del MD.
        p.respondida = True
        p.respondida_por = "agente"


def verificar_integridad_rubrica(
    preguntas: list[PreguntaSeguimiento],
    *,
    codigos_oficiales: set[str] | None = None,
) -> list[str]:
    """Valida fidelidad texto/código post-apply. Devuelve warnings (no lanza)."""
    avisos: list[str] = []
    oficiales = {c.lower() for c in (codigos_oficiales or set()) if c}
    for p in preguntas:
        cod = (p.codigo_pregunta or "").strip()
        if oficiales and cod and cod.lower() not in oficiales:
            msg = f"código ajeno al cuestionario: {cod} ({p.id})"
            avisos.append(msg)
            logger.warning("Integridad rúbrica: %s", msg)
        if not cod and oficiales:
            msg = f"ítem sin codigo_pregunta: {p.id}"
            avisos.append(msg)
            logger.warning("Integridad rúbrica: %s", msg)
        # Heurística: respuesta que cita otro código del cuestionario
        if cod and p.respuesta and oficiales:
            otros = [
                c for c in oficiales
                if c != cod.lower() and re.search(
                    rf"\b{re.escape(c)}\b", p.respuesta or "", re.IGNORECASE
                )
            ]
            if otros:
                msg = (
                    f"{cod}: respuesta menciona otro(s) código(s) "
                    f"{', '.join(otros[:3])}"
                )
                avisos.append(msg)
                logger.warning("Integridad rúbrica: %s", msg)
    return avisos


def _es_critico(rango: str | None) -> bool:
    u = (rango or "").upper()
    return "5" in u or "CRÍTIC" in u or "CRITIC" in u


def _es_relevante(rango: str | None) -> bool:
    u = (rango or "").upper()
    return "3" in u or "RELEVANT" in u


def calcular_estatus_desde_preguntas(
    preguntas: list[PreguntaSeguimiento],
    *,
    tipo_coincide: bool = True,
) -> str:
    """Verde / Amarillo / Rojo según criticidad de hallazgos (sin N/A)."""
    if not tipo_coincide:
        return "Rojo"
    worst = "Verde"
    for p in preguntas:
        est = (p.estado or "").lower()
        if est in {"si", "na"}:
            continue
        if est in _ESTADOS_HALLAZGO:
            if _es_critico(p.rango_criticidad):
                return "Rojo"
            if worst != "Rojo":
                worst = "Amarillo"
    return worst


def construir_hallazgos_analista(
    preguntas: list[PreguntaSeguimiento],
) -> list[HallazgoAnalista]:
    """Solo ítems no conformes (excluye si/na). Sin N/A en el JSON de Nest."""
    out: list[HallazgoAnalista] = []
    for p in preguntas:
        est = (p.estado or "").lower()
        if est not in _ESTADOS_HALLAZGO:
            continue
        if est == "na":
            continue
        out.append(
            HallazgoAnalista(
                codigo_pregunta=(p.codigo_pregunta or p.id or "").strip(),
                pregunta_evaluada=p.texto,
                fundamento_legal=p.fundamento_legal,
                rango_criticidad=p.rango_criticidad,
                accion_legal=p.accion_legal,
                advertencia_gerencia=p.advertencia_gerencia,
                estado=est if est in {"no", "parcial", "no_consta"} else "no_consta",  # type: ignore[arg-type]
                respuesta=p.respuesta,
                ref=p.ref,
            )
        )
    return out


def _fundamento_para_json(
    fundamento: str | None,
    *,
    accion: str | None = None,
    max_chars: int = 520,
) -> str | None:
    """Cita compacta priorizando artículos mencionados en la acción legal.

    En varios MD la «Normativa principal» repite Art. 78 LCP genérico; la norma
    útil (p. ej. Art. 13 SUNAI) está en complementaria. Preferimos las citas
    de la acción/advertencia cuando aparecen en el texto completo.
    """
    if not fundamento:
        return None
    t = re.sub(r"\*+", "", str(fundamento)).strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"^\s*Normativa\s+principal\s*", "", t, flags=re.IGNORECASE).strip()

    nums: list[str] = []
    for src in (accion or "",):
        nums.extend(re.findall(r"(?:Art\.?|Artículo)\s*(\d+(?:\.\d+)?)", src, flags=re.I))
    # También números sueltos tipo «91.1 LOCGR» / «13 SUNAI»
    nums.extend(re.findall(r"\b(\d+(?:\.\d+)?)\s*(?:SUNAI|LOCGR|LOPA|LCP|RLCP|LCC)\b", accion or "", flags=re.I))
    # únicos preservando orden
    seen: set[str] = set()
    nums_u: list[str] = []
    for n in nums:
        root = n.split(".")[0]
        if root not in seen:
            seen.add(root)
            nums_u.append(root)

    bloques = re.split(r"(?=Artículo\s+\d)", t, flags=re.IGNORECASE)
    bloques = [b.strip() for b in bloques if b.strip()]
    elegidos: list[str] = []
    if nums_u and bloques:
        for b in bloques:
            mnum = re.match(r"Artículo\s+(\d+)", b, flags=re.IGNORECASE)
            if mnum and mnum.group(1) in nums_u:
                # Primera frase / ~280 chars del artículo
                breve = b[:280].rstrip()
                if not breve.endswith("."):
                    punto = breve.find(". ")
                    if punto > 40:
                        breve = breve[: punto + 1]
                elegidos.append(breve)
            if len(elegidos) >= 2:
                break
    if not elegidos and bloques:
        # Primer artículo (principal) acotado
        elegidos = [bloques[0][:320].rstrip()]

    out = " ".join(elegidos).strip()
    if len(out) > max_chars:
        out = out[:max_chars].rstrip() + "…"
    return out or None


def construir_json_analista(
    *,
    nombre_documento: str,
    tipo_contrato: str,
    estatus_global: str,
    hallazgos: list[HallazgoAnalista],
) -> dict:
    """Parte 2 del formato Mariana (objeto estructurado)."""
    return {
        "documento_evaluado": nombre_documento,
        "tipo_contrato": tipo_contrato,
        "estatus_global": estatus_global,
        "hallazgos": [
            {
                "codigo_pregunta": h.codigo_pregunta,
                "pregunta_evaluada": h.pregunta_evaluada,
                "fundamento_legal": _fundamento_para_json(
                    h.fundamento_legal, accion=h.accion_legal
                ),
                "rango_criticidad": h.rango_criticidad,
                "accion_legal": h.accion_legal,
                "advertencia_gerencia": h.advertencia_gerencia,
            }
            for h in hallazgos
        ],
    }


def _fundamento_corto(fundamento: str | None, *, max_chars: int = 280) -> str:
    """Cita breve para el informe de usuario (el JSON completo va al Jurídico)."""
    t = re.sub(r"\*+", "", str(fundamento or "")).strip()
    t = re.sub(r"\s+", " ", t)
    if not t:
        return "fundamento pendiente de ampliación del corpus"
    # Preferir el bloque de normativa principal si existe
    m = re.search(
        r"(?:Normativa principal\s*)?((?:Art[ií]culo|Art\.?)[^.]{0,200}\.)",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        t = m.group(1).strip()
    if len(t) > max_chars:
        t = t[:max_chars].rstrip() + "…"
    return t


def construir_informe_usuario_markdown(
    *,
    nombre_documento: str,
    tipo_contrato: str,
    estatus_global: str,
    preguntas: list[PreguntaSeguimiento],
) -> str:
    """Informe legal en prosa continua para el usuario.

    Sin N/A, sin emojis, sin listar el texto completo de cada pregunta.
    El fundamento se muestra abreviado; el JSON estructurado conserva el detalle.
    """
    aplicables = [p for p in preguntas if (p.estado or "").lower() != "na"]
    aciertos = [p for p in aplicables if (p.estado or "").lower() == "si"]
    hallazgos = [
        p for p in aplicables if (p.estado or "").lower() in _ESTADOS_HALLAZGO
    ]

    explicacion = {
        "Verde": (
            "los puntos evaluados para este tipo de contrato resultan conformes"
        ),
        "Amarillo": (
            "existen observaciones formales o hallazgos relevantes/ordinarios "
            "que admiten subsanación"
        ),
        "Rojo": (
            "se identificó al menos un hallazgo crítico o una inconsistencia "
            "grave que requiere atención inmediata"
        ),
    }.get(estatus_global, "corresponde revisar el detalle de hallazgos")

    partes: list[str] = [
        (
            f"He finalizado la revisión técnica y legal del documento "
            f"«{nombre_documento}», correspondiente a un contrato de "
            f"{tipo_contrato or 'N/D'}."
        ),
        (
            f"El estatus global del documento es {estatus_global}: "
            f"{explicacion}. "
            f"Se valoraron {len(aplicables)} ítems aplicables a este tipo de "
            f"contrato ({len(aciertos)} conformes y {len(hallazgos)} con "
            f"observación o incumplimiento)."
        ),
    ]

    if aciertos:
        ejemplos = []
        for p in aciertos[:5]:
            cod = p.codigo_pregunta or p.id
            breve = (p.respuesta or "conforme").strip().rstrip(".")
            if len(breve) > 120:
                breve = breve[:120].rstrip() + "…"
            ejemplos.append(f"{cod} ({breve})")
        extra = ""
        if len(aciertos) > 5:
            extra = f" y {len(aciertos) - 5} punto(s) adicional(es) conforme(s)"
        partes.append(
            "Entre los aspectos que cumplen se destacan: "
            + "; ".join(ejemplos)
            + extra
            + "."
        )

    if hallazgos:
        partes.append("A continuación se exponen los hallazgos relevantes.")
        for i, p in enumerate(hallazgos, start=1):
            cod = p.codigo_pregunta or p.id
            est = (p.estado or "no").lower()
            grado = {
                "no": "incumplimiento",
                "parcial": "cumplimiento parcial",
                "no_consta": "ausencia de evidencia suficiente en el documento",
            }.get(est, "observación")
            cuerpo = (p.respuesta or "").strip().rstrip(".")
            if not cuerpo or cuerpo.lower().startswith("estado:"):
                cuerpo = "No se constató el cumplimiento del control requerido"
            if len(cuerpo) > 220:
                cuerpo = cuerpo[:220].rstrip() + "…"
            fund = _fundamento_corto(p.fundamento_legal)
            crit = (p.rango_criticidad or "N/D").strip()
            accion = (p.accion_legal or "").strip()
            adv = (p.advertencia_gerencia or "").strip()
            ref = (p.ref or "").strip()

            cuerpo = cuerpo.rstrip(".")
            fund = fund.rstrip(".")
            bloque = (
                f"Hallazgo {i} ({cod}, criticidad {crit}): se observa "
                f"{grado}. {cuerpo}. Fundamento: {fund}."
            )
            if accion:
                if len(accion) > 220:
                    accion = accion[:220].rstrip() + "…"
                bloque += f" Acción recomendada: {accion.rstrip('.')}."
            if adv:
                if len(adv) > 180:
                    adv = adv[:180].rstrip() + "…"
                bloque += f" Advertencia a la gerencia: {adv.rstrip('.')}."
            if ref and ref.lower() != "null":
                bloque += f" Evidencia: {ref}."
            partes.append(bloque)
    else:
        partes.append(
            "No se registraron hallazgos para los ítems aplicables a este "
            "tipo de contrato."
        )

    partes.append(
        "Este informe se limita a la revisión del documento cargado; "
        "el dictamen jurídico del expediente podrá integrar estos hallazgos "
        "con el resto de las piezas del procedimiento."
    )
    return "\n\n".join(partes).rstrip() + "\n"


def resumen_rubrica_chat(doc: DocumentoAnalizado) -> str:
    """Resumen corto post-upload: estatus + extracto del informe (sin rúbrica)."""
    if doc.informe_markdown.strip():
        # Primeras líneas del informe de usuario
        lines = [ln for ln in doc.informe_markdown.splitlines() if ln.strip()]
        return "\n".join(lines[:12])
    hechas, total = progreso_cuestionario(doc)
    estatus = doc.estatus_global or ("Verde" if doc.cumple else "Amarillo")
    return (
        f"Estatus global: {estatus}. Ítems evaluados: {hechas}/{total} "
        f"(solo tipo de contrato de la sesión)."
    )


def sincronizar_informe_documento(doc: DocumentoAnalizado) -> None:
    """Asegura informe de usuario sin sección de rúbrica tabular."""
    base = doc.informe_markdown or ""
    for marker in (_QA_SECTION, _QA_SECTION_ALT, _QA_SECTION_ALT2):
        if marker in base:
            base = base.split(marker)[0].rstrip()
            break
    # Quitar bloque de cobertura técnica si quedó en el MD del LLM
    if "### Cobertura de rúbrica" in base:
        base = base.split("### Cobertura de rúbrica")[0].rstrip()
    doc.informe_markdown = re.sub(r"\n{3,}", "\n\n", base).rstrip() + ("\n" if base else "")


def registrar_respuesta(
    doc: DocumentoAnalizado,
    texto_respuesta: str,
    *,
    pregunta_id: str | None = None,
) -> PreguntaSeguimiento | None:
    """Compat: permite sobreescribir un ítem (p. ej. corrección manual)."""
    target: PreguntaSeguimiento | None = None
    if pregunta_id:
        target = next((p for p in doc.preguntas_seguimiento if p.id == pregunta_id), None)
    if target is None:
        for p in doc.preguntas_seguimiento:
            if not p.respondida:
                target = p
                break
    if target is None:
        return None
    target.respuesta = texto_respuesta.strip()
    target.respondida = True
    target.respondida_por = "usuario"
    sincronizar_informe_documento(doc)
    return target


def mensaje_pregunta_actual(doc: DocumentoAnalizado) -> str:
    return resumen_rubrica_chat(doc) or (
        f"Rúbrica del documento «{doc.tipo.value}» sin ítems precargados."
    )


def construir_informe_global_estructurado(sesion: SesionCompliance) -> str:
    """Informe global determinístico (sin LLM) con hallazgos y recomendaciones."""
    from app.agents.knowledge import etiqueta_modalidad

    mod = (
        etiqueta_modalidad(sesion.modalidad)
        if sesion.modalidad
        else "N/D"
    )
    tipo = sesion.tipo_contratacion.value if sesion.tipo_contratacion else "N/D"
    lineas: list[str] = [
        f"# Informe global de auditoría — {sesion.nomenclatura or sesion.id}",
        "",
        "## 1. Identificación del expediente",
        "",
        f"- **Nomenclatura:** {sesion.nomenclatura or 'N/D'}",
        f"- **Modalidad:** {mod}",
        f"- **Tipo de contratación:** {tipo}",
        f"- **Documentos revisados:** {len(sesion.documentos_analizados)}",
        "",
        "## 2. Documentos auditados",
        "",
        "| Tipo | Archivo | Identidad | Cumple | Observaciones |",
        "| --- | --- | --- | --- | --- |",
    ]
    criticas: list[str] = []
    advertencias: list[str] = []
    recomendaciones: list[str] = []

    for d in sesion.documentos_analizados:
        identidad = "OK" if getattr(d, "tipo_coincide", True) else (
            f"INCORRECTO→{d.tipo_detectado or '?'}"
        )
        lineas.append(
            f"| {d.tipo.value} | {d.nombre_archivo} | {identidad} | "
            f"{'sí' if d.cumple else 'no'} | {len(d.observaciones)} |"
        )
        for o in d.observaciones:
            item = f"**{d.tipo.value}** ({d.nombre_archivo}): {o.descripcion}"
            if o.severidad == "critica":
                criticas.append(item)
            elif o.severidad == "advertencia":
                advertencias.append(item)
            if o.subsanacion:
                recomendaciones.append(
                    f"[{d.tipo.value}] {o.subsanacion}"
                )
        if not getattr(d, "tipo_coincide", True):
            recomendaciones.append(
                f"[{d.tipo.value}] Reponer el documento correcto o reclasificar "
                f"el archivo (detectado: {d.tipo_detectado or 'otro'})."
            )

    lineas.extend(["", "## 3. Hallazgos críticos", ""])
    if criticas:
        lineas.extend(f"- {c}" for c in criticas)
    else:
        lineas.append("- Sin hallazgos críticos registrados.")

    lineas.extend(["", "## 4. Advertencias", ""])
    if advertencias:
        lineas.extend(f"- {a}" for a in advertencias)
    else:
        lineas.append("- Sin advertencias registradas.")

    lineas.extend(["", "## 5. Rúbrica por documento (Analista)", ""])
    hubo_qa = False
    for d in sesion.documentos_analizados:
        if not d.preguntas_seguimiento:
            continue
        hubo_qa = True
        hechas, total = progreso_cuestionario(d)
        lineas.append(f"### {d.tipo.value} — {d.nombre_archivo} ({hechas}/{total})")
        lineas.append("")
        for p in d.preguntas_seguimiento:
            est = f" [{p.estado}]" if p.estado else ""
            lineas.append(f"- **{p.texto}**{est}")
            if p.respondida and p.respuesta:
                lineas.append(f"  - Analista: {p.respuesta}")
                if p.ref:
                    lineas.append(f"  - Ref: {p.ref}")
            else:
                lineas.append("  - Analista: _(pendiente)_")
        lineas.append("")
    if not hubo_qa:
        lineas.append("- Aún no hay rúbricas asociadas.")

    lineas.extend(["", "## 5b. Dictámenes jurídicos", ""])
    if sesion.dictamenes_juridicos:
        for dj in sesion.dictamenes_juridicos:
            lineas.append(
                f"### Dictamen {dj.alcance.value} — {dj.fecha.isoformat()} "
                f"({len(dj.documento_ids)} doc(s))"
            )
            lineas.append("")
            lineas.append(dj.markdown or "_(vacío)_")
            lineas.append("")
            if dj.slots_pendientes:
                lineas.append(
                    "Limitación — slots pendientes: "
                    + ", ".join(s.value for s in dj.slots_pendientes)
                )
                lineas.append("")
    else:
        lineas.append("- Sin dictámenes jurídicos registrados.")

    pendientes_slots = [
        s.tipo_documento.value
        for s in sesion.checklist_slots
        if not s.auditado
    ]
    lineas.extend(["", "## 6. Cobertura del checklist sugerido", ""])
    if pendientes_slots:
        lineas.append(
            "Slots aún sin documento de identidad correcta: "
            + ", ".join(pendientes_slots)
        )
        recomendaciones.append(
            "Completar la carga de los documentos pendientes del checklist sugerido."
        )
    else:
        lineas.append("- Todos los slots sugeridos tienen al menos un documento con tipo coincidente.")

    # Dedup recomendaciones
    seen: set[str] = set()
    recs_uniq: list[str] = []
    for r in recomendaciones:
        if r not in seen:
            seen.add(r)
            recs_uniq.append(r)

    lineas.extend(["", "## 7. Conclusión", ""])
    n_ok = sum(1 for d in sesion.documentos_analizados if d.cumple and getattr(d, "tipo_coincide", True))
    n_bad = len(sesion.documentos_analizados) - n_ok
    lineas.append(
        f"De {len(sesion.documentos_analizados)} documento(s) revisado(s), "
        f"{n_ok} cumplen con identidad correcta y evaluación favorable; "
        f"{n_bad} presentan incumplimientos, tipo incorrecto u observaciones abiertas."
    )

    lineas.extend(["", "## 8. Recomendaciones", ""])
    if recs_uniq:
        for i, r in enumerate(recs_uniq, start=1):
            lineas.append(f"{i}. {r}")
    else:
        lineas.append(
            "1. Mantener trazabilidad del expediente y archivar los informes por documento."
        )
        lineas.append(
            "2. Revisar periódicamente la coherencia de nomenclatura entre piezas del expediente."
        )

    lineas.append("")
    return "\n".join(lineas)
