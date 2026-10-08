"""API de sesiones conversacionales de compliance."""

from __future__ import annotations

import json
import math
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

from app.agents.cuestionario import (
    aplicar_rubrica_agente,
    calcular_estatus_desde_preguntas,
    construir_hallazgos_analista,
    construir_informe_usuario_markdown,
    construir_json_analista,
    verificar_integridad_rubrica,
    resumen_rubrica_chat,
    sincronizar_informe_documento,
)
from app.agents.extractor import procesar_archivo
from app.agents.knowledge import etiqueta_modalidad
from app.agents.knowledge.cuestionarios_loader import (
    cargar_cuestionario,
    formato_cuestionario_compacto,
    particionar_items_por_tipo,
)
from app.agents.modalities import get_modalidad_agent
from app.agents.orchestrator import crear_sesion, ejecutar_dictamen_juridico, procesar_mensaje
from app.agents.report_agent import (
    DocumentoNoEncontradoError,
    InformeNoDisponibleError,
    SesionNoEncontradaError,
    export_dictamen_juridico,
    export_documento,
    export_global,
    informe_documento_markdown,
    informe_global_markdown,
)
from app.core import session_store
from app.core.llm_client import analizar_documento
from app.core.security import (
    max_upload_bytes,
    sanitizar_nombre_archivo,
    validar_extension_upload,
)
from app.models.schemas import (
    AnalisisDocumentoResponse,
    CrearSesionRequest,
    CrearSesionResponse,
    DocumentoAnalizado,
    EstadoSesion,
    ExtraccionMeta,
    HechosClave,
    InformeMarkdownResponse,
    JuridicoRequest,
    JuridicoResponse,
    MensajeRequest,
    MensajeResponse,
    MetricasAnalisis,
    Modalidad,
    MontoClave,
    NivelRiesgo,
    Observacion,
    PlazoClave,
    PreguntaSeguimiento,
    RespuestasDocumentoRequest,
    SandboxAnalizarResponse,
    SesionCompliance,
    SesionesListaResponse,
    SesionResumen,
    TipoContratacion,
    TipoDocumento,
)

router = APIRouter(prefix="/sesiones", tags=["sesiones"])

_SEVERIDADES = {"info", "advertencia", "critica"}

# Errores de proveedor LLM → HTTP 502/504 (no 500 opaco / Failed to fetch)
_LLM_ERRORS = (
    APITimeoutError,
    RateLimitError,
    APIError,
    APIConnectionError,
    RuntimeError,
)


def _http_from_llm(exc: BaseException) -> HTTPException:
    """Mapea fallos del LLM a respuestas HTTP con detail usable en el front."""
    status = getattr(exc, "status_code", None)
    raw = str(exc).lower()
    if (
        isinstance(exc, APITimeoutError)
        or status in {504, 524, 408}
        or "origin_response_timeout" in raw
        or "error 524" in raw
        or "timed out" in raw
    ):
        return HTTPException(
            status_code=504,
            detail=(
                "El modelo tardó demasiado en responder. "
                "Reintenta en unos segundos; si el documento es muy largo, "
                "vuelve a subirlo."
            ),
        )
    if isinstance(exc, RateLimitError):
        return HTTPException(
            status_code=429,
            detail="Cuota o rate limit del proveedor LLM agotada. Reintenta más tarde o usa el fallback.",
        )
    msg = str(exc).strip() or type(exc).__name__
    # RuntimeError dual-fail ya viene acortado desde llm_client
    if len(msg) > 500:
        msg = msg[:500] + "…"
    return HTTPException(status_code=502, detail=f"Error LLM: {msg}")


def _as_bool(val: object, *, default: bool = True) -> bool:
    """Parsea bool tolerante (evita que bool('false') == True)."""
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in {"true", "1", "yes", "si", "sí", "verdadero"}:
            return True
        if s in {"false", "0", "no", "falso", ""}:
            return False
    return default


def _require_sesion(sesion_id: str) -> SesionCompliance:
    sesion = session_store.get_sesion(sesion_id)
    if sesion is None:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    return sesion


def _parse_observaciones(raw: object) -> list[Observacion]:
    if not isinstance(raw, list):
        return []
    out: list[Observacion] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        severidad = str(item.get("severidad", "advertencia")).lower()
        if severidad not in _SEVERIDADES:
            # Mapear criticidad del cuestionario oficial si viene así
            crit = str(item.get("rango_criticidad") or "").upper()
            if "5" in crit or "CRÍTICO" in crit or "CRITICO" in crit:
                severidad = "critica"
            elif "3" in crit or "RELEVANTE" in crit:
                severidad = "advertencia"
            elif "1" in crit or "ORDINARIA" in crit:
                severidad = "info"
            else:
                severidad = "advertencia"
        descripcion = str(
            item.get("descripcion") or item.get("pregunta_evaluada") or ""
        ).strip()
        if not descripcion:
            continue
        subs = item.get("subsanacion") or item.get("accion_legal")
        ref = item.get("ref")

        def _opt(key: str) -> str | None:
            v = item.get(key)
            return str(v).strip() if v not in (None, "") else None

        out.append(
            Observacion(
                severidad=severidad,  # type: ignore[arg-type]
                descripcion=descripcion,
                subsanacion=str(subs).strip() if subs else None,
                ref=str(ref).strip() if ref else None,
                codigo_pregunta=_opt("codigo_pregunta"),
                fundamento_legal=_opt("fundamento_legal"),
                rango_criticidad=_opt("rango_criticidad"),
                accion_legal=_opt("accion_legal"),
                advertencia_gerencia=_opt("advertencia_gerencia"),
            )
        )
    return out


def _slots_pendientes(sesion: SesionCompliance) -> list[TipoDocumento]:
    """Slots sin un documento de identidad correcta (tipo_coincide)."""
    auditados = {
        d.tipo for d in sesion.documentos_analizados if getattr(d, "tipo_coincide", True)
    }
    return [s.tipo_documento for s in sesion.checklist_slots if s.tipo_documento not in auditados]


def _asegurar_obs_tipo_incorrecto(
    observaciones: list[Observacion],
    *,
    tipo_declarado: TipoDocumento,
    tipo_detectado: str | None,
) -> list[Observacion]:
    """Garantiza observación crítica si el archivo no es el tipo declarado."""
    texto_clave = "tipo declarado"
    if any(texto_clave in o.descripcion.lower() or "no corresponde" in o.descripcion.lower() for o in observaciones):
        return observaciones
    detectado = tipo_detectado or "otro tipo de documento"
    observaciones = list(observaciones)
    observaciones.insert(
        0,
        Observacion(
            severidad="critica",
            descripcion=(
                f"El archivo no corresponde al tipo declarado ({tipo_declarado.value}). "
                f"Se detectó: {detectado}."
            ),
            subsanacion=(
                f"Sube el documento correcto para «{tipo_declarado.value}» "
                "o vuelve a cargar este archivo eligiendo el tipo que realmente es."
            ),
        ),
    )
    return observaciones


def _norm_nomen(s: str) -> str:
    return "".join(ch for ch in (s or "").upper() if ch.isalnum() or ch in "-/")


def _asegurar_obs_nomenclatura(
    observaciones: list[Observacion],
    *,
    nomenclatura_sesion: str | None,
    nomenclatura_encontrada: str | None,
) -> list[Observacion]:
    """Si el documento trae otra nomenclatura clara, fuerza observación."""
    ses = (nomenclatura_sesion or "").strip()
    hall = (nomenclatura_encontrada or "").strip()
    if not ses or not hall:
        return observaciones
    if _norm_nomen(ses) == _norm_nomen(hall):
        return observaciones
    # Si una contiene a la otra (variantes cortas), no forzar
    ns, nh = _norm_nomen(ses), _norm_nomen(hall)
    if ns and nh and (ns in nh or nh in ns):
        return observaciones
    clave = "nomenclatura"
    if any(clave in o.descripcion.lower() for o in observaciones):
        return observaciones
    observaciones = list(observaciones)
    observaciones.insert(
        0,
        Observacion(
            severidad="critica",
            descripcion=(
                f"La nomenclatura del documento («{hall}») no coincide con la de "
                f"la sesión («{ses}»)."
            ),
            subsanacion=(
                "Verifica que el archivo pertenezca a este expediente o corrige "
                "la nomenclatura de la sesión si corresponde."
            ),
        ),
    )
    return observaciones


def _fecha_inicio_sesion(sesion: SesionCompliance) -> datetime | None:
    fi = getattr(sesion, "fecha_inicio", None)
    if fi is not None:
        return fi
    if sesion.historial:
        return sesion.historial[0].timestamp
    if sesion.documentos_analizados:
        return sesion.documentos_analizados[0].fecha_analisis
    return None


def _calcular_riesgo(sesion: SesionCompliance) -> NivelRiesgo:
    tiene_critica = False
    tiene_advertencia = False
    for d in sesion.documentos_analizados:
        if getattr(d, "tipo_coincide", True) is False:
            tiene_critica = True
        for o in d.observaciones:
            if o.severidad == "critica":
                tiene_critica = True
            elif o.severidad == "advertencia":
                tiene_advertencia = True
    if tiene_critica:
        return NivelRiesgo.RIESGO_ALTO
    if tiene_advertencia:
        return NivelRiesgo.OBSERVACIONES
    return NivelRiesgo.SIN_HALLAZGOS


def _sesion_resumen(sesion: SesionCompliance) -> SesionResumen:
    hint = None
    if sesion.historial:
        hint = sesion.historial[-1].timestamp.isoformat()
    elif sesion.documentos_analizados:
        hint = sesion.documentos_analizados[-1].fecha_analisis.isoformat()

    docs_ok = [
        d
        for d in sesion.documentos_analizados
        if getattr(d, "tipo_coincide", True) is not False
    ]
    # Un tipo correcto por slot (si hay varios intentos del mismo tipo, cuenta 1)
    tipos_ok = {d.tipo for d in docs_ok}
    docs_revisados = len(tipos_ok)
    docs_totales = len(sesion.checklist_slots)
    if docs_totales > 0:
        progreso_pct = min(100, int(round(100 * docs_revisados / docs_totales)))
    else:
        progreso_pct = 0

    fi = _fecha_inicio_sesion(sesion)
    return SesionResumen(
        id=sesion.id,
        nomenclatura=sesion.nomenclatura,
        modalidad=sesion.modalidad,
        tipo_contratacion=sesion.tipo_contratacion,
        estado=sesion.estado,
        docs_count=len(sesion.documentos_analizados),
        updated_hint=hint,
        fecha_inicio=fi.isoformat() if fi else None,
        docs_revisados=docs_revisados,
        docs_totales=docs_totales,
        progreso_pct=progreso_pct,
        riesgo=_calcular_riesgo(sesion),
    )


def _parse_hechos_clave(raw: object) -> HechosClave:
    if not isinstance(raw, dict):
        return HechosClave()
    montos: list[MontoClave] = []
    for m in raw.get("montos") or []:
        if not isinstance(m, dict):
            continue
        etiqueta = str(m.get("etiqueta") or "").strip()
        if not etiqueta:
            continue
        valor_num = m.get("valor_num")
        try:
            vn = float(valor_num) if valor_num is not None and valor_num != "" else None
        except (TypeError, ValueError):
            vn = None
        montos.append(
            MontoClave(
                etiqueta=etiqueta,
                texto=str(m.get("texto") or "").strip(),
                valor_num=vn,
                moneda=str(m["moneda"]).strip() if m.get("moneda") else None,
            )
        )
    plazos: list[PlazoClave] = []
    for p in raw.get("plazos") or []:
        if not isinstance(p, dict):
            continue
        etiqueta = str(p.get("etiqueta") or "").strip()
        if not etiqueta:
            continue
        plazos.append(
            PlazoClave(
                etiqueta=etiqueta,
                texto=str(p.get("texto") or "").strip(),
            )
        )
    partes = [
        str(x).strip()
        for x in (raw.get("partes") or [])
        if str(x).strip()
    ]
    nom = raw.get("nomenclatura_encontrada")
    otros = [
        str(x).strip()
        for x in (raw.get("otros") or [])
        if str(x).strip()
    ]
    return HechosClave(
        montos=montos,
        plazos=plazos,
        partes=partes,
        nomenclatura_encontrada=str(nom).strip() if nom not in (None, "") else None,
        otros=otros,
    )


def _formatear_hechos_clave(hc: HechosClave) -> str:
    partes: list[str] = []
    if hc.montos:
        montos_txt = "; ".join(
            (
                f"{m.etiqueta}={m.valor_num}"
                + (f" {m.moneda}" if m.moneda else "")
                + (f" ({m.texto})" if m.texto and m.valor_num is None else "")
            ).strip()
            for m in hc.montos
        )
        partes.append(f"montos=[{montos_txt}]")
    if hc.plazos:
        plazos_txt = "; ".join(f"{p.etiqueta}={p.texto}" for p in hc.plazos)
        partes.append(f"plazos=[{plazos_txt}]")
    if hc.partes:
        partes.append(f"partes={hc.partes}")
    if hc.nomenclatura_encontrada:
        partes.append(f"nomenclatura={hc.nomenclatura_encontrada}")
    if hc.otros:
        partes.append(f"otros={hc.otros}")
    return "; ".join(partes) if partes else "(sin hechos clave)"


def _memoria_expediente(sesion: SesionCompliance) -> str:
    """Bloque de memoria estructurada para cross-validation en el Analista."""
    lineas: list[str] = ["MEMORIA DEL EXPEDIENTE (documentos con identidad correcta):"]
    hubo = False
    for d in sesion.documentos_analizados:
        if getattr(d, "tipo_coincide", True) is False:
            continue
        hubo = True
        hc = d.hechos_clave if isinstance(d.hechos_clave, HechosClave) else HechosClave()
        lineas.append(
            f"- {d.tipo.value} ({d.nombre_archivo}): {_formatear_hechos_clave(hc)}; "
            f"cumple={d.cumple}; resumen={d.resumen[:200]}"
        )
    if not hubo:
        lineas.append("- (aún no hay documentos válidos previos)")
    lineas.append(
        "Instrucción: si el documento ACTUAL contradice montos, plazos, "
        "nomenclatura o partes de la memoria, emite observación critica o "
        "advertencia explícita citando el documento previo en 'ref'."
    )
    return "\n".join(lineas)


def _coincide_busqueda(sesion: SesionCompliance, q: str) -> bool:
    """Match case-insensitive en nomenclatura, modalidad (código/etiqueta) y tipo."""
    needle = q.strip().lower()
    if not needle:
        return True
    haystack: list[str] = []
    if sesion.nomenclatura:
        haystack.append(sesion.nomenclatura)
    if sesion.modalidad:
        haystack.append(sesion.modalidad.value)
        haystack.append(etiqueta_modalidad(sesion.modalidad))
    if sesion.tipo_contratacion:
        haystack.append(sesion.tipo_contratacion.value)
    return any(needle in part.lower() for part in haystack)


@router.get("/", response_model=SesionesListaResponse)
def listar_sesiones(
    page: int = Query(default=1, ge=1, description="Número de página"),
    page_size: int = Query(default=20, ge=1, le=100, description="Tamaño de página"),
    q: str | None = Query(
        default=None,
        description="Búsqueda libre en nomenclatura, modalidad o tipo",
    ),
    modalidad: Modalidad | None = Query(default=None),
    tipo_contratacion: TipoContratacion | None = Query(default=None),
    estado: EstadoSesion | None = Query(default=None),
) -> SesionesListaResponse:
    """Lista sesiones con paginación, búsqueda (q) y filtros exactos."""
    sesiones = session_store.listar_sesiones()
    filtradas: list[SesionCompliance] = []
    for s in sesiones:
        if modalidad is not None and s.modalidad != modalidad:
            continue
        if tipo_contratacion is not None and s.tipo_contratacion != tipo_contratacion:
            continue
        if estado is not None and s.estado != estado:
            continue
        if q and not _coincide_busqueda(s, q):
            continue
        filtradas.append(s)

    resumenes = [_sesion_resumen(s) for s in filtradas]
    resumenes.sort(key=lambda s: s.updated_hint or "", reverse=True)

    total = len(resumenes)
    pages = max(1, math.ceil(total / page_size)) if total else 0
    # Si piden una página fuera de rango, devolver vacía pero con meta correcta
    start = (page - 1) * page_size
    items = resumenes[start : start + page_size]
    return SesionesListaResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
    )


@router.post("/", response_model=CrearSesionResponse)
def abrir_sesion(body: CrearSesionRequest | None = None) -> CrearSesionResponse:
    body = body or CrearSesionRequest()
    sesion, mensaje = crear_sesion(
        nomenclatura=body.nomenclatura,
        modalidad=body.modalidad,
        tipo_contratacion=body.tipo_contratacion,
    )
    return CrearSesionResponse(sesion=sesion, mensaje=mensaje)


@router.post("/sandbox/analizar", response_model=SandboxAnalizarResponse)
async def sandbox_analizar(
    archivo: UploadFile = File(...),
    nomenclatura: str = Form(...),
    modalidad: Modalidad = Form(...),
    tipo_contratacion: TipoContratacion = Form(...),
    tipo_documento: TipoDocumento = Form(...),
) -> SandboxAnalizarResponse:
    """Flujo consolidado para el sandbox HTML: crea sesión + analiza documento."""
    cuest_prev = cargar_cuestionario(modalidad, tipo_documento)
    n_excluidos = 0
    n_aplicables = 0
    if cuest_prev and cuest_prev.items:
        aplic, excl = particionar_items_por_tipo(
            cuest_prev.items, tipo_contratacion.value
        )
        n_aplicables = len(aplic)
        n_excluidos = len(excl)

    sesion, _msg = crear_sesion(
        nomenclatura=nomenclatura.strip(),
        modalidad=modalidad,
        tipo_contratacion=tipo_contratacion,
    )
    analisis_resp = await cargar_documento(
        sesion.id,
        archivo=archivo,
        tipo_documento=tipo_documento,
        tipo_documento_form=None,
    )
    sesion_fresh = _require_sesion(sesion.id)
    doc = analisis_resp.documento
    return SandboxAnalizarResponse(
        sesion=sesion_fresh,
        documento=doc,
        informe_markdown=doc.informe_markdown or "",
        json_analista=doc.json_analista or {},
        metricas=doc.metricas or MetricasAnalisis(),
        mensaje=analisis_resp.mensaje,
        items_evaluados=n_aplicables or len(doc.preguntas_seguimiento or []),
        items_excluidos_otro_tipo=n_excluidos,
    )


@router.post("/{sesion_id}/mensaje", response_model=MensajeResponse)
def enviar_mensaje(sesion_id: str, body: MensajeRequest) -> MensajeResponse:
    try:
        sesion, respuesta = procesar_mensaje(sesion_id, body.mensaje)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except _LLM_ERRORS as exc:
        raise _http_from_llm(exc) from exc
    return MensajeResponse(sesion=sesion, respuesta=respuesta)


@router.get("/{sesion_id}", response_model=SesionCompliance)
def obtener_sesion(sesion_id: str) -> SesionCompliance:
    return _require_sesion(sesion_id)


@router.put("/{sesion_id}", response_model=SesionCompliance)
def restaurar_sesion(sesion_id: str, body: SesionCompliance) -> SesionCompliance:
    """Rehidrata una sesión (p. ej. snapshot en cache del portal) en esta instancia."""
    snapshot = body.model_copy(update={"id": sesion_id})
    return session_store.guardar_sesion(sesion_id, snapshot)


@router.get("/{sesion_id}/checklist")
def obtener_checklist(sesion_id: str):
    sesion = _require_sesion(sesion_id)
    if sesion.estado == EstadoSesion.CONFIGURANDO or not sesion.checklist_slots:
        raise HTTPException(
            status_code=409,
            detail="La sesión aún no está configurada (falta nomenclatura/modalidad/tipo).",
        )
    return {
        "modalidad": sesion.modalidad,
        "tipo_contratacion": sesion.tipo_contratacion,
        "slots": sesion.checklist_slots,
        "pendientes": _slots_pendientes(sesion),
    }


@router.post("/{sesion_id}/documentos", response_model=AnalisisDocumentoResponse)
async def cargar_documento(
    sesion_id: str,
    archivo: UploadFile = File(...),
    tipo_documento: TipoDocumento | None = Query(default=None),
    tipo_documento_form: TipoDocumento | None = Form(default=None, alias="tipo_documento"),
) -> AnalisisDocumentoResponse:
    tipo = tipo_documento if tipo_documento is not None else tipo_documento_form
    if tipo is None:
        raise HTTPException(
            status_code=422,
            detail="Se requiere tipo_documento (query o form).",
        )

    sesion = _require_sesion(sesion_id)
    if sesion.estado != EstadoSesion.ACTIVA or not sesion.modalidad or not sesion.tipo_contratacion:
        raise HTTPException(
            status_code=409,
            detail="Configura nomenclatura, modalidad y tipo antes de cargar documentos.",
        )

    agent = get_modalidad_agent(sesion.modalidad, sesion.tipo_contratacion)
    requisitos = agent.requisitos(tipo)
    cuest = cargar_cuestionario(sesion.modalidad, tipo)
    tipo_ct = sesion.tipo_contratacion.value if sesion.tipo_contratacion else ""
    items_llm = list(cuest.items) if cuest and cuest.items else []
    items_excluidos: list = []
    if cuest and cuest.items and tipo_ct:
        items_llm, items_excluidos = particionar_items_por_tipo(cuest.items, tipo_ct)
    # Solo preguntas del tipo de contrato: las de otro tipo no van al LLM ni al informe.
    preguntas = (
        [it.texto for it in items_llm]
        if items_llm
        else (agent.preguntas(tipo) if not (cuest and cuest.items) else [])
    )
    cuestionario_md = (
        formato_cuestionario_compacto(cuest, items=items_llm)
        if cuest and items_llm
        else None
    )
    cuestionario_items = (
        [
            {
                "codigo": it.codigo,
                "texto": it.texto,
                "rango_criticidad": it.rango_criticidad,
                "fundamento_legal": it.fundamento_legal,
            }
            for it in items_llm
        ]
        if items_llm
        else None
    )

    nombre_seguro = sanitizar_nombre_archivo(archivo.filename)
    try:
        suffix = validar_extension_upload(nombre_seguro)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    limite = max_upload_bytes()
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            total = 0
            while True:
                chunk = await archivo.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > limite:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Archivo demasiado grande. Máximo "
                            f"{limite // (1024 * 1024)} MB."
                        ),
                    )
                tmp.write(chunk)

        procesado = procesar_archivo(tmp_path, nombre_seguro)
        docs_previos = _memoria_expediente(sesion)
        analisis = analizar_documento(
            tipo,
            procesado,
            requisitos,
            system_experto=agent.system_prompt_experto(),
            preguntas_preestablecidas=preguntas,
            cuestionario_markdown=cuestionario_md,
            cuestionario_items=cuestionario_items,
            modalidad=sesion.modalidad,
            nomenclatura=sesion.nomenclatura,
            docs_previos_resumen=docs_previos,
            nombre_archivo=nombre_seguro,
            tipo_contratacion=tipo_ct,
            rubrica_prefill=None,
            items_orden_completo=[it.codigo for it in items_llm] if items_llm else None,
        )

        preg_objs: list[PreguntaSeguimiento] = []
        tipo_coincide = _as_bool(analisis.get("tipo_coincide"), default=True)
        tipo_detectado_raw = analisis.get("tipo_detectado")
        tipo_detectado = (
            str(tipo_detectado_raw).strip() if tipo_detectado_raw not in (None, "") else None
        )

        # Solo ítems aplicables al tipo de contrato de la sesión.
        if tipo_coincide:
            seen_q: set[str] = set()
            fuente_items = items_llm if items_llm else []
            if fuente_items:
                for it in fuente_items:
                    t = it.texto.strip()
                    if not t or t in seen_q:
                        continue
                    seen_q.add(t)
                    preg_objs.append(
                        PreguntaSeguimiento(
                            id=f"q{len(preg_objs) + 1}",
                            texto=t,
                            codigo_pregunta=it.codigo,
                            fundamento_legal=it.fundamento_legal,
                            rango_criticidad=it.rango_criticidad,
                            accion_legal=it.accion_legal,
                            advertencia_gerencia=it.advertencia_gerencia,
                        )
                    )
            else:
                for texto in preguntas:
                    t = str(texto).strip()
                    if not t or t in seen_q:
                        continue
                    seen_q.add(t)
                    preg_objs.append(
                        PreguntaSeguimiento(id=f"q{len(preg_objs) + 1}", texto=t)
                    )
            aplicar_rubrica_agente(
                preg_objs,
                analisis.get("rubrica") or analisis.get("respuestas_cuestionario"),
            )
            if fuente_items:
                verificar_integridad_rubrica(
                    preg_objs,
                    codigos_oficiales={it.codigo for it in fuente_items if it.codigo},
                )

        observaciones = _parse_observaciones(
            analisis.get("observaciones") or analisis.get("hallazgos")
        )
        estatus_global = calcular_estatus_desde_preguntas(
            preg_objs, tipo_coincide=tipo_coincide
        )
        # Observaciones críticas de identidad pueden forzar Rojo más abajo.
        cumple = estatus_global == "Verde"
        if not tipo_coincide:
            cumple = False
            estatus_global = "Rojo"
            observaciones = _asegurar_obs_tipo_incorrecto(
                observaciones,
                tipo_declarado=tipo,
                tipo_detectado=tipo_detectado,
            )

        hechos_raw = analisis.get("hechos_clave") or {}
        nom_doc = None
        if isinstance(hechos_raw, dict):
            raw_nom = hechos_raw.get("nomenclatura_encontrada")
            if raw_nom not in (None, ""):
                nom_doc = str(raw_nom).strip()
        observaciones = _asegurar_obs_nomenclatura(
            observaciones,
            nomenclatura_sesion=sesion.nomenclatura,
            nomenclatura_encontrada=nom_doc,
        )
        if any(
            "nomenclatura" in o.descripcion.lower() and o.severidad == "critica"
            for o in observaciones
        ):
            if estatus_global != "Rojo":
                estatus_global = "Rojo"
            cumple = False

        canon = procesado.get("canonico") or {}
        extraccion = ExtraccionMeta(
            metodo=str(canon.get("metodo") or procesado.get("modo") or "texto"),
            paginas=int(canon.get("paginas") or 0),
            bloques=int(canon.get("bloques") or 0),
            advertencias=list(canon.get("advertencias") or []),
            cobertura_ocr=canon.get("cobertura_ocr"),
        )
        # Truncar texto persistido para no inflar el store (máx ~80k chars)
        texto_ext = str(procesado.get("texto_extraido") or "")
        if len(texto_ext) > 80000:
            texto_ext = texto_ext[:80000] + "\n…[truncado en persistencia]"

        hallazgos = construir_hallazgos_analista(preg_objs) if preg_objs else []
        tipo_label = tipo_ct or (
            sesion.tipo_contratacion.value if sesion.tipo_contratacion else "N/D"
        )
        json_analista = construir_json_analista(
            nombre_documento=nombre_seguro,
            tipo_contrato=tipo_label,
            estatus_global=estatus_global or "Amarillo",
            hallazgos=hallazgos,
        )
        informe_usuario = (
            construir_informe_usuario_markdown(
                nombre_documento=nombre_seguro,
                tipo_contrato=tipo_label,
                estatus_global=estatus_global or "Amarillo",
                preguntas=preg_objs,
            )
            if preg_objs and tipo_coincide
            else str(analisis.get("informe_markdown") or "")
        )
        met_raw = analisis.get("metricas") or {}
        metricas = MetricasAnalisis(
            latencia_ms=int(met_raw.get("latencia_ms") or 0),
            tokens_prompt=int(met_raw.get("tokens_prompt") or 0),
            tokens_completion=int(met_raw.get("tokens_completion") or 0),
            tokens_total=int(met_raw.get("tokens_total") or 0),
            llamadas_llm=int(met_raw.get("llamadas_llm") or 0),
        )

        documento = DocumentoAnalizado(
            id=str(uuid.uuid4()),
            tipo=tipo,
            nombre_archivo=nombre_seguro,
            resumen=str(analisis.get("resumen", "")),
            observaciones=observaciones,
            cumple=cumple,
            fecha_analisis=datetime.now(timezone.utc),
            informe_markdown=informe_usuario,
            preguntas_seguimiento=preg_objs,
            hallazgos=hallazgos,
            json_analista=json_analista,
            metricas=metricas,
            tipo_coincide=tipo_coincide,
            tipo_detectado=tipo_detectado,
            texto_extraido=texto_ext,
            extraccion=extraccion,
            hechos_clave=_parse_hechos_clave(analisis.get("hechos_clave")),
            estatus_global=estatus_global,  # type: ignore[arg-type]
        )
        sincronizar_informe_documento(documento)

        sesion.documentos_analizados.append(documento)
        # Solo marcar el slot como auditado si el archivo es del tipo declarado
        if tipo_coincide:
            for slot in sesion.checklist_slots:
                if slot.tipo_documento == tipo:
                    slot.auditado = True

        pendientes = _slots_pendientes(sesion)
        sesion.documento_en_cuestionario = None
        if not tipo_coincide:
            documento.preguntas_seguimiento = []
            msg = (
                f"Documento incorrecto: declaraste {tipo.value}, pero el archivo "
                f"parece ser {tipo_detectado or 'de otro tipo'}. "
                "El slot NO queda marcado como auditado: puedes volver a subir "
                f"el «{tipo.value}» correcto, o cargar este archivo eligiendo "
                "el tipo que realmente es. "
                f"Informe en /documentos/{documento.id}/informe."
            )
        else:
            estatus_txt = documento.estatus_global or (
                "Verde" if documento.cumple else "Amarillo"
            )
            msg = (
                f"Revisión finalizada de «{nombre_seguro}» "
                f"(estatus {estatus_txt}). "
                f"Informe en /documentos/{documento.id}/informe "
                f"(también .pdf / .docx)."
            )
            if pendientes:
                msg += f" Documentos aún sin revisar: {len(pendientes)}."
            rubrica_txt = resumen_rubrica_chat(documento)
            if rubrica_txt:
                msg += "\n\n" + rubrica_txt

        session_store.guardar_sesion(sesion.id, sesion)
        return AnalisisDocumentoResponse(
            documento=documento,
            mensaje=msg,
            slots_pendientes=pendientes,
            metricas=documento.metricas,
            json_analista=documento.json_analista,
        )
    except _LLM_ERRORS as exc:
        raise _http_from_llm(exc) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "El modelo devolvió una respuesta incompleta al analizar el "
                "documento. Reintenta la carga; si persiste, prueba con un "
                "archivo más liviano."
            ),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@router.post("/{sesion_id}/juridico", response_model=JuridicoResponse)
def revision_juridica(sesion_id: str, body: JuridicoRequest) -> JuridicoResponse:
    sesion = _require_sesion(sesion_id)
    try:
        dictamen = ejecutar_dictamen_juridico(
            sesion,
            mensaje=body.mensaje,
            documento_ids=body.documento_ids,
            forzar_final=body.forzar_final,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except _LLM_ERRORS as exc:
        raise _http_from_llm(exc) from exc

    session_store.guardar_sesion(sesion.id, sesion)
    return JuridicoResponse(
        sesion=sesion,
        dictamen=dictamen,
        respuesta=dictamen.markdown,
    )


@router.get("/{sesion_id}/juridico/{dictamen_id}/informe.pdf")
def get_dictamen_juridico_pdf(sesion_id: str, dictamen_id: str):
    try:
        data, media, filename = export_dictamen_juridico(
            sesion_id, dictamen_id, "pdf"
        )
    except SesionNoEncontradaError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DocumentoNoEncontradoError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(
        content=data,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{sesion_id}/juridico/{dictamen_id}/informe.docx")
def get_dictamen_juridico_docx(sesion_id: str, dictamen_id: str):
    try:
        data, media, filename = export_dictamen_juridico(
            sesion_id, dictamen_id, "docx"
        )
    except SesionNoEncontradaError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DocumentoNoEncontradoError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(
        content=data,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/{sesion_id}/documentos/{doc_id}/respuestas", response_model=DocumentoAnalizado)
def responder_preguntas(
    sesion_id: str,
    doc_id: str,
    body: RespuestasDocumentoRequest,
) -> DocumentoAnalizado:
    """Compat: permite corregir ítems de rúbrica; ya no abre modo Q&A de chat."""
    from app.agents.cuestionario import registrar_respuesta

    sesion = _require_sesion(sesion_id)
    doc = next((d for d in sesion.documentos_analizados if d.id == doc_id), None)
    if doc is None:
        raise HTTPException(status_code=404, detail="Documento no encontrado")

    for pregunta in doc.preguntas_seguimiento:
        if pregunta.id in body.respuestas:
            registrar_respuesta(
                doc,
                body.respuestas[pregunta.id],
                pregunta_id=pregunta.id,
            )

    sincronizar_informe_documento(doc)
    sesion.documento_en_cuestionario = None

    session_store.guardar_sesion(sesion.id, sesion)
    return doc


@router.get("/{sesion_id}/documentos/{doc_id}/informe", response_model=InformeMarkdownResponse)
def get_informe_documento(sesion_id: str, doc_id: str) -> InformeMarkdownResponse:
    try:
        md = informe_documento_markdown(sesion_id, doc_id)
    except SesionNoEncontradaError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DocumentoNoEncontradoError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return InformeMarkdownResponse(informe_markdown=md)


@router.get("/{sesion_id}/documentos/{doc_id}/informe.pdf")
def get_informe_documento_pdf(sesion_id: str, doc_id: str):
    try:
        data, media, filename = export_documento(sesion_id, doc_id, "pdf")
    except (SesionNoEncontradaError, DocumentoNoEncontradoError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(
        content=data,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{sesion_id}/documentos/{doc_id}/informe.docx")
def get_informe_documento_docx(sesion_id: str, doc_id: str):
    try:
        data, media, filename = export_documento(sesion_id, doc_id, "docx")
    except (SesionNoEncontradaError, DocumentoNoEncontradoError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(
        content=data,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{sesion_id}/informe", response_model=InformeMarkdownResponse)
def get_informe_global(
    sesion_id: str,
    usar_llm: bool = Query(default=True),
) -> InformeMarkdownResponse:
    try:
        md = informe_global_markdown(sesion_id, usar_llm=usar_llm)
    except SesionNoEncontradaError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InformeNoDisponibleError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except _LLM_ERRORS as exc:
        raise _http_from_llm(exc) from exc
    return InformeMarkdownResponse(informe_markdown=md)


@router.get("/{sesion_id}/informe.pdf")
def get_informe_global_pdf(
    sesion_id: str,
    usar_llm: bool = Query(default=True),
):
    try:
        data, media, filename = export_global(sesion_id, "pdf", usar_llm=usar_llm)
    except SesionNoEncontradaError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InformeNoDisponibleError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except _LLM_ERRORS as exc:
        raise _http_from_llm(exc) from exc
    return Response(
        content=data,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{sesion_id}/informe.docx")
def get_informe_global_docx(
    sesion_id: str,
    usar_llm: bool = Query(default=True),
):
    try:
        data, media, filename = export_global(sesion_id, "docx", usar_llm=usar_llm)
    except SesionNoEncontradaError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InformeNoDisponibleError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except _LLM_ERRORS as exc:
        raise _http_from_llm(exc) from exc
    return Response(
        content=data,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
