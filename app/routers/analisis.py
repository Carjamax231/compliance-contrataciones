"""Análisis de un documento sin estado: no crea ni guarda ninguna sesión."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, UploadFile

from app.agents.cuestionario import informe_documento_descargable
from app.agents.modalities import get_modalidad_agent
from app.models.schemas import (
    AnalisisSinEstadoResponse,
    EstadoSesion,
    Modalidad,
    SesionCompliance,
    TipoContratacion,
    TipoDocumento,
)
from app.routers.sesiones import _analizar_archivo, _subir_y_procesar

router = APIRouter(prefix="/analizar", tags=["analisis"])


@router.post("", response_model=AnalisisSinEstadoResponse)
async def analizar(
    archivo: UploadFile = File(...),
    nomenclatura: str = Form(...),
    modalidad: Modalidad = Form(...),
    tipo_contratacion: TipoContratacion = Form(...),
    tipo_documento: TipoDocumento = Form(...),
) -> AnalisisSinEstadoResponse:
    sesion = SesionCompliance(
        id=str(uuid.uuid4()),
        nomenclatura=nomenclatura,
        modalidad=modalidad,
        tipo_contratacion=tipo_contratacion,
        estado=EstadoSesion.ACTIVA,
        fecha_inicio=datetime.now(timezone.utc),
        checklist_slots=get_modalidad_agent(modalidad, tipo_contratacion).obtener_checklist(),
    )

    def procesar(ruta: str, nombre: str) -> AnalisisSinEstadoResponse:
        documento = _analizar_archivo(sesion, tipo_documento, ruta, nombre)
        # Un solo valor para el resumen y para la respuesta: nunca se contradicen.
        estatus = documento.estatus_global or "Amarillo"
        informe = informe_documento_descargable(
            documento.informe_markdown or "",
            nomenclatura=nomenclatura,
            modalidad=modalidad,
            tipo_contratacion=tipo_contratacion,
            tipo_documento=tipo_documento,
            nombre_archivo=documento.nombre_archivo,
            estatus_global=estatus,
            fecha=documento.fecha_analisis,
            tipo_coincide=documento.tipo_coincide,
            tipo_detectado=documento.tipo_detectado,
        )
        return AnalisisSinEstadoResponse(
            json_analista=documento.json_analista,
            informe_markdown=informe,
            estatus_global=estatus,
            tipo_coincide=documento.tipo_coincide,
            tipo_detectado=documento.tipo_detectado,
            metricas=documento.metricas,
        )

    return await _subir_y_procesar(archivo, procesar)
