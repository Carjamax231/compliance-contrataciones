"""Catálogos para los clientes: modalidades, tipos de contratación y checklists."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter

from app.agents.knowledge import (
    etiqueta_documento,
    etiqueta_modalidad,
    etiqueta_tipo_contratacion,
)
from app.agents.modalities import get_modalidad_agent
from app.models.schemas import Modalidad, TipoContratacion

router = APIRouter(prefix="/catalogo", tags=["catalogo"])

_CUESTIONARIOS = Path(__file__).resolve().parents[2] / "cuestionarios"
# Modalidades revisadas por el equipo legal con un expediente completo.
_VALIDADAS = {
    Modalidad.CA_ACTO_UNICO_APERTURA_UNICA,
    Modalidad.CONTRATACION_DIRECTA,
}


def _estado(modalidad: Modalidad) -> str:
    if modalidad in _VALIDADAS:
        return "disponible"
    carpeta = _CUESTIONARIOS / modalidad.value
    if carpeta.is_dir() and any(carpeta.glob("*.md")):
        return "sin_validar"
    return "sin_cuestionarios"


@lru_cache(maxsize=1)
def _catalogo() -> dict:
    modalidades = []
    for m in Modalidad:
        # El checklist no depende del tipo de contratación.
        slots = get_modalidad_agent(m, TipoContratacion.BIENES).obtener_checklist()
        modalidades.append({
            "valor": m.value,
            "nombre": etiqueta_modalidad(m),
            "estado": _estado(m),
            "checklist": [
                {
                    "tipo_documento": s.tipo_documento.value,
                    "nombre": s.descripcion or etiqueta_documento(s.tipo_documento),
                }
                for s in slots
            ],
        })
    return {
        "modalidades": modalidades,
        "tipos_contratacion": [
            {"valor": t.value, "nombre": etiqueta_tipo_contratacion(t)} for t in TipoContratacion
        ],
    }


@router.get("")
def catalogo() -> dict:
    """`estado`: disponible, sin_validar (cuestionarios sin revisión legal) o sin_cuestionarios."""
    return _catalogo()
