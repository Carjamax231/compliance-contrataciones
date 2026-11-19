"""Catálogo oficial de documentos por modalidad × tipo de contratación."""

from __future__ import annotations

from typing import TypedDict

from app.agents.knowledge.preguntas import preguntas_para
from app.models.schemas import Modalidad, TipoContratacion, TipoDocumento

PLACEHOLDER = "(pendiente de definir con basamento legal específico)"

ACTA_INICIO_ELEMENTOS = [
    "Descripción del objeto de la contratación y número del procedimiento",
    "Monto estimado de la contratación",
    "Verificación de la situación legal de las empresas (RNC)",
    "Empresas seleccionadas con Nivel Estimado de Contratación y Calificación Financiera",
    "Razones técnicas que fundamentaron la escogencia de las empresas participantes",
    "Cronograma de ejecución de la modalidad de selección",
    "Firma de los miembros de la Comisión de Contrataciones",
]


class SlotDef(TypedDict):
    tipo_documento: TipoDocumento
    descripcion: str
    elementos_requeridos: list[str]


class RamaKnowledge(TypedDict):
    slots: list[SlotDef]
    preguntas_por_documento: dict[TipoDocumento, list[str]]


class ModalidadKnowledge(TypedDict):
    descripcion: str
    ramas: dict[TipoContratacion, RamaKnowledge]


def _slot(
    tipo: TipoDocumento,
    descripcion: str,
    elementos: list[str] | None = None,
) -> SlotDef:
    return {
        "tipo_documento": tipo,
        "descripcion": descripcion,
        "elementos_requeridos": list(elementos or [PLACEHOLDER]),
    }


def _slots_from(pairs: list[tuple[TipoDocumento, str]], *, acta_inicio_elems: bool = False) -> list[SlotDef]:
    out: list[SlotDef] = []
    for tipo, desc in pairs:
        elems = ACTA_INICIO_ELEMENTOS if (acta_inicio_elems and tipo == TipoDocumento.ACTA_INICIO) else None
        out.append(_slot(tipo, desc, elems))
    return out


def _preguntas_rama(
    modalidad: Modalidad,
    tipo: TipoContratacion,
    slots: list[SlotDef],
) -> dict[TipoDocumento, list[str]]:
    return {
        s["tipo_documento"]: preguntas_para(modalidad, tipo, s["tipo_documento"])
        for s in slots
    }


def _rama_de(slots: list[SlotDef]) -> RamaKnowledge:
    return {
        "slots": slots,
        "preguntas_por_documento": {
            s["tipo_documento"]: preguntas_para(
                Modalidad.CA_ACTO_UNICO_APERTURA_UNICA,
                TipoContratacion.BIENES,
                s["tipo_documento"],
            )
            for s in slots
        },
    }


def _rama_para_todos(
    builder,
    *,
    modalidad: Modalidad,
) -> dict[TipoContratacion, RamaKnowledge]:
    out: dict[TipoContratacion, RamaKnowledge] = {}
    for tipo in TipoContratacion:
        base = builder()
        slots = base["slots"]
        out[tipo] = {
            "slots": slots,
            "preguntas_por_documento": _preguntas_rama(modalidad, tipo, slots),
        }
    return out


# --- Listados oficiales (reemplazo total; no combinar con catálogo legacy) ---

def _rama_ca_apertura_unica() -> RamaKnowledge:
    return _rama_de(
        _slots_from(
            [
                (TipoDocumento.ACTIVIDADES_PREVIAS, "Actividades Previas"),
                (TipoDocumento.ACTA_INICIO, "Acta de Inicio"),
                (TipoDocumento.PLIEGO_CONDICIONES, "Pliego de Condiciones"),
                (TipoDocumento.LLAMADO, "Llamado a participar"),
                (TipoDocumento.ACTA_RECEPCION_SOBRES, "Acta de Recepción de Sobres"),
                (TipoDocumento.ACTA_APERTURA_SOBRES, "Acta de Apertura de Sobres"),
                (TipoDocumento.OFERTAS, "Ofertas"),
                (
                    TipoDocumento.INFORME_EVALUACION_RECOMENDACION,
                    "Informe de Evaluación y Recomendación",
                ),
                (TipoDocumento.ADJUDICACION_O_EQUIVALENTE, "Adjudicación o Equivalente"),
                (TipoDocumento.NOTIFICACION_ADJUDICADOS, "Notificación a Adjudicados"),
                (TipoDocumento.NOTIFICACION_NO_ADJUDICADOS, "Notificación a No Adjudicados"),
                (TipoDocumento.CONTRATO, "Contrato (u Orden de Compra/Servicio)"),
            ],
            acta_inicio_elems=True,
        )
    )


def _rama_ca_apertura_diferida() -> RamaKnowledge:
    return _rama_de(
        _slots_from(
            [
                (TipoDocumento.ACTIVIDADES_PREVIAS, "Actividades Previas"),
                (TipoDocumento.ACTA_INICIO, "Acta de Inicio"),
                (TipoDocumento.PLIEGO_CONDICIONES, "Pliego de Condiciones"),
                (TipoDocumento.LLAMADO, "Llamado"),
                (
                    TipoDocumento.ACTA_RECEPCION_MV_CALIF_OFERTAS,
                    "Acta de Recepción de Manifestaciones de Voluntad, Documentos de Calificación y Ofertas",
                ),
                (
                    TipoDocumento.ACTA_APERTURA_MV_CALIFICACION,
                    "Acta de Apertura de Manifestaciones de Voluntad y Documentos de Calificación",
                ),
                (TipoDocumento.INFORME_CALIFICACION, "Informe de Calificación"),
                (TipoDocumento.NOTIFICACION_CALIFICACION, "Notificación de Calificación"),
                (
                    TipoDocumento.ACTA_APERTURA_OFERTAS_DEVOLUCION,
                    "Acta de Apertura de Ofertas y Devolución de Sobres",
                ),
                (TipoDocumento.OFERTAS, "Ofertas"),
                (
                    TipoDocumento.GARANTIA_SOSTENIMIENTO_OFERTA,
                    "Garantía de Sostenimiento de la Oferta",
                ),
                (TipoDocumento.INFORME_RECOMENDACION, "Informe de Recomendación"),
                (TipoDocumento.ADJUDICACION_O_EQUIVALENTE, "Adjudicación o Equivalente"),
                (TipoDocumento.NOTIFICACION_INTERESADOS, "Notificación a Interesados"),
                (
                    TipoDocumento.CONTRATO,
                    "Contrato, Orden de Compra u Orden de Servicio",
                ),
                (TipoDocumento.RESPONSABILIDAD_SOCIAL, "Responsabilidad Social"),
            ]
        )
    )


def _rama_ca_acto_separado() -> RamaKnowledge:
    return _rama_de(
        _slots_from(
            [
                (TipoDocumento.ACTIVIDADES_PREVIAS, "Actividades Previas"),
                (TipoDocumento.ACTA_INICIO, "Acta de Inicio"),
                (TipoDocumento.PLIEGO_CONDICIONES, "Pliego de Condiciones"),
                (TipoDocumento.LLAMADO, "Llamado"),
                (
                    TipoDocumento.ACTA_RECEPCION_MV_CALIFICACION,
                    "Acta de Recepción de Manifestaciones de Voluntad y Documentos de Calificación",
                ),
                (
                    TipoDocumento.ACTA_APERTURA_MV_CALIFICACION,
                    "Acta de Apertura de Manifestaciones de Voluntad y Documentos de Calificación",
                ),
                (TipoDocumento.INFORME_CALIFICACION, "Informe de Calificación"),
                (TipoDocumento.NOTIFICACION_CALIFICACION, "Notificación de Calificación"),
                (TipoDocumento.ACTA_RECEPCION_OFERTAS, "Acta de Recepción de Ofertas"),
                (TipoDocumento.ACTA_APERTURA_OFERTAS, "Acta de Apertura de Ofertas"),
                (TipoDocumento.OFERTAS, "Ofertas"),
                (
                    TipoDocumento.GARANTIA_SOSTENIMIENTO_OFERTA,
                    "Garantía de Sostenimiento de la Oferta",
                ),
                (TipoDocumento.INFORME_RECOMENDACION, "Informe de Recomendación"),
                (TipoDocumento.ADJUDICACION_O_EQUIVALENTE, "Adjudicación o Equivalente"),
                (TipoDocumento.NOTIFICACION_INTERESADOS, "Notificación a Interesados"),
                (
                    TipoDocumento.CONTRATO,
                    "Contrato, Orden de Compra u Orden de Servicio",
                ),
            ]
        )
    )


def _rama_concurso_cerrado() -> RamaKnowledge:
    return _rama_de(
        _slots_from(
            [
                (TipoDocumento.ACTIVIDADES_PREVIAS, "Actividades Previas"),
                (TipoDocumento.ACTA_INICIO, "Acta de Inicio"),
                (TipoDocumento.PLIEGO_CONDICIONES, "Pliego de Condiciones"),
                (TipoDocumento.INVITACIONES, "Invitaciones"),
                (
                    TipoDocumento.ACTA_RECEPCION_MV_CALIF_OFERTAS,
                    "Acta de Recepción de Manifestaciones de Voluntad, Calificación / Ofertas",
                ),
                (
                    TipoDocumento.ACTA_APERTURA_MV_CALIFICACION,
                    "Acta de Apertura de Manifestaciones de Voluntad, Calificación / Ofertas",
                ),
                (TipoDocumento.OFERTAS, "Ofertas"),
                (
                    TipoDocumento.GARANTIA_SOSTENIMIENTO_OFERTA,
                    "Garantía de Sostenimiento de la Oferta",
                ),
                (
                    TipoDocumento.INFORME_EVALUACION_RECOMENDACION,
                    "Informe de Evaluación y Recomendación",
                ),
                (TipoDocumento.ADJUDICACION_O_EQUIVALENTE, "Adjudicación o Equivalente"),
                (TipoDocumento.NOTIFICACION_INTERESADOS, "Notificación a Interesados"),
                (
                    TipoDocumento.CONTRATO,
                    "Contrato, Orden de Compra u Orden de Servicio",
                ),
                (TipoDocumento.RESPONSABILIDAD_SOCIAL, "Responsabilidad Social"),
            ]
        )
    )


def _rama_consulta_precio() -> RamaKnowledge:
    return _rama_de(
        _slots_from(
            [
                (TipoDocumento.ACTIVIDADES_PREVIAS, "Actividades Previas"),
                (
                    TipoDocumento.PUNTO_DE_CUENTA,
                    "Documento que Autoriza el Inicio del Procedimiento (Punto de Cuenta)",
                ),
                (TipoDocumento.ACTA_INICIO, "Acta de Inicio"),
                (
                    TipoDocumento.PLIEGO_CONDICIONES,
                    "Pliego de Condiciones / Condiciones de Contratación",
                ),
                (TipoDocumento.INVITACIONES, "Invitaciones"),
                (
                    TipoDocumento.ACTA_RECEPCION_CALIF_OFERTAS,
                    "Acta de Recepción de Documentos de Calificación / Ofertas",
                ),
                (
                    TipoDocumento.ACTA_APERTURA_CALIF_OFERTAS,
                    "Acta de Apertura de Documentos de Calificación / Ofertas",
                ),
                (TipoDocumento.OFERTAS, "Ofertas"),
                (
                    TipoDocumento.GARANTIA_SOSTENIMIENTO_OFERTA,
                    "Garantía de Sostenimiento de la Oferta",
                ),
                (
                    TipoDocumento.INFORME_EVALUACION_RECOMENDACION,
                    "Informe de Evaluación y Recomendación",
                ),
                (
                    TipoDocumento.INFORME_OPINION_COMISION,
                    "Informe de Opinión Comisión de Contrataciones",
                ),
                (TipoDocumento.ADJUDICACION_O_EQUIVALENTE, "Adjudicación o Equivalente"),
                (TipoDocumento.NOTIFICACION_INTERESADOS, "Notificación a Interesados"),
                (
                    TipoDocumento.CONTRATO,
                    "Contrato, Orden de Compra u Orden de Servicio",
                ),
                (TipoDocumento.RESPONSABILIDAD_SOCIAL, "Responsabilidad Social"),
            ]
        )
    )


def _rama_contratacion_directa() -> RamaKnowledge:
    """Listado oficial definitivo (11 documentos)."""
    return _rama_de(
        _slots_from(
            [
                (TipoDocumento.ACTIVIDADES_PREVIAS, "Actividades Previas"),
                (
                    TipoDocumento.ACTO_MOTIVADO_INICIO,
                    "Acto Motivado que Autoriza Inicio Procedimiento",
                ),
                (TipoDocumento.ACTA_INICIO, "Acta de Inicio"),
                (
                    TipoDocumento.PLIEGO_CONDICIONES,
                    "Pliego / Condiciones de Contratación",
                ),
                (TipoDocumento.INVITACIONES, "Invitaciones"),
                (TipoDocumento.OFERTAS, "Ofertas"),
                (
                    TipoDocumento.INFORME_VERIFICACION_RAZONABILIDAD,
                    "Informe de Verificación de Razonabilidad de Precios y Recomendación",
                ),
                (
                    TipoDocumento.INFORME_OPINION_COMISION,
                    "Informe de Opinión Comisión de Contrataciones",
                ),
                (TipoDocumento.ADJUDICACION_O_EQUIVALENTE, "Adjudicación o Equivalente"),
                (
                    TipoDocumento.NOTIFICACION_INTERESADOS,
                    "Notificación de la Adjudicación (solo al beneficiario, art. 126 RLCP)",
                ),
                (
                    TipoDocumento.CONTRATO,
                    "Contrato, Orden de Compra u Orden de Servicio",
                ),
            ]
        )
    )


def _rama_excluidas() -> RamaKnowledge:
    # El listado fuente duplicaba «Notificación a Interesados» (10 y 11); se conserva una sola.
    return _rama_de(
        _slots_from(
            [
                (TipoDocumento.ACTIVIDADES_PREVIAS, "Actividades Previas"),
                (
                    TipoDocumento.ACTO_MOTIVADO_INICIO,
                    "Acto Motivado que Autoriza Inicio Procedimiento",
                ),
                (TipoDocumento.ACTA_INICIO, "Acta de Inicio"),
                (TipoDocumento.CONDICIONES_CONTRATACION, "Condiciones de Contratación"),
                (TipoDocumento.INVITACIONES, "Invitaciones"),
                (TipoDocumento.OFERTAS, "Ofertas"),
                (
                    TipoDocumento.GARANTIA_SOSTENIMIENTO_OFERTA,
                    "Garantía de Sostenimiento de la Oferta",
                ),
                (
                    TipoDocumento.INFORME_VERIFICACION_ADJUDICACION,
                    "Informe de Verificación y Recomendación para la Adjudicación",
                ),
                (TipoDocumento.ADJUDICACION_ACTO_MOTIVADO, "Adjudicación (Acto Motivado)"),
                (TipoDocumento.NOTIFICACION_INTERESADOS, "Notificación a Interesados"),
                (
                    TipoDocumento.CONTRATO,
                    "Contrato, Orden de Compra u Orden de Servicio",
                ),
            ]
        )
    )


KNOWLEDGE: dict[Modalidad, ModalidadKnowledge] = {
    Modalidad.CA_ACTO_UNICO_APERTURA_UNICA: {
        "descripcion": "Concurso Abierto - Acto Único Apertura Única.",
        "ramas": _rama_para_todos(
            _rama_ca_apertura_unica,
            modalidad=Modalidad.CA_ACTO_UNICO_APERTURA_UNICA,
        ),
    },
    Modalidad.CA_ACTO_UNICO_APERTURA_DIFERIDA: {
        "descripcion": "Concurso Abierto - Acto Único Apertura Diferida.",
        "ramas": _rama_para_todos(
            _rama_ca_apertura_diferida,
            modalidad=Modalidad.CA_ACTO_UNICO_APERTURA_DIFERIDA,
        ),
    },
    Modalidad.CA_ACTO_SEPARADO: {
        "descripcion": "Concurso Abierto - Acto Separado.",
        "ramas": _rama_para_todos(
            _rama_ca_acto_separado,
            modalidad=Modalidad.CA_ACTO_SEPARADO,
        ),
    },
    Modalidad.CONCURSO_CERRADO: {
        "descripcion": "Concurso Cerrado.",
        "ramas": _rama_para_todos(
            _rama_concurso_cerrado,
            modalidad=Modalidad.CONCURSO_CERRADO,
        ),
    },
    Modalidad.CONSULTA_PRECIO: {
        "descripcion": "Consulta de Precio.",
        "ramas": _rama_para_todos(
            _rama_consulta_precio,
            modalidad=Modalidad.CONSULTA_PRECIO,
        ),
    },
    Modalidad.CONTRATACION_DIRECTA: {
        "descripcion": "Contratación Directa.",
        "ramas": _rama_para_todos(
            _rama_contratacion_directa,
            modalidad=Modalidad.CONTRATACION_DIRECTA,
        ),
    },
    Modalidad.MODALIDADES_EXCLUIDAS: {
        "descripcion": "Modalidades Excluidas.",
        "ramas": _rama_para_todos(
            _rama_excluidas,
            modalidad=Modalidad.MODALIDADES_EXCLUIDAS,
        ),
    },
}


def get_rama(modalidad: Modalidad, tipo: TipoContratacion) -> RamaKnowledge:
    return KNOWLEDGE[modalidad]["ramas"][tipo]


def get_descripcion_modalidad(modalidad: Modalidad) -> str:
    return KNOWLEDGE[modalidad]["descripcion"]


def etiqueta_modalidad(modalidad: Modalidad) -> str:
    labels = {
        Modalidad.CA_ACTO_UNICO_APERTURA_UNICA: (
            "Concurso Abierto - Acto Único Apertura Única"
        ),
        Modalidad.CA_ACTO_UNICO_APERTURA_DIFERIDA: (
            "Concurso Abierto - Acto Único Apertura Diferida"
        ),
        Modalidad.CA_ACTO_SEPARADO: "Concurso Abierto - Acto Separado",
        Modalidad.CONCURSO_CERRADO: "Concurso Cerrado",
        Modalidad.CONSULTA_PRECIO: "Consulta de Precio",
        Modalidad.CONTRATACION_DIRECTA: "Contratación Directa",
        Modalidad.MODALIDADES_EXCLUIDAS: "Modalidades Excluidas",
    }
    return labels[modalidad]


def etiqueta_tipo_contratacion(tipo: TipoContratacion) -> str:
    labels = {
        TipoContratacion.BIENES: "Bienes",
        TipoContratacion.OBRAS: "Obra",
        TipoContratacion.SERVICIOS: "Servicio",
    }
    return labels[tipo]


def etiqueta_documento(tipo: TipoDocumento) -> str:
    """Etiqueta humana por defecto (el slot.descripcion puede ser más específico)."""
    labels = {
        TipoDocumento.ACTIVIDADES_PREVIAS: "Actividades Previas",
        TipoDocumento.ACTA_INICIO: "Acta de Inicio",
        TipoDocumento.PLIEGO_CONDICIONES: "Pliego de Condiciones",
        TipoDocumento.CONDICIONES_CONTRATACION: "Condiciones de Contratación",
        TipoDocumento.LLAMADO: "Llamado a participar",
        TipoDocumento.INVITACIONES: "Invitaciones",
        TipoDocumento.PUNTO_DE_CUENTA: (
            "Documento que Autoriza el Inicio del Procedimiento (Punto de Cuenta)"
        ),
        TipoDocumento.ACTO_MOTIVADO_INICIO: (
            "Acto Motivado que Autoriza Inicio Procedimiento"
        ),
        TipoDocumento.ACTA_RECEPCION_SOBRES: "Acta de Recepción de Sobres",
        TipoDocumento.ACTA_APERTURA_SOBRES: "Acta de Apertura de Sobres",
        TipoDocumento.ACTA_RECEPCION_MV_CALIF_OFERTAS: (
            "Acta de Recepción de Manifestaciones de Voluntad, Documentos de Calificación y Ofertas"
        ),
        TipoDocumento.ACTA_APERTURA_MV_CALIFICACION: (
            "Acta de Apertura de Manifestaciones de Voluntad y Documentos de Calificación"
        ),
        TipoDocumento.INFORME_CALIFICACION: "Informe de Calificación",
        TipoDocumento.NOTIFICACION_CALIFICACION: "Notificación de Calificación",
        TipoDocumento.ACTA_APERTURA_OFERTAS_DEVOLUCION: (
            "Acta de Apertura de Ofertas y Devolución de Sobres"
        ),
        TipoDocumento.ACTA_RECEPCION_MV_CALIFICACION: (
            "Acta de Recepción de Manifestaciones de Voluntad y Documentos de Calificación"
        ),
        TipoDocumento.ACTA_RECEPCION_OFERTAS: "Acta de Recepción de Ofertas",
        TipoDocumento.ACTA_APERTURA_OFERTAS: "Acta de Apertura de Ofertas",
        TipoDocumento.ACTA_RECEPCION_CALIF_OFERTAS: (
            "Acta de Recepción de Documentos de Calificación / Ofertas"
        ),
        TipoDocumento.ACTA_APERTURA_CALIF_OFERTAS: (
            "Acta de Apertura de Documentos de Calificación / Ofertas"
        ),
        TipoDocumento.OFERTAS: "Ofertas",
        TipoDocumento.GARANTIA_SOSTENIMIENTO_OFERTA: (
            "Garantía de Sostenimiento de la Oferta"
        ),
        TipoDocumento.INFORME_EVALUACION_RECOMENDACION: (
            "Informe de Evaluación y Recomendación"
        ),
        TipoDocumento.INFORME_RECOMENDACION: "Informe de Recomendación",
        TipoDocumento.INFORME_VERIFICACION_RAZONABILIDAD: (
            "Informe de Verificación de Razonabilidad de Precios y Recomendación"
        ),
        TipoDocumento.INFORME_VERIFICACION_ADJUDICACION: (
            "Informe de Verificación y Recomendación para la Adjudicación"
        ),
        TipoDocumento.INFORME_OPINION_COMISION: (
            "Informe de Opinión Comisión de Contrataciones"
        ),
        TipoDocumento.ADJUDICACION_O_EQUIVALENTE: "Adjudicación o Equivalente",
        TipoDocumento.ADJUDICACION_ACTO_MOTIVADO: "Adjudicación (Acto Motivado)",
        TipoDocumento.NOTIFICACION_ADJUDICADOS: "Notificación a Adjudicados",
        TipoDocumento.NOTIFICACION_NO_ADJUDICADOS: "Notificación a No Adjudicados",
        TipoDocumento.NOTIFICACION_INTERESADOS: "Notificación a Interesados",
        TipoDocumento.CONTRATO: "Contrato (u Orden de Compra/Servicio)",
        TipoDocumento.RESPONSABILIDAD_SOCIAL: "Responsabilidad Social",
        TipoDocumento.OTROS: "Otros documentos",
    }
    return labels.get(tipo, tipo.value.replace("_", " ").title())
