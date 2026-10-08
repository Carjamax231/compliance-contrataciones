"""Esquemas Pydantic del dominio de compliance multi-modalidad."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class TipoContratacion(str, Enum):
    """Objeto de la contratación: bienes, obras o servicios."""

    BIENES = "BIENES"
    OBRAS = "OBRAS"
    SERVICIOS = "SERVICIOS"


class Modalidad(str, Enum):
    """Modalidades de selección / contratación del RLCP."""

    CA_ACTO_UNICO_APERTURA_UNICA = "CA_ACTO_UNICO_APERTURA_UNICA"
    CA_ACTO_UNICO_APERTURA_DIFERIDA = "CA_ACTO_UNICO_APERTURA_DIFERIDA"
    CA_ACTO_SEPARADO = "CA_ACTO_SEPARADO"
    CONCURSO_CERRADO = "CONCURSO_CERRADO"
    CONSULTA_PRECIO = "CONSULTA_PRECIO"
    CONTRATACION_DIRECTA = "CONTRATACION_DIRECTA"
    MODALIDADES_EXCLUIDAS = "MODALIDADES_EXCLUIDAS"


class EstadoSesion(str, Enum):
    """Estado del ciclo de vida de una sesión de compliance."""

    CONFIGURANDO = "CONFIGURANDO"
    ACTIVA = "ACTIVA"
    CERRADA = "CERRADA"


class NivelRiesgo(str, Enum):
    """Semáforo de riesgo del expediente (derivado de hallazgos)."""

    SIN_HALLAZGOS = "SIN_HALLAZGOS"
    OBSERVACIONES = "OBSERVACIONES"
    RIESGO_ALTO = "RIESGO_ALTO"


class RolAgente(str, Enum):
    """Rol dentro de la dupla por modalidad."""

    ANALISTA = "ANALISTA"
    JURIDICO = "JURIDICO"


class AlcanceDictamen(str, Enum):
    """Si el dictamen cubre el expediente completo o solo lo auditado hasta ahora."""

    PARCIAL = "PARCIAL"
    FINAL = "FINAL"


class TipoDocumento(str, Enum):
    """Catálogo oficial de documentos auditables por modalidad."""

    ACTIVIDADES_PREVIAS = "ACTIVIDADES_PREVIAS"
    ACTA_INICIO = "ACTA_INICIO"
    PLIEGO_CONDICIONES = "PLIEGO_CONDICIONES"
    CONDICIONES_CONTRATACION = "CONDICIONES_CONTRATACION"
    LLAMADO = "LLAMADO"
    INVITACIONES = "INVITACIONES"
    PUNTO_DE_CUENTA = "PUNTO_DE_CUENTA"
    ACTO_MOTIVADO_INICIO = "ACTO_MOTIVADO_INICIO"

    ACTA_RECEPCION_SOBRES = "ACTA_RECEPCION_SOBRES"
    ACTA_APERTURA_SOBRES = "ACTA_APERTURA_SOBRES"

    ACTA_RECEPCION_MV_CALIF_OFERTAS = "ACTA_RECEPCION_MV_CALIF_OFERTAS"
    ACTA_APERTURA_MV_CALIFICACION = "ACTA_APERTURA_MV_CALIFICACION"
    INFORME_CALIFICACION = "INFORME_CALIFICACION"
    NOTIFICACION_CALIFICACION = "NOTIFICACION_CALIFICACION"
    ACTA_APERTURA_OFERTAS_DEVOLUCION = "ACTA_APERTURA_OFERTAS_DEVOLUCION"

    ACTA_RECEPCION_MV_CALIFICACION = "ACTA_RECEPCION_MV_CALIFICACION"
    ACTA_RECEPCION_OFERTAS = "ACTA_RECEPCION_OFERTAS"
    ACTA_APERTURA_OFERTAS = "ACTA_APERTURA_OFERTAS"

    ACTA_RECEPCION_CALIF_OFERTAS = "ACTA_RECEPCION_CALIF_OFERTAS"
    ACTA_APERTURA_CALIF_OFERTAS = "ACTA_APERTURA_CALIF_OFERTAS"

    OFERTAS = "OFERTAS"
    GARANTIA_SOSTENIMIENTO_OFERTA = "GARANTIA_SOSTENIMIENTO_OFERTA"

    INFORME_EVALUACION_RECOMENDACION = "INFORME_EVALUACION_RECOMENDACION"
    INFORME_RECOMENDACION = "INFORME_RECOMENDACION"
    INFORME_VERIFICACION_RAZONABILIDAD = "INFORME_VERIFICACION_RAZONABILIDAD"
    INFORME_VERIFICACION_ADJUDICACION = "INFORME_VERIFICACION_ADJUDICACION"
    INFORME_OPINION_COMISION = "INFORME_OPINION_COMISION"

    ADJUDICACION_O_EQUIVALENTE = "ADJUDICACION_O_EQUIVALENTE"
    ADJUDICACION_ACTO_MOTIVADO = "ADJUDICACION_ACTO_MOTIVADO"

    NOTIFICACION_ADJUDICADOS = "NOTIFICACION_ADJUDICADOS"
    NOTIFICACION_NO_ADJUDICADOS = "NOTIFICACION_NO_ADJUDICADOS"
    NOTIFICACION_INTERESADOS = "NOTIFICACION_INTERESADOS"

    CONTRATO = "CONTRATO"
    RESPONSABILIDAD_SOCIAL = "RESPONSABILIDAD_SOCIAL"

    OTROS = "OTROS"


class RolMensaje(str, Enum):
    """Rol en el historial conversacional."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Observacion(BaseModel):
    """Hallazgo de auditoría sobre un documento."""

    severidad: Literal["info", "advertencia", "critica"]
    descripcion: str
    subsanacion: str | None = None
    ref: str | None = None
    codigo_pregunta: str | None = None
    fundamento_legal: str | None = None
    rango_criticidad: str | None = None
    accion_legal: str | None = None
    advertencia_gerencia: str | None = None


class HallazgoAnalista(BaseModel):
    """Hallazgo estructurado (Parte 2 Mariana) para Nest / agente Jurídico."""

    codigo_pregunta: str
    pregunta_evaluada: str
    fundamento_legal: str | None = None
    rango_criticidad: str | None = None
    accion_legal: str | None = None
    advertencia_gerencia: str | None = None
    estado: Literal["no", "parcial", "no_consta"] | None = None
    respuesta: str | None = None
    ref: str | None = None


class MetricasAnalisis(BaseModel):
    """Latencia y tokens de un análisis (sandbox / observabilidad)."""

    latencia_ms: int = 0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_total: int = 0
    llamadas_llm: int = 0


class PreguntaSeguimiento(BaseModel):
    """Ítem de rúbrica interna: el Analista responde contra el documento."""

    id: str
    texto: str
    respondida: bool = False
    respuesta: str | None = None
    # si | no | parcial | na | no_consta
    estado: Literal["si", "no", "parcial", "na", "no_consta"] | None = None
    ref: str | None = None
    respondida_por: Literal["agente", "usuario"] = "agente"
    codigo_pregunta: str | None = None
    fundamento_legal: str | None = None
    rango_criticidad: str | None = None
    accion_legal: str | None = None
    advertencia_gerencia: str | None = None


class SlotChecklist(BaseModel):
    """Slot sugerido del checklist dinámico (no obligatorio ni ordenado)."""

    tipo_documento: TipoDocumento
    descripcion: str
    elementos_requeridos: list[str] = Field(default_factory=list)
    auditado: bool = False


class MensajeChat(BaseModel):
    """Turno del historial de chat de la sesión."""

    rol: RolMensaje
    contenido: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ExtraccionMeta(BaseModel):
    """Metadatos de extracción previa al análisis LLM."""

    metodo: str = "texto"
    paginas: int = 0
    bloques: int = 0
    advertencias: list[str] = Field(default_factory=list)
    cobertura_ocr: float | None = None


class MontoClave(BaseModel):
    """Monto relevante extraído del documento (para memoria cruzada)."""

    etiqueta: str
    texto: str = ""
    valor_num: float | None = None
    moneda: str | None = None


class PlazoClave(BaseModel):
    """Plazo o fecha relevante extraída del documento."""

    etiqueta: str
    texto: str


class HechosClave(BaseModel):
    """Hechos estructurados del documento para cross-validation del expediente."""

    montos: list[MontoClave] = Field(default_factory=list)
    plazos: list[PlazoClave] = Field(default_factory=list)
    partes: list[str] = Field(default_factory=list)
    nomenclatura_encontrada: str | None = None
    otros: list[str] = Field(default_factory=list)


class DocumentoAnalizado(BaseModel):
    """Resultado del análisis de un documento cargado a la sesión."""

    id: str
    tipo: TipoDocumento
    nombre_archivo: str
    resumen: str
    observaciones: list[Observacion] = Field(default_factory=list)
    cumple: bool
    fecha_analisis: datetime
    informe_markdown: str = ""
    preguntas_seguimiento: list[PreguntaSeguimiento] = Field(default_factory=list)
    hallazgos: list[HallazgoAnalista] = Field(default_factory=list)
    json_analista: dict | None = None
    metricas: MetricasAnalisis | None = None
    tipo_coincide: bool = True
    tipo_detectado: str | None = None
    texto_extraido: str = ""
    extraccion: ExtraccionMeta | None = None
    hechos_clave: HechosClave = Field(default_factory=HechosClave)
    # Semáforo del Analista: Verde | Amarillo | Rojo
    estatus_global: Literal["Verde", "Amarillo", "Rojo"] | None = None


class DictamenJuridico(BaseModel):
    """Dictamen del agente Jurídico persistido en la sesión."""

    id: str
    fecha: datetime = Field(default_factory=datetime.utcnow)
    alcance: AlcanceDictamen = AlcanceDictamen.PARCIAL
    documento_ids: list[str] = Field(default_factory=list)
    mensaje_usuario: str = ""
    markdown: str = ""
    slots_pendientes: list[TipoDocumento] = Field(default_factory=list)


class SesionCompliance(BaseModel):
    """Sesión conversacional de auditoría de compliance."""

    id: str
    nomenclatura: str | None = None
    modalidad: Modalidad | None = None
    tipo_contratacion: TipoContratacion | None = None
    estado: EstadoSesion = EstadoSesion.CONFIGURANDO
    # Sesiones antiguas pueden no traer este campo; el listado hace fallback.
    fecha_inicio: datetime | None = None
    historial: list[MensajeChat] = Field(default_factory=list)
    checklist_slots: list[SlotChecklist] = Field(default_factory=list)
    documentos_analizados: list[DocumentoAnalizado] = Field(default_factory=list)
    # Documento cuyo cuestionario de seguimiento está activo (uno a la vez).
    documento_en_cuestionario: str | None = None
    # True si el usuario pidió cambiar nomenclatura y aún no envió el código nuevo.
    pendiente_cambio_nomenclatura: bool = False
    dictamenes_juridicos: list[DictamenJuridico] = Field(default_factory=list)


class SesionResumen(BaseModel):
    """Resumen liviano para el panel de sesiones del frontend."""

    id: str
    nomenclatura: str | None = None
    modalidad: Modalidad | None = None
    tipo_contratacion: TipoContratacion | None = None
    estado: EstadoSesion
    docs_count: int = 0
    updated_hint: str | None = None
    fecha_inicio: str | None = None
    docs_revisados: int = 0
    docs_totales: int = 0
    progreso_pct: int = 0
    riesgo: NivelRiesgo = NivelRiesgo.SIN_HALLAZGOS


class SesionesListaResponse(BaseModel):
    """Listado paginado de sesiones (GET /api/sesiones/)."""

    items: list[SesionResumen]
    total: int
    page: int
    page_size: int
    pages: int


class CrearSesionRequest(BaseModel):
    """Alta de sesión; campos opcionales activan la sesión de inmediato."""

    nomenclatura: str | None = None
    modalidad: Modalidad | None = None
    tipo_contratacion: TipoContratacion | None = None


class CrearSesionResponse(BaseModel):
    sesion: SesionCompliance
    mensaje: str


class MensajeRequest(BaseModel):
    mensaje: str


class MensajeResponse(BaseModel):
    sesion: SesionCompliance
    respuesta: str


class AnalisisDocumentoResponse(BaseModel):
    documento: DocumentoAnalizado
    mensaje: str
    slots_pendientes: list[TipoDocumento] = Field(default_factory=list)
    metricas: MetricasAnalisis | None = None
    json_analista: dict | None = None


class SandboxAnalizarResponse(BaseModel):
    """Respuesta consolidada del sandbox de prueba."""

    sesion: SesionCompliance
    documento: DocumentoAnalizado
    informe_markdown: str
    json_analista: dict
    metricas: MetricasAnalisis
    mensaje: str
    items_evaluados: int = 0
    items_excluidos_otro_tipo: int = 0


class RespuestasDocumentoRequest(BaseModel):
    respuestas: dict[str, str] = Field(
        ...,
        description="Mapa pregunta_id -> respuesta del usuario",
    )


class InformeMarkdownResponse(BaseModel):
    informe_markdown: str


class JuridicoRequest(BaseModel):
    mensaje: str = (
        "Emite una revisión jurídica del expediente con los documentos "
        "y respuestas disponibles hasta ahora."
    )
    documento_ids: list[str] | None = None
    forzar_final: bool = False


class JuridicoResponse(BaseModel):
    sesion: SesionCompliance
    dictamen: DictamenJuridico
    respuesta: str
