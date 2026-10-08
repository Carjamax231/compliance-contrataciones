"""Fuente de conocimiento: filesystem local o Google Cloud Storage."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


def knowledge_source() -> str:
    raw = (os.getenv("KNOWLEDGE_SOURCE") or "local").strip().lower()
    if raw in {"gcs", "bucket", "cloud"}:
        return "gcs"
    return "local"


def _bucket_cuestionarios() -> str:
    return (os.getenv("GCS_BUCKET_CUESTIONARIOS") or "").strip()


def _bucket_basamento() -> str:
    return (os.getenv("GCS_BUCKET_BASAMENTO") or "").strip()


@lru_cache(maxsize=1)
def _gcs_client():
    from google.cloud import storage

    return storage.Client()


@lru_cache(maxsize=256)
def _leer_gcs_cached(bucket_name: str, blob_path: str) -> str | None:
    try:
        client = _gcs_client()
        blob = client.bucket(bucket_name).blob(blob_path)
        if not blob.exists():
            return None
        return blob.download_as_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("GCS read fail gs://%s/%s: %s", bucket_name, blob_path, exc)
        return None


def leer_texto(
    *,
    local_path: Path,
    gcs_blob: str,
    kind: str,
) -> str | None:
    """Lee un MD: GCS si KNOWLEDGE_SOURCE=gcs, si no local. Fallback cruzado."""
    src = knowledge_source()
    if src == "gcs":
        bucket = (
            _bucket_cuestionarios() if kind == "cuestionario" else _bucket_basamento()
        )
        if bucket:
            text = _leer_gcs_cached(bucket, gcs_blob)
            if text is not None:
                return text
            logger.warning(
                "GCS miss gs://%s/%s — fallback local %s",
                bucket,
                gcs_blob,
                local_path,
            )
        else:
            logger.warning(
                "KNOWLEDGE_SOURCE=gcs sin bucket para %s — fallback local", kind
            )

    if local_path.is_file():
        try:
            return local_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("No se pudo leer %s: %s", local_path, exc)
            return None

    # Último intento: GCS aunque el modo sea local (útil si solo hay buckets)
    if src != "gcs":
        bucket = (
            _bucket_cuestionarios() if kind == "cuestionario" else _bucket_basamento()
        )
        if bucket:
            return _leer_gcs_cached(bucket, gcs_blob)
    return None


def listar_basamento_rel_paths(local_dir: Path) -> list[str]:
    """Rutas relativas *.md bajo basamento (GCS o local)."""
    src = knowledge_source()
    bucket = _bucket_basamento()
    if src == "gcs" and bucket:
        try:
            client = _gcs_client()
            blobs = client.list_blobs(bucket)
            out: list[str] = []
            for b in blobs:
                name = b.name or ""
                if not name.lower().endswith(".md"):
                    continue
                if Path(name).name.upper() == "README.MD":
                    continue
                out.append(name.replace("\\", "/"))
            if out:
                return sorted(out)
            logger.warning("Bucket basamento vacío: %s — fallback local", bucket)
        except Exception as exc:
            logger.warning("No se pudo listar GCS basamento: %s — fallback local", exc)

    if not local_dir.is_dir():
        return []
    return sorted(
        str(p.relative_to(local_dir)).replace("\\", "/")
        for p in local_dir.rglob("*.md")
        if p.name.upper() != "README.MD"
    )
