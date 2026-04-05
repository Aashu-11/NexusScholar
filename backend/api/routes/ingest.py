"""
/api/ingest — PDF upload and ingestion endpoint.
Runs the full ingestion pipeline: parse → normalize → chunk → index.
"""

from __future__ import annotations
import uuid
import shutil
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from backend.config import settings
from backend.ingestion.service import ingest_pdf_file

logger = logging.getLogger(__name__)
router = APIRouter()


class IngestResponse(BaseModel):
    job_id: str
    paper_id: str
    title: str
    chunks_created: int
    status: str = "completed"


@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(file: UploadFile = File(...)):
    """Upload and ingest a PDF paper into the corpus."""
    from backend.main import get_store, get_bm25, get_dense

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    store = get_store()

    # Save uploaded file
    pdf_dir = Path(settings.PDF_STORE_PATH)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / f"{uuid.uuid4().hex[:8]}_{file.filename}"

    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        bm25 = get_bm25()
        dense = get_dense()
        result = await ingest_pdf_file(str(pdf_path), store, bm25, dense)

        return IngestResponse(
            job_id=uuid.uuid4().hex[:12],
            paper_id=result["paper_id"],
            title=result["title"],
            chunks_created=result["chunks_created"],
        )

    except Exception as e:
        logger.exception(f"Ingestion failed for {file.filename}")
        raise HTTPException(500, f"Ingestion failed: {str(e)}")
