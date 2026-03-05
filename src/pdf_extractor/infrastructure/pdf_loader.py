"""PyMuPDF-based document loader."""

from __future__ import annotations

import hashlib
from pathlib import Path

import fitz  # PyMuPDF

from pdf_extractor.domain.entities import Document
from pdf_extractor.domain.interfaces import IDocumentLoader


class PyMuPDFDocumentLoader(IDocumentLoader):
    """Loads a PDF file and returns one Document per page."""

    def load(self, file_path: Path) -> list[Document]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        mtime = str(file_path.stat().st_mtime)
        base_id = hashlib.sha256(f"{file_path}{mtime}".encode()).hexdigest()[:16]

        documents: list[Document] = []
        with fitz.open(str(file_path)) as pdf:
            for page_num, page in enumerate(pdf):
                text = page.get_text().strip()
                if not text:
                    continue
                doc_id = f"{base_id}-p{page_num}"
                documents.append(
                    Document(
                        doc_id=doc_id,
                        file_path=file_path,
                        text=text,
                        page_number=page_num,
                        metadata={"source": str(file_path), "page": str(page_num)},
                    )
                )

        return documents
