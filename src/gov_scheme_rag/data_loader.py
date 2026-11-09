"""Load text from local scheme PDFs into LangChain documents."""

import logging
import re
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

from gov_scheme_rag.config import DATA_DIR
from gov_scheme_rag.identity import canonical_filename

logger = logging.getLogger(__name__)


class SchemeDataLoader:
    """Create one document per non-empty PDF page."""

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.data_dir = data_dir
        self.stats = {"files": 0, "failed_files": 0, "empty_pages": 0, "pages": 0}

    def load(self) -> list[Document]:
        """Read every PDF below the configured directory."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"PDF directory does not exist: {self.data_dir}")

        documents: list[Document] = []
        for pdf_path in sorted(self.data_dir.rglob("*.pdf")):
            self.stats["files"] += 1
            try:
                documents.extend(self._load_pdf(pdf_path))
            except Exception as error:
                self.stats["failed_files"] += 1
                logger.warning("Skipping %s: %s", pdf_path.name, error)
        logger.info("Loaded %s non-empty PDF pages", len(documents))
        return documents

    @staticmethod
    def _scheme_name(filename: str) -> str:
        return " ".join(word.capitalize() for word in re.sub(r"[-_]+", " ", Path(filename).stem).split())

    def _load_pdf(self, pdf_path: Path) -> list[Document]:
        documents = []
        with pymupdf.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text().strip()
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "filename": pdf_path.name,
                                "scheme_name": self._scheme_name(pdf_path.name),
                                "page": page_number,
                                "source": str(pdf_path),
                                "document_id": str(pdf_path.relative_to(self.data_dir).parent /
                                                   canonical_filename(pdf_path.name)).replace("\\", "/"),
                            },
                        )
                    )
                    self.stats["pages"] += 1
                else:
                    self.stats["empty_pages"] += 1
        return documents
