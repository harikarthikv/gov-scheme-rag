"""Load text from local scheme PDFs into LangChain documents."""

import logging
import re
from pathlib import Path

import fitz
from langchain_core.documents import Document

from src.config import DATA_DIR

logger = logging.getLogger(__name__)


class SchemeDataLoader:
    """Create one document per non-empty PDF page."""

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.data_dir = data_dir

    def load(self) -> list[Document]:
        """Read every PDF below the configured directory."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"PDF directory does not exist: {self.data_dir}")

        documents: list[Document] = []
        for pdf_path in sorted(self.data_dir.rglob("*.pdf")):
            try:
                documents.extend(self._load_pdf(pdf_path))
            except Exception as error:
                logger.warning("Skipping %s: %s", pdf_path.name, error)
        logger.info("Loaded %s non-empty PDF pages", len(documents))
        return documents

    @staticmethod
    def _scheme_name(filename: str) -> str:
        return " ".join(word.capitalize() for word in re.sub(r"[-_]+", " ", Path(filename).stem).split())

    def _load_pdf(self, pdf_path: Path) -> list[Document]:
        documents = []
        with fitz.open(pdf_path) as pdf:
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
                            },
                        )
                    )
        return documents


def load_all_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Convenience wrapper used by index-building entry points."""
    return SchemeDataLoader(data_dir).load()
