from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document


@dataclass
class Question:
    id: int
    question: str
    options: dict[str, str]
    answer: str
    hint: str | None = None
    explanation: str | None = None
    source: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "options": self.options,
            "answer": self.answer,
            "hint": self.hint,
            "explanation": self.explanation,
            "source": self.source,
        }


class BaseParser(ABC):
    PARENT_URL = "https://www.crewcn.com/download/?Smallclassname=13"

    def __init__(self, file_path: str, exam_title: str | None = None):
        self.file_path = Path(file_path)
        self.exam_title = exam_title or self.file_path.stem
        self.doc = Document(file_path)

    @abstractmethod
    def parse(self) -> list[Question]:
        """Parse the document and return list of questions."""
        pass

    def _make_source(self) -> dict[str, str]:
        """Create source metadata for questions."""
        return {
            "parent_url": self.PARENT_URL,
            "exam_title": self.exam_title,
            "file_name": self.file_path.name,
        }

    def get_all_text(self) -> list[str]:
        """Get all paragraph text from the document."""
        return [para.text.strip() for para in self.doc.paragraphs if para.text.strip()]

    def to_json(self, questions: list[Question]) -> list[dict]:
        """Convert questions to JSON-serializable format."""
        return [q.to_dict() for q in questions]
