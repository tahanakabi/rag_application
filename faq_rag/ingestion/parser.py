"""Parse the FAQ markdown into raw (section, question, answer) records.

Document structure:
    ## Section / category heading
    ### Question text
    answer paragraph(s) ...
    ### Next question
    ...
The answer of a question is everything between its ``###`` heading and the next
``###`` or ``##`` heading. Base64 image markup is kept in the raw answer here;
OCR extraction and stripping happen in the chunker.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

H2_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$")
H3_RE = re.compile(r"^###\s+(?P<title>.+?)\s*$")


@dataclass
class QARecord:
    section: str
    question: str
    answer: str


def parse_markdown(path: Path) -> list[QARecord]:
    """Split the markdown file into a list of Q/A records with section context."""
    lines = path.read_text(encoding="utf-8").splitlines()

    records: list[QARecord] = []
    current_section = "General"
    current_question: str | None = None
    answer_buf: list[str] = []

    def flush() -> None:
        """A recursive function to flush all records."""
        nonlocal answer_buf, current_question
        if current_question is not None:
            answer = "\n".join(answer_buf).strip()
            if answer:  # skip questions with empty answers
                records.append(
                    QARecord(
                        section=current_section,
                        question=current_question.strip(),
                        answer=answer,
                    )
                )
        answer_buf = []

    for line in lines:
        if line=="":
            continue
        h3 = H3_RE.match(line)
        # A line can match H3 (### ) — check H3 before H2 since ### also starts with ##.
        if h3:
            flush()
            current_question = h3.group("title")
            continue
        h2 = H2_RE.match(line)
        if h2 and not line.startswith("###"):
            flush()
            current_section = h2.group("title").strip()
            current_question = None
            continue
        if current_question is not None:
            answer_buf.append(line)

    flush()
    return records

if __name__ == "__main__":
    records = parse_markdown(Path("../../taxonomy_faqs_cleaned.md"))
