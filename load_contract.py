import pdfplumber
from docx import Document

def load_contract_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    if path.lower().endswith((".doc", ".docx")):
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)

    # TXT fallback (encoding-safe)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()
