from __future__ import annotations


def build_source_url(paper: dict) -> str:
    doi = paper.get("doi")
    if doi and str(doi).startswith("10."):
        return f"https://doi.org/{doi}"

    source_url = paper.get("source_url")
    if source_url:
        return source_url

    arxiv_id = paper.get("arxiv_id")
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"

    return ""


def build_pdf_url(paper: dict) -> str:
    pdf_url = paper.get("pdf_url")
    if pdf_url:
        return pdf_url

    arxiv_id = paper.get("arxiv_id")
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    return ""
