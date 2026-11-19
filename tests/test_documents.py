"""Document translation with the fake engine: structure and formatting preserved."""

from __future__ import annotations

import pytest

from afrilingua.core.errors import InvalidInputError
from afrilingua.documents import output_path, translate_document


def test_txt_keeps_paragraph_structure(fake_engine, tmp_path):
    src = tmp_path / "speech.txt"
    src.write_text(
        "Good morning. Welcome to the meeting.\n\n\nSecond paragraph\nwraps here.\n\n  2024  \n",
        encoding="utf-8",
    )
    progress: list[float] = []
    out = translate_document(
        fake_engine, src, tmp_path / "out", "yoruba", "english", on_progress=progress.append
    )
    assert out == tmp_path / "out" / "speech.yoruba.txt"
    text = out.read_text(encoding="utf-8")
    assert text == (
        "[yoruba] Good morning. [yoruba] Welcome to the meeting.\n\n\n"
        "[yoruba] Second paragraph wraps here.\n\n  2024  \n"
    )
    assert progress[0] == 0.0 and progress[-1] == 1.0
    assert progress == sorted(progress)


def test_txt_batches_of_32(fake_engine, tmp_path):
    src = tmp_path / "long.txt"
    src.write_text(" ".join(f"Sentence number {i}." for i in range(70)), encoding="utf-8")
    translate_document(fake_engine, src, tmp_path, "hausa", "english")
    mt = fake_engine.manager.get("mt")
    assert [len(c[0]) for c in mt.calls] == [32, 32, 6]
    assert all(c[1] == "english" and c[2] == "hausa" for c in mt.calls)


def test_txt_autodetects_source(fake_engine, tmp_path):
    from afrilingua.backends.lid import HeuristicLanguageIdentifier

    fake_engine.detector = HeuristicLanguageIdentifier(use_lingua=False)
    src = tmp_path / "ig.txt"
    src.write_text("Kedu ka ị mere? Daalụ nke ukwuu.", encoding="utf-8")
    translate_document(fake_engine, src, tmp_path, "english")
    mt = fake_engine.manager.get("mt")
    assert mt.calls[-1][1] == "igbo"


def _make_docx(path):
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    doc.add_heading("Annual Report", level=1)
    p = doc.add_paragraph()
    r1 = p.add_run("Hello ")
    r1.bold = True
    r1.font.size = Pt(14)
    p.add_run("world. ")
    p.add_run("How are you?").italic = True
    doc.add_paragraph("", style="Normal")
    doc.add_paragraph("Second item", style="List Bullet")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Name"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "Total"
    table.cell(1, 1).text = "42"
    nested = table.cell(1, 0).add_table(rows=1, cols=1)
    nested.cell(0, 0).text = "Nested cell"
    section = doc.sections[0]
    section.header.paragraphs[0].text = "Confidential"
    section.footer.paragraphs[0].text = "Page footer"
    doc.save(str(path))


def test_docx_preserves_structure_and_styles(fake_engine, tmp_path):
    from docx import Document

    src = tmp_path / "report.docx"
    _make_docx(src)
    out = translate_document(fake_engine, src, tmp_path / "out", "igbo", "english")
    assert out.name == "report.igbo.docx"
    doc = Document(str(out))
    paras = doc.paragraphs
    assert paras[0].text == "[igbo] Annual Report" and paras[0].style.name == "Heading 1"
    # translation lands in the first run, keeping its formatting; other runs emptied
    assert paras[1].text == "[igbo] Hello world. [igbo] How are you?"
    assert paras[1].runs[0].bold is True and paras[1].runs[0].font.size.pt == 14
    assert all(r.text == "" for r in paras[1].runs[1:])
    assert paras[2].text == ""
    assert paras[3].text == "[igbo] Second item" and paras[3].style.name == "List Bullet"
    t = doc.tables[0]
    assert t.cell(0, 0).text == "[igbo] Name" and t.cell(0, 1).text == "[igbo] Value"
    assert t.cell(1, 1).text == "42"  # numbers are not sent to MT
    assert t.cell(1, 0).tables[0].cell(0, 0).text == "[igbo] Nested cell"
    sec = doc.sections[0]
    assert sec.header.paragraphs[0].text == "[igbo] Confidential"
    assert sec.footer.paragraphs[0].text == "[igbo] Page footer"


def test_docx_hyperlink_text_not_duplicated(fake_engine, tmp_path):
    from docx import Document
    from docx.oxml import OxmlElement

    doc = Document()
    p = doc.add_paragraph("Visit ")
    link = OxmlElement("w:hyperlink")
    r = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = "our site"
    r.append(t)
    link.append(r)
    p._p.append(link)
    p.add_run(" today.")
    src = tmp_path / "link.docx"
    doc.save(str(src))
    assert Document(str(src)).paragraphs[0].text == "Visit our site today."
    out = translate_document(fake_engine, src, tmp_path, "hausa", "english")
    assert Document(str(out)).paragraphs[0].text == "[hausa] Visit our site today."


def test_pdf_to_docx_with_page_breaks(fake_engine, tmp_path):
    fitz = pytest.importorskip("fitz")
    from docx import Document

    src = tmp_path / "slides.pdf"
    pdf = fitz.open()
    for text in ("First page text.", "Second page text."):
        page = pdf.new_page()
        page.insert_text((72, 72), text)
    pdf.save(str(src))
    pdf.close()
    out = translate_document(fake_engine, src, tmp_path, "yoruba", "english")
    assert out.suffix == ".docx" and out == output_path(src, tmp_path, "yoruba")
    doc = Document(str(out))
    texts = [p.text for p in doc.paragraphs if p.text]
    assert texts == ["[yoruba] First page text.", "[yoruba] Second page text."]
    xml = doc.element.body.xml
    assert xml.count('w:type="page"') == 1


def test_rejects_unsupported_and_missing(fake_engine, tmp_path):
    bad = tmp_path / "x.rtf"
    bad.write_text("hi")
    with pytest.raises(InvalidInputError):
        translate_document(fake_engine, bad, tmp_path, "yoruba")
    with pytest.raises(InvalidInputError):
        translate_document(fake_engine, tmp_path / "missing.txt", tmp_path, "yoruba")
    empty = tmp_path / "empty.txt"
    empty.write_text("\n\n")
    with pytest.raises(InvalidInputError):
        translate_document(fake_engine, empty, tmp_path, "yoruba")


def test_sentence_splitter():
    from afrilingua.backends.text_frontend import split_sentences

    assert split_sentences("Dr. Ade arrived. He said hi! Did he? Yes.") == [
        "Dr. Ade arrived.",
        "He said hi!",
        "Did he?",
        "Yes.",
    ]
    assert split_sentences("e.g. this stays. Next one") == ["e.g. this stays.", "Next one"]
    long = "word, " * 100
    parts = split_sentences(long, max_chars=120)
    assert all(len(p) <= 120 for p in parts) and len(parts) > 1
