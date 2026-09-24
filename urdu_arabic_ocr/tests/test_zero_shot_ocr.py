"""CPU tests for zero_shot_ocr (no GPU or model download needed).

Run from urdu_arabic_ocr/:  python -m pytest -q tests
"""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import zero_shot_ocr as z  # noqa: E402

pymupdf = z.pymupdf
CH = 30                                     # glyph height of the synthetic "text" (px at 300 DPI)


# ---------------------------------------------------------------------------- synthetic pages
def text_line(img, x0, x1, y, rng):
    """A line of word blobs (with dots) between x0 and x1, baseline at y."""
    x = x1
    while x > x0 + 40:
        w = int(rng.integers(40, 140))
        a = max(x0, x - w)
        cv2.rectangle(img, (a, y - CH), (x, y), 0, -1)
        cv2.circle(img, ((a + x) // 2, y - CH - 12), 5, 0, -1)
        x = a - int(rng.integers(18, 30))


def page(kind, W=2480, H=3508, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 255, np.uint8)
    L, R = 200, 2280
    pitch = int(2.1 * CH)
    if kind == "one":
        for y in range(400, 3200, pitch):
            text_line(img, L, R, y, rng)
    elif kind == "two":
        text_line(img, 900, 1580, 350, rng)                     # centred title across the gutter
        mid, gut = (L + R) // 2, 50
        for y in range(500, 3200, pitch):
            text_line(img, mid + gut, R, y, rng)
            text_line(img, L, mid - gut, y + 7, rng)             # columns need not share baselines
    elif kind == "poetry":
        mid, gap = (L + R) // 2, 220
        for i, y in enumerate(range(500, 3000, pitch)):
            if i % 3 == 2:
                continue                                         # blank line between couplets
            text_line(img, mid + gap, R, y, rng)
            text_line(img, L, mid - gap, y, rng)
    elif kind == "rule":
        mid = (L + R) // 2
        cv2.line(img, (mid, 380), (mid, 3250), 0, 3)
        for y in range(400, 3200, pitch):
            text_line(img, mid + 60, R, y, rng)
            text_line(img, L, mid - 60, y, rng)
    return img


def rotate(img, deg):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)


def kinds(lay):
    return [r.kind for r in lay.regions]


# ---------------------------------------------------------------------------- layout
def test_single_column_page_is_one_region_or_tall_pieces():
    lay = z.analyze_page(page("one"), z.OCRConfig())
    assert lay.gutter is None
    assert set(kinds(lay)) == {"full"}
    # a full-width A4 column is cut into pieces that keep >= min_scale of the resolution
    cfg = z.OCRConfig()
    for r in lay.regions:
        x0, y0, x1, y1 = r.bbox
        assert (x1 - x0) * (y1 - y0) * cfg.min_scale ** 2 <= cfg.max_pixels * 1.05
    ys = [r.bbox for r in lay.regions]
    assert all(a[3] <= b[1] + 1 for a, b in zip(ys, ys[1:]))      # top to bottom, no overlap


def test_two_columns_with_title_reads_title_then_right_then_left():
    lay = z.analyze_page(page("two"), z.OCRConfig())
    assert lay.gutter is not None, lay.notes
    k = kinds(lay)
    assert k[0] == "full"                                          # the centred title
    body = k[1:]
    assert body[:len(body) // 2] == ["right"] * (len(body) // 2)
    assert body[len(body) // 2:] == ["left"] * (len(body) - len(body) // 2)
    g = sum(lay.gutter) / 2
    assert abs(g - 1240) < 40


@pytest.mark.parametrize("deg", [-1.5, 1.0])
def test_gutter_found_on_skewed_scans(deg):
    lay = z.analyze_page(rotate(page("two"), deg), z.OCRConfig())
    assert abs(abs(lay.angle) - abs(deg)) < 0.4
    assert lay.gutter is not None, lay.notes
    assert "right" in kinds(lay) and "left" in kinds(lay)


def test_gutter_moves_with_scan_drift():
    img = page("two")
    shifted = np.full_like(img, 255)
    shifted[:, 90:] = img[:, :-90]                                 # page scanned 90 px to the right
    a, b = z.analyze_page(img, z.OCRConfig()), z.analyze_page(shifted, z.OCRConfig())
    assert abs((sum(b.gutter) - sum(a.gutter)) / 2 - 90) < 15


def test_poetry_is_not_split_in_auto_mode():
    lay = z.analyze_page(page("poetry"), z.OCRConfig())
    assert lay.gutter is None
    assert any("poetry" in n for n in lay.notes)
    assert set(kinds(lay)) == {"full"}


def test_forced_two_columns_splits_poetry_and_forced_one_never_splits():
    assert z.analyze_page(page("poetry"), z.OCRConfig(columns="2")).gutter is not None
    assert z.analyze_page(page("two"), z.OCRConfig(columns="1")).gutter is None


def test_printed_column_rule_is_used_as_gutter():
    lay = z.analyze_page(page("rule"), z.OCRConfig())
    assert lay.gutter is not None and any("rule" in n for n in lay.notes)


def test_blank_page_has_no_regions():
    lay = z.analyze_page(np.full((3508, 2480), 255, np.uint8), z.OCRConfig())
    assert lay.regions == []


def test_crop_never_exceeds_max_pixels():
    cfg = z.OCRConfig(max_pixels=400 * 28 * 28)
    lay = z.analyze_page(page("one"), cfg)
    for r in lay.regions:
        im = z.crop_image(lay, r, cfg.max_pixels)
        assert im.size[0] * im.size[1] <= cfg.max_pixels


# ---------------------------------------------------------------------------- normalisation
def test_urdu_line_gets_urdu_letters():
    # Arabic kaf / yeh / heh typed into an Urdu sentence
    src = "يه ايك كتاب هے جو ميں نے پڑھی."
    assert z.normalize_text(src) == "یہ ایک کتاب ہے جو میں نے پڑھی۔"


def test_vocalised_arabic_is_untouched():
    for src in ("ب\u0650س\u0652م\u0650 الل\u064E\u0651ه\u0650 الر\u064E\u0651ح\u0652م\u064E\u0670ن\u0650 الر\u064E\u0651ح\u0650يم\u0650",
                "ق\u064Eال\u064E ر\u064Eس\u064Fول\u064F الل\u064E\u0651ه\u0650 ص\u064Eل\u064E\u0651ى الل\u064E\u0651ه\u064F ع\u064Eل\u064Eي\u0652ه\u0650 و\u064Eس\u064Eل\u064E\u0651م\u064E",
                "ذ\u064E\u0670ل\u0650ك\u064E ال\u0652ك\u0650ت\u064Eاب\u064F ل\u064Eا ر\u064Eي\u0652ب\u064E \u06DB ف\u0650يه\u0650 \u06DB ه\u064Fد\u064Bى ل\u0650\u0651ل\u0652م\u064Fت\u064E\u0651ق\u0650ين\u064E"):
        assert z.line_script(src) == "arabic"
        assert z.normalize_text(src) == src


def test_unvocalised_arabic_is_untouched():
    src = "قال الشيخ في هذا الكتاب إن العلم نور"
    assert z.line_script(src) == "arabic"
    assert z.normalize_text(src) == src


def test_vocalised_arabic_phrase_inside_urdu_line_is_protected():
    src = "آپ نے فرمایا: إ\u0650ن\u064E\u0651م\u064Eا ال\u0652أ\u064Eع\u0652م\u064Eال\u064F ب\u0650الن\u0650\u0651ي\u064E\u0651ات\u0650 اور يه حديث مشهور هے"
    out = z.normalize_text(src)
    assert "إ\u0650ن\u064E\u0651م\u064Eا ال\u0652أ\u064Eع\u0652م\u064Eال\u064F ب\u0650الن\u0650\u0651ي\u064E\u0651ات\u0650" in out
    assert "یہ" in out and "ہے" in out


def test_unclear_lines_follow_setting():
    src = "كتب"
    assert z.normalize_text(src, "keep") == src
    assert z.normalize_text(src, "urdu") == "کتب"


def test_markdown_and_flags_survive_normalisation():
    src = "## باب اول\n\nيه[?] كتاب هے?\n\n1. پهلا"
    out = z.normalize_text(src)
    assert out.startswith("## باب اول")
    assert "یہ[?]" in out and out.count("[?]") == 1
    assert "ہے؟" in out
    assert "1. پہلا" in out


def test_urdu_digits():
    assert "۱۹۲۵" in z.normalize_text("سنه ١٩٢٥ ميں")


# ---------------------------------------------------------------------------- output checks
def test_loop_detection_and_trim():
    t = "شروع " + "اور پھر " * 12 + "ختم"
    assert z.find_loop(t)
    trimmed, looped = z.trim_loop(t)
    assert looped and trimmed.count("اور پھر") == 1 and "[?…]" in trimmed and trimmed.endswith("ختم")
    assert z.find_loop("یہ ایک عام جملہ ہے جس میں کوئی تکرار نہیں۔") is None


def test_strip_fence_and_split_hawashi():
    md = "```markdown\n# عنوان\n\nمتن\n\n### Hawashi\n\n۱. حاشیہ\n```"
    body, haw = z.split_hawashi(z.strip_wrappers(md))
    assert body == "# عنوان\n\nمتن" and haw == "۱. حاشیہ"
    assert z.split_hawashi("متن بلا حاشیہ") == ("متن بلا حاشیہ", "")
    assert z.split_hawashi("متن\n## حواشی\nنوٹ")[1] == "نوٹ"


def test_flag_uncertain_marks_low_probability_words():
    text = "یہ کتاب ہے۔"
    spans = [(0, 2), (2, 3), (3, 7), (7, 8), (8, 11)]
    lps = np.log([0.9, 0.9, 0.2, 0.9, 0.9])
    out, n, nw = z.flag_uncertain(text, spans, lps, 0.4)
    assert out == "یہ کتاب[?] ہے۔" and n == 1 and nw == 3


class FakeTok:
    """Byte-level BPE stand-in: one token per UTF-8 byte."""
    all_special_ids = []

    def convert_ids_to_tokens(self, ids):
        inv = {v: k for k, v in z._BYTE_DEC.items()}
        return [inv[i] for i in ids]


def test_token_char_spans_roundtrip():
    text = "ب\u0650س\u0652م\u0650 abc"
    ids = list(text.encode())
    spans = z.token_char_spans(FakeTok(), ids, text)
    assert spans is not None and spans[-1][1] == len(text)
    assert z.token_char_spans(FakeTok(), ids, text + "x") is None


def test_assess_flags_low_yield_latin_and_many_marks():
    cfg = z.OCRConfig()
    d = {"body": "مختصر", "hawashi": "", "est_letters": 400, "issues": [], "n_words": 1, "n_flagged": 0}
    assert any("expected" in r for r in z.assess([d], cfg))
    d = {"body": "I'm sorry, I cannot read this image clearly at all.", "hawashi": "", "est_letters": 0,
         "issues": [], "n_words": 10, "n_flagged": 0}
    assert any("Latin" in r for r in z.assess([d], cfg))
    d = {"body": "یہ ایک مکمل صفحہ ہے", "hawashi": "", "est_letters": 0, "issues": [], "n_words": 10, "n_flagged": 3}
    assert any("marked" in r for r in z.assess([d], cfg))


# ---------------------------------------------------------------------------- text layer
def test_bidi_runs():
    assert z.bidi_runs("سنہ ۱۹۲۵ء میں") == [(True, "سنہ "), (False, "۱۹۲۵"), (True, "ء میں")]
    assert [r for r, _ in z.bidi_runs("کتاب Latin text ہے")] == [True, False, True]
    assert "".join(t for _, t in z.bidi_runs("(1) Qur'an 2:255 کی تفسیر")) == "(1) Qur'an 2:255 کی تفسیر"


def test_plain_lines():
    md = "## باب\n\n**متن**[?] یہاں\n\n### Hawashi\n\n- حاشیہ\n\n---"
    assert z.plain_lines(md) == ["باب", "متن یہاں", "حاشیہ"]


@pytest.fixture(scope="module")
def font_path(tmp_path_factory):
    try:
        return z.text_layer_font(str(tmp_path_factory.mktemp("font")), log=lambda *a: None)
    except Exception as e:  # offline
        pytest.skip(f"no Arabic font available: {e}")


LINES = ["یہ ایک کتاب ہے۔ ب\u0650س\u0652م\u0650 الل\u064E\u0651ه\u0650 الر\u064E\u0651ح\u0652م\u064E\u0670ن\u0650", "دوسری سطر 123 اور Latin text ہے۔", "سنہ ۱۹۲۵ء میں"]


def test_hidden_text_layer_round_trips(font_path):
    doc = pymupdf.open()
    p = doc.new_page()
    tw = pymupdf.TextWriter(p.rect)
    z.write_rtl_lines(tw, pymupdf.Font(fontfile=font_path), LINES, (50, 50, 545, 200))
    tw.write_text(p, render_mode=3)
    doc = pymupdf.open("pdf", doc.tobytes())
    got = [ln.replace("\u200F", "") for ln in doc[0].get_text().strip().split("\n")]
    assert got == LINES
    assert doc[0].search_for("کتاب") and doc[0].search_for("Latin")


# ---------------------------------------------------------------------------- end to end (fake model)
class FakeEngine:
    """Returns canned Markdown; the first read of every crop on page 2 loops, page 3 is cut off."""

    def __init__(self):
        self.tok = FakeTok()
        self.calls = []

    def ocr(self, images, budgets, repetition_penalty=1.0, prompt=None):
        out = []
        for im in images:
            self.calls.append((im.size, repetition_penalty))
            text = "## باب اول\n\nيه ايك كتاب هے\n\n### Hawashi\n\nق\u064Eال\u064E الل\u064E\u0651ه\u064F ت\u064Eع\u064Eال\u064Eى"
            finished = True
            if self.page == 2 and repetition_penalty == 1.0:
                text = "اور پھر " * 20
            if self.page == 3 and len(self.calls) == 1:
                finished = False
            out.append({"text": text, "ids": [], "lp": np.zeros(0), "finished": finished, "mean_lp": -0.1})
        return out

    def chat_text(self, prompt, system, max_new_tokens=1024):
        return "God Almighty said"


@pytest.fixture()
def book(tmp_path):
    path = str(tmp_path / "book.pdf")
    doc = pymupdf.open()
    for kind in ("two", "one", "one"):
        img = page(kind)
        ok, buf = cv2.imencode(".png", img)
        p = doc.new_page(width=595, height=842)
        p.insert_image(p.rect, stream=buf.tobytes())
    doc.save(path)
    return path


def test_pipeline_cache_resume_and_exports(book, tmp_path, font_path, monkeypatch):
    monkeypatch.setattr(z, "text_layer_font", lambda *a, **k: font_path)
    cfg = z.OCRConfig(translate_hawashi=True)
    chap = str(tmp_path / "Chapter_01")
    eng = FakeEngine()
    doc = pymupdf.open(book)

    for pno in (1, 2, 3):
        eng.page, eng.calls = pno, []
        rec, lay = z.process_page(eng, doc, pno, cfg, log=lambda *a: None)
        z.save_record(chap, rec)
        if pno == 1:
            assert [d["kind"] for d in rec["regions"]][:1] == ["full"]
            assert {"right", "left"} <= {d["kind"] for d in rec["regions"]}
            md = z.page_markdown(rec)
            assert "یہ ایک کتاب ہے" in md                               # Urdu normalised
            assert "ق\u064Eال\u064E الل\u064E\u0651ه\u064F ت\u064Eع\u064Eال\u064Eى" in md                      # Arabic hawashi untouched
            assert md.count("### Hawashi") == 1                          # merged across crops
            for d in rec["regions"]:
                x0, y0, x1, y1 = d["bbox"]
                assert 0 <= x0 < x1 <= 595 and 0 <= y0 < y1 <= 842    # PDF points
        if pno == 2:
            assert all(d["retried"] == "loop" for d in rec["regions"])
            assert any(pen > 1.0 for _, pen in eng.calls)
            assert not rec["review"]["needed"] or all("loop" not in r for r in rec["review"]["reasons"])
        if pno == 3:
            assert rec["regions"][0]["retried"] == "length"
            assert len(eng.calls) > len(rec["regions"])                  # cut-off crop re-read in halves

    # resume: nothing left to do
    eng.page = 1
    assert z.run_pages(eng, book, [1, 2, 3], chap, cfg, log=lambda *a: None) == []
    assert os.path.exists(os.path.join(chap, "needs_review.txt"))

    # reviewer edit replaces the OCR text of page 3
    r3 = z.load_record(chap, 3)
    r3["corrected"] = "# دوسرا باب\n\nنظرثانی شدہ متن"
    z.save_record(chap, r3)

    md_path, pdf_path = z.export_chapter(book, chap, "Chapter_01", [1, 2, 3], cfg, engine=eng, log=lambda *a: None)
    md = open(md_path, encoding="utf-8").read()
    assert "[p. 1]" in md and "نظرثانی شدہ متن" in md
    assert os.path.exists(os.path.join(chap, "pages", "page_0003.md"))
    assert "God Almighty said" in open(os.path.join(chap, "Chapter_01_Hawashi_English.md"), encoding="utf-8").read()

    out = pymupdf.open(pdf_path)
    assert out.page_count == 4                                           # 3 scans + translation appendix
    txt = out[0].get_text().replace("\u200F", "")
    assert "یہ ایک کتاب ہے" in txt and "باب اول" in txt and "[?]" not in txt
    assert out[0].search_for("کتاب")
    assert "نظرثانی شدہ متن" in out[2].get_text().replace("\u200F", "")
    toc = out.get_toc()
    assert [1, "باب اول", 1] in toc and [1, "دوسرا باب", 3] in toc
    assert toc[-1][1].startswith("Appendix")
    assert out[0].get_images()                                           # original scan kept


def test_webhook_never_raises():
    assert z.notify("", "x") is False
    assert z.notify("http://127.0.0.1:9/hook", "x", log=lambda *a: None) is False
