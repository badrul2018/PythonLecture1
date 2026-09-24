"""Build Urdu_Arabic_OCR_ZeroShot.ipynb from zero_shot_ocr.py plus the Colab cells below.

    python build_notebook.py

The notebook writes the module to /content with %%writefile, so the code that runs in Colab is
exactly the code the tests in tests/ check. Edit zero_shot_ocr.py or this file, never the .ipynb.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "Urdu_Arabic_OCR_ZeroShot.ipynb")

INTRO = """# Urdu + Arabic book OCR · zero-shot Qwen2.5-VL-7B

Turns a range of pages from a scanned Urdu/Arabic PDF into **Markdown** and a **searchable PDF** (the original scans with an invisible, selectable text layer and bookmarks).

**How it works**
1. Each page is rendered at 300 DPI and straightened. A vertical projection profile finds the column gutter **on every page**, so scanning drift doesn't matter. Two-column zones are read right column first, then left. Full-width titles above or between the columns are read on their own, and poetry with a wide central gap is *not* split.
2. [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) reads each crop zero-shot (no LoRA) and writes Markdown. Marginal notes go under `### Hawashi`. Tall crops are cut at blank rows first so the vision encoder never sees small, blurred Nastaliq.
3. Words the model was unsure of get a **`[?]`** mark, taken from its own token probabilities. Repetition loops are re-read with a repetition penalty. Cut-off crops are re-read in two halves.
4. Urdu lines get Urdu letter forms (`ك→ک`, `ي→ی`, `ه→ہ`, `٠١٢→۰۱۲`, `?→؟`, `.→۔`). Vocalised Arabic words and Arabic lines are never changed.
5. Every page is saved to Drive as soon as it is done. If Colab disconnects, run the cells again and it resumes. Pages with problems go to `needs_review.txt` and to the review screen (step 7).

**Before you start:** *Runtime ▸ Change runtime type* ▸ **L4** or **A100** (the model needs about 17 GB of GPU memory). A T4 falls back to 4-bit weights, which read Nastaliq noticeably worse. Run the cells in order, and check one page in step 5 before you run a whole chapter.

**Optional alerts:** to get a message when a batch finishes or stops, add your Discord/Slack webhook URL as a Colab secret named `OCR_WEBHOOK_URL` (🔑 in the left sidebar) and allow this notebook to read it."""

SETTINGS = """#@title 1 · Settings { display-mode: "form" }
#@markdown **Book and chapter**
PDF_PATH = "/content/drive/MyDrive/OCR/book.pdf"  #@param {type:"string"}
CHAP_NAME = "Chapter_01"  #@param {type:"string"}
START_PAGE = 13  #@param {type:"integer"}
END_PAGE = 15  #@param {type:"integer"}
OUTPUT_FOLDER = "/content/drive/MyDrive/OCR"  #@param {type:"string"}
#@markdown PDF page numbers (the first page of the file is 1), not the numbers printed on the pages.

#@markdown ---
#@markdown **Reading**
COLUMNS = "auto"  #@param ["auto", "1", "2"]
#@markdown `auto` finds a gutter on each page. Use `1` for books with a lot of poetry, and `2` if step 5 shows missed gutters.
FLAG_BELOW_CONFIDENCE = 0.4  #@param {type:"slider", min:0, max:0.95, step:0.05}
#@markdown A word gets `[?]` when the model's confidence in any part of it is below this. 0 = no marks.
NORMALIZE_UNCLEAR_LINES = "keep"  #@param ["keep", "urdu"]
#@markdown Lines that are neither clearly Urdu nor clearly Arabic: `keep` leaves them as the model wrote them. `urdu` suits books with only a little Arabic.

#@markdown ---
#@markdown **Output**
PAGE_MARKERS = True  #@param {type:"boolean"}
#@markdown Adds `[p. 12]` before each page in the chapter Markdown.
BOOKMARKS = "headings"  #@param ["headings", "pages", "off"]
#@markdown `headings` = PDF bookmarks from the `#` / `##` / `###` headings (one per page if the chapter has none).
TRANSLATE_HAWASHI = False  #@param {type:"boolean"}
#@markdown Adds a machine-translated English gloss of the hawashi as an appendix. This adds about 10–30 s for each page that has notes.
USE_WEBHOOK = True  #@param {type:"boolean"}
#@markdown Sends alerts to the `OCR_WEBHOOK_URL` secret, if it is set: start, every 25 pages, finish, and errors.

#@markdown ---
#@markdown **Advanced**
REDO_PAGES = ""  #@param {type:"string"}
#@markdown Pages to OCR again even though they are saved, e.g. `12, 15-17`. Leave empty to resume normally.
MAX_IMAGE_TOKENS = 2048  #@param [1280, 2048, 3072] {type:"raw"}
#@markdown The most vision tokens per crop (× 28 × 28 pixels). Use a higher value for very small print. It is slower and needs more memory.
BATCH_SIZE = 2  #@param {type:"integer"}
QUANTIZE = "auto"  #@param ["auto", "none", "4bit"]
print("Settings saved.")"""

INSTALL = """#@title 2 · Connect Google Drive and install packages
import os, subprocess, sys
from google.colab import drive
drive.mount("/content/drive")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pymupdf", "fonttools"], check=True)
import torch, transformers
from packaging.version import Version
if Version(transformers.__version__) < Version("4.49"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.49", "accelerate"], check=True)
    print("⚠ transformers was upgraded. Now use Runtime ▸ Restart session, then run the cells again from step 1.")
if torch.cuda.is_available():
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)} ({gb:.0f} GB)")
    if gb < 20 or QUANTIZE == "4bit":
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=True)
        print("⚠ Less than 20 GB of GPU memory: the model will load in 4-bit. An L4 or A100 reads better.")
else:
    print("⚠ No GPU. Runtime ▸ Change runtime type ▸ L4 or A100.")
print(f"✓ PDF found: {PDF_PATH}" if os.path.exists(PDF_PATH) else f"⚠ No file at {PDF_PATH} – upload the PDF to Drive and fix PDF_PATH in step 1.")"""

IMPORT = """#@title 3b · Code: viewer, review screen, settings (just run)
import html, io, base64, os, sys
sys.path.insert(0, "/content")
import importlib, zero_shot_ocr
importlib.reload(zero_shot_ocr)
from zero_shot_ocr import *

FONT_CSS = ("<link href='https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu&family=Amiri&display=swap' "
            "rel='stylesheet'>")


def _img_b64(img, q=80):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode()


def _esc(t, mark):
    t, m = html.escape(t), html.escape(mark)
    return t.replace(m, f"<span style='background:#fde68a;color:#92400e;border-radius:3px;padding:0 2px'>{m}</span>")


def page_html(rec, lay=None, mark="[?]", eta=""):
    rows = []
    for i, d in enumerate(rec["regions"], 1):
        col = {"full": "#16a34a", "right": "#2563eb", "left": "#ea580c"}[d["kind"]]
        warn = "".join(f"<span style='color:#b91c1c'> ⚠ {html.escape(x)}</span>" for x in d["issues"])
        info = f"{i} · {d['kind']} · mean p {d['mean_p']}" + (f" · retried ({d['retried']})" if d["retried"] else "")
        txt = d["body"] + (("\\n\\n### Hawashi\\n" + d["hawashi"]) if d["hawashi"] else "")
        rows.append(f"<div style='margin:6px 0;padding:6px 8px;border-right:4px solid {col};background:#fafafa'>"
                    f"<div dir='ltr' style='font:11px sans-serif;color:{col}'>{html.escape(info)}{warn}</div>"
                    f"<div dir='rtl' style=\\"font-family:'Noto Nastaliq Urdu','Amiri',serif;font-size:17px;"
                    f"line-height:2.2\\">{_esc(txt, mark).replace(chr(10), '<br>')}</div></div>")
    rv = rec["review"]
    status = ("<span style='color:#b45309'>needs review: " + html.escape("; ".join(rv["reasons"][:4])) + "</span>"
              if rv["needed"] else "<span style='color:#15803d'>no issues flagged</span>")
    img = f"<img src='data:image/jpeg;base64,{_img_b64(draw_regions(lay, 700))}' style='width:100%;border:1px solid #ddd'>" if lay else ""
    notes = html.escape(" · ".join(rec.get("layout_notes", [])))
    return (f"{FONT_CSS}<div style='font:13px sans-serif;margin-bottom:6px'><b>Page {rec['page']}</b> · "
            f"{rec['seconds']}s · {len(rec['regions'])} crop(s) · {rec['n_flagged']} of {rec['n_words']} words "
            f"marked · {status} {html.escape(eta)}<br><span style='color:#666'>{notes}</span></div>"
            f"<div style='display:flex;gap:12px;align-items:flex-start'><div style='flex:0 0 40%'>{img}"
            f"<div style='font:11px sans-serif;color:#555'><span style='color:#16a34a'>■</span> full width "
            f"<span style='color:#2563eb'>■</span> right column <span style='color:#ea580c'>■</span> left column</div></div>"
            f"<div style='flex:1;max-height:900px;overflow:auto'>{''.join(rows)}</div></div>")


def show_page(rec, lay, eta=""):
    from IPython.display import HTML, clear_output, display
    clear_output(wait=True)
    display(HTML(page_html(rec, lay, cfg.flag_mark, eta)))


def review_pages(pdf_path, chap_dir, pages, only_flagged=True):
    \"\"\"Page image on the left, editable Markdown on the right. 'Save' replaces that page's OCR text.\"\"\"
    import ipywidgets as W
    from IPython.display import HTML, display
    doc = pymupdf.open(pdf_path)
    recs = {p: r for p in pages if (r := load_record(chap_dir, p)) is not None}
    queue = [p for p in pages if p in recs and (not only_flagged or recs[p]["review"]["needed"])]
    if not queue:
        display(HTML("<b>Nothing to review</b> – no processed page was flagged. Untick ONLY_FLAGGED to open every page."))
        return
    state = {"i": 0}
    css = W.HTML(FONT_CSS + "<style>.rtl-ed textarea{direction:rtl;font-family:'Noto Nastaliq Urdu','Amiri',serif;"
                 "font-size:17px;line-height:2.1}</style>")
    head, msg = W.HTML(), W.HTML()
    img = W.Image(format="jpeg", layout=W.Layout(width="100%"))
    ta = W.Textarea(layout=W.Layout(width="100%", height="760px"))
    ta.add_class("rtl-ed")
    b_prev, b_save, b_skip = W.Button(description="◀ Previous"), W.Button(description="Save & next ▶", button_style="success"), W.Button(description="Skip ▶")
    b_reset = W.Button(description="Discard my edits", button_style="warning")

    def show():
        p = queue[state["i"]]
        r = recs[p]
        img.value = doc[p - 1].get_pixmap(dpi=110).tobytes("jpeg")
        edited = " · <b style='color:#15803d'>edited</b>" if r.get("corrected") is not None else ""
        head.value = (f"<b>Page {p}</b> ({state['i'] + 1} of {len(queue)}){edited}<br>"
                      f"<span style='color:#b45309'>{html.escape('; '.join(r['review']['reasons']) or '–')}</span><br>"
                      f"<span style='color:#666'>Words marked {cfg.flag_mark} were uncertain: delete the mark once checked. "
                      f"Keep '### Hawashi' above the marginal notes.</span>")
        ta.value = page_markdown(r)
        msg.value = ""

    def save(_=None):
        p = queue[state["i"]]
        recs[p]["corrected"] = ta.value
        save_record(chap_dir, recs[p])
        write_page_markdown(chap_dir, recs[p])
        write_review_queue(chap_dir, [load_record(chap_dir, q) for q in pages])
        msg.value = f"<span style='color:#15803d'>Saved page {p}.</span>"
        if state["i"] < len(queue) - 1:
            state["i"] += 1; show()

    def move(d):
        state["i"] = max(0, min(len(queue) - 1, state["i"] + d)); show()

    def reset(_):
        p = queue[state["i"]]
        recs[p]["corrected"] = None
        save_record(chap_dir, recs[p]); write_page_markdown(chap_dir, recs[p]); show()
        msg.value = f"Page {p} is back to the OCR text."

    b_save.on_click(save); b_skip.on_click(lambda _: move(1)); b_prev.on_click(lambda _: move(-1)); b_reset.on_click(reset)
    show()
    display(W.VBox([css, head, W.HBox([W.Box([img], layout=W.Layout(width="45%")), W.Box([ta], layout=W.Layout(width="55%"))]),
                    W.HBox([b_prev, b_save, b_skip, b_reset]), msg]))


def parse_pages(s):
    out = []
    for part in str(s).replace(" ", "").split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def webhook_url():
    if not USE_WEBHOOK:
        return ""
    try:
        from google.colab import userdata
        return userdata.get("OCR_WEBHOOK_URL") or ""
    except Exception:
        return ""


def current_setup():
    c = OCRConfig(columns=str(COLUMNS), conf_flag=float(FLAG_BELOW_CONFIDENCE), normalize_unclear=NORMALIZE_UNCLEAR_LINES,
                  page_markers=bool(PAGE_MARKERS), bookmarks=BOOKMARKS, translate_hawashi=bool(TRANSLATE_HAWASHI),
                  max_pixels=int(MAX_IMAGE_TOKENS) * 28 * 28, batch_size=int(BATCH_SIZE), quantize=QUANTIZE,
                  webhook_url=webhook_url())
    if int(END_PAGE) < int(START_PAGE):
        raise ValueError("END_PAGE is before START_PAGE")
    chap_dir = os.path.join(OUTPUT_FOLDER, CHAP_NAME)
    os.makedirs(chap_dir, exist_ok=True)
    return c, chap_dir, list(range(int(START_PAGE), int(END_PAGE) + 1)), parse_pages(REDO_PAGES)


cfg, CHAP_DIR, PAGES, REDO = current_setup()
print(f"Chapter folder: {CHAP_DIR} · pages {PAGES[0]}–{PAGES[-1]} ({len(PAGES)})"
      + (" · webhook alerts on" if cfg.webhook_url else ""))"""

LOAD = """#@title 4 · Load Qwen2.5-VL-7B-Instruct (first run downloads ≈17 GB)
cfg, CHAP_DIR, PAGES, REDO = current_setup()
engine = QwenVLEngine.load(cfg)
print("✓ Ready.")"""

TEST_MD = """### Step 5: test one page before the whole chapter
The page is shown with its crops in reading order: green = full width, blue = right column, orange = left column. Check that:
* **The column split is right.** If poetry was split into two columns, set `COLUMNS` to `1`. If a two-column page was not split, set it to `2`.
* **The text reads well.** Yellow `[?]` marks show words the model was unsure of. If small print reads badly, raise `MAX_IMAGE_TOKENS`.

This test is not saved. Step 6 reads the page again with your final settings."""

TEST = """#@title 5 · Test page
TEST_PAGE = 13  #@param {type:"integer"}
cfg, CHAP_DIR, PAGES, REDO = current_setup()
rec, lay = process_page(engine, pymupdf.open(PDF_PATH), int(TEST_PAGE), cfg)
show_page(rec, lay)"""

RUN = """#@title 6 · Run OCR on the chapter (resumes automatically)
assert "engine" in globals(), "Run step 4 first."
cfg, CHAP_DIR, PAGES, REDO = current_setup()
run_pages(engine, PDF_PATH, PAGES, CHAP_DIR, cfg, redo=REDO, show=show_page)
print("Finished. Next: step 7 (review), then step 8 (export).")"""

REVIEW_MD = """### Step 7: review
Only flagged pages are shown. A page is flagged when little text came back for the ink on the page (under 15 % of the expected letters), when a repetition loop or cut-off output could not be repaired, when the model wrote mostly Latin text, or when more than 8 % of its words are marked `[?]`. The same list is in `needs_review.txt`.

The page image is on the left and the Markdown is on the right. Correct the text, delete `[?]` marks you have checked, and click **Save & next**. Your version replaces the OCR text in the export, and saved edits stay in Drive."""

REVIEW = """#@title 7 · Review flagged pages
ONLY_FLAGGED = True  #@param {type:"boolean"}
cfg, CHAP_DIR, PAGES, REDO = current_setup()
review_pages(PDF_PATH, CHAP_DIR, PAGES, only_flagged=ONLY_FLAGGED)"""

EXPORT = """#@title 8 · Export Markdown + searchable PDF
cfg, CHAP_DIR, PAGES, REDO = current_setup()
md_path, pdf_path = export_chapter(PDF_PATH, CHAP_DIR, CHAP_NAME, PAGES, cfg, engine=globals().get("engine"))
print(f"Also saved: pages/page_NNNN.md (one file per page) and {CHAP_NAME}_raw.md (unedited model output, crop by crop).")
notify(cfg.webhook_url, f"Export done: {CHAP_NAME} ({len(PAGES)} pages).")"""


def cell(kind, src, hidden=False):
    lines = src.split("\n")
    c = {"cell_type": kind, "metadata": {}, "source": [ln + "\n" for ln in lines[:-1]] + [lines[-1]]}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
        if hidden:
            c["metadata"]["cellView"] = "form"
    return c


def build():
    with open(os.path.join(HERE, "zero_shot_ocr.py"), encoding="utf-8") as f:
        module = f.read().rstrip("\n")
    cells = [
        cell("markdown", INTRO),
        cell("code", SETTINGS, hidden=True),
        cell("code", INSTALL, hidden=True),
        cell("markdown", "### Step 3: pipeline code\nJust run both cells. `3a` writes the pipeline module "
                         "(generated from `zero_shot_ocr.py` in the repository) to `/content`."),
        cell("code", "%%writefile /content/zero_shot_ocr.py\n" + module),
        cell("code", IMPORT, hidden=True),
        cell("code", LOAD, hidden=True),
        cell("markdown", TEST_MD),
        cell("code", TEST, hidden=True),
        cell("code", RUN, hidden=True),
        cell("markdown", REVIEW_MD),
        cell("code", REVIEW, hidden=True),
        cell("code", EXPORT, hidden=True),
    ]
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 0,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4", "machine_shape": "hm"},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}}}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return OUT


if __name__ == "__main__":
    print(build())
