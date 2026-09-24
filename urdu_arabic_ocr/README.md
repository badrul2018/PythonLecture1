# Urdu + Arabic book OCR: zero-shot Qwen2.5-VL-7B

This Colab pipeline turns a scanned Urdu/Arabic book (Nastaliq and vocalised classical Arabic, including lithographs) into Markdown and a searchable PDF. It replaces the earlier Qwen2-VL-2B + two-LoRA notebook (`legacy/Urdu_Arabic_OCR_1.ipynb`) with **Qwen2.5-VL-7B-Instruct, used zero-shot**.

| File | What it is |
| --- | --- |
| `Urdu_Arabic_OCR_ZeroShot.ipynb` | The notebook to open in Colab. It is generated, so don't edit it by hand. |
| `zero_shot_ocr.py` | The whole pipeline. The notebook writes this file to `/content` and imports it. |
| `build_notebook.py` | Rebuilds the notebook from `zero_shot_ocr.py` and the Colab cells. |
| `tests/` | CPU tests: layout, normalisation, flags, text layer, and an end-to-end run with a fake model. |
| `legacy/` | The previous Qaari/Qari notebook, kept for comparison. |

## Running it

1. Upload the notebook to Colab. Then choose *Runtime ▸ Change runtime type ▸* **L4** or **A100**.
2. Put the book PDF in Drive and fill in step 1: `PDF_PATH`, `CHAP_NAME`, and the page range.
3. Run steps 2 to 4. The first model download is about 17 GB.
4. Run **step 5** on a typical page. Check the column split (the coloured boxes) and the text.
5. Run steps 6 (OCR), 7 (review the flagged pages) and 8 (export).

Output goes to `OUTPUT_FOLDER/CHAP_NAME/`:

- `CHAP_NAME.md`: the whole chapter, with `[p. N]` markers.
- `pages/page_NNNN.md`: one file per page, written as each page finishes.
- `Searchable_CHAP_NAME.pdf`: the original scans, an invisible text layer, and bookmarks.
- `needs_review.txt`: flagged pages and the reason for each.
- `CHAP_NAME_raw.md`: unedited model output, crop by crop.
- `CHAP_NAME_Hawashi_English.md`: only if `TRANSLATE_HAWASHI` is on.
- `_cache/`: one JSON file per page. It lets a run resume after a disconnect, and it keeps your review edits.

## Pipeline

1. **Render and deskew.** Each page is rendered at 300 DPI and deskewed by up to ±3° using the projection-profile method from the old notebook.
2. **Column-aware segmentation.** The pipeline builds a vertical ink profile of every page and looks for a low-ink run in the middle 30–70 % of the text box. A printed column rule counts as the gutter. The search runs per page, so scanning drift is handled. Rows where ink crosses the gutter (titles, full-width lines) become full-width zones. Reading order is: full-width zone, then right column, then left column, then the next zone.
   **Poetry guard:** if the central gap is wider than 8 % of the text width (`max_gutter_frac`), it is treated as the gap between hemistichs and the page is not split. Set `COLUMNS` to `1` or `2` to override this.
3. **Resolution control.** A crop is never shrunk below 60 % of the 300-DPI scan. A taller crop is first cut at blank rows between lines. The draft sent full pages at `max_pixels = 1280·28²` (≈1 MP), which shrinks a 300-DPI A4 page to about 40 %. At that size the dots and small Nastaliq letters blur.
4. **Zero-shot reading.** The system prompt from the plan is tightened. It asks for Markdown only, preserved harakat, Urdu letter forms for Urdu, poetry one hemistich per line, no running headers, and marginal notes under `### Hawashi`. Crops are read in batches with greedy decoding. A small logits processor records each token's log-probability. It stores one float per step, where `output_scores` would store the full vocabulary.
5. **Repairs.**
   - Output with a repetition loop is read again with `repetition_penalty=1.1`.
   - Output that stops at the token budget is read again as two halves, cut at a blank row.
6. **`[?]` marks.** A word whose least-confident token has probability below `FLAG_BELOW_CONFIDENCE` gets `[?]`. The code comes from the old notebook and works because Qwen2.5 uses the same byte-level BPE.
7. **Orthographic normalisation.** Each line is first classified as Urdu, Arabic or unclear. The test uses only the unvocalised words: Urdu-only letters (ٹ ڈ ڑ ں ھ ہ ے, پ چ ژ گ), common Urdu function words, Arabic function words, ال-words and ة. On Urdu lines:
   - letters: `ك→ک`, `ي/ى→ی`, `ه→ہ`, `٠١٢→۰۱۲`
   - composed forms: `ہ+ٔ→ۂ`
   - punctuation: `, ; ? .` become `، ؛ ؟ ۔`
   - tatweel is removed
   Words with two or more harakat, tanween or Qur'anic marks are skipped even inside Urdu lines. Arabic lines are returned unchanged.
8. **Review queue.** A page is flagged in any of these cases:
   - Yield is below 15 %. Yield = letters returned ÷ letters expected from the ink, at about 2 per connected component.
   - A loop or cut-off could not be repaired.
   - The output is mostly Latin text (usually model commentary or a refusal).
   - More than 8 % of the words are marked `[?]`.
9. **Searchable PDF.** Each scan page is placed unchanged with `show_pdf_page`. For each crop, the text is written invisibly (`render_mode=3`) inside that crop's own rectangle, so search hits land in the right column. The PDF gets bookmarks from `#`/`##`/`###` headings, or one per page when the chapter has no headings.

## Changes from the draft script, and why

| Draft | Problem | Now |
| --- | --- | --- |
| `normalize_urdu_text` maps `ك→ک`, `ي→ی` on the whole page | Rewrites Qur'anic and Arabic text. The plan says Arabic blocks are skipped, but the code never checked. | Classifies each line and protects each vocalised word. Tested on mixed lines. |
| `detect_and_split_columns` returns `[image]` | No column split at all | Projection-profile gutter per page, full-width zones, poetry guard, column rules |
| Whole page at ≈1 MP | Nastaliq dots and small letters blur at about 40 % scale | Crops stay at ≥ 60 % scale. Tall crops are cut at blank rows. |
| `insert_textbox(..., render_mode=3)` with the default font | With the default font, Arabic comes out as `?? ???? ??`. If the page's text doesn't fit the box, `insert_textbox` writes nothing and returns a negative number. Both were checked with PyMuPDF 1.28. | `TextWriter` with Amiri (cmap limited to base code points, so search matches `ک`, not presentation forms), placed right-to-left per run with LTR digits and Latin. Round-trip is tested with PyMuPDF. |
| `open(f"{OUTPUT_DIR}/page_N.md")` | The folder is never created. A crash loses the whole batch. | Per-page JSON cache on Drive, atomic writes, automatic resume, `REDO_PAGES` |
| `max_new_tokens=2048`, no check | Silent truncation and repetition loops | Budget from the ink, loop retry, cut-off retry in halves, all flagged |
| `requests.post("YOUR_WEBHOOK_URL")` | Crashes, and puts the URL in the notebook | Colab secret `OCR_WEBHOOK_URL`, never raises. Sends start, every 25 pages, finish, and errors (with the page it stopped on). Payload has both `content` (Discord) and `text` (Slack). |
| `page_text` concatenated per crop | `### Hawashi` repeated once per column | Hawashi merged into one section per page |

## Dr. Tariq's three questions: what was decided

1. **Bookmarks.** Yes. `BOOKMARKS = "headings"` builds the PDF outline from the Markdown headings. It falls back to one bookmark per page, and `pages` / `off` are also available. It adds almost no run time.
2. **English gloss of the hawashi.** Built, but off by default (`TRANSLATE_HAWASHI`). It is a text-only pass of the same model, so it needs no extra download. It adds roughly 10–30 s for each page with notes. A 7B model's translation of classical Arabic is a reading aid, not a citable translation. The appendix says so, and it is written as both Markdown and PDF pages.
3. **Open-source release.** This is your decision. The code has no book-specific data and no secrets (the webhook URL stays in Colab secrets), and Qwen2.5-VL-7B-Instruct is Apache-2.0. Before you publish any **dataset**, check the copyright status of the scanned editions. For a release, add a `LICENSE` and a small hand-checked ground-truth set (for example, 20 pages) so others can measure CER.

## Runtime and cost

The draft's "~8 s per page" is too optimistic for a 7B model writing 800–1500 tokens per dense page. Decoding is limited by memory bandwidth, so expect roughly:

| GPU | Per dense page | 100 pages | Colab compute units |
| --- | --- | --- | --- |
| L4 (24 GB, bf16) | ~25–50 s (batch 2) | ~45–80 min | ~4–7 CU |
| A100 40 GB (bf16) | ~10–20 s (batch 4) | ~20–35 min | ~5–8 CU |
| T4 (16 GB, 4-bit) | ~60–120 s | 2–3 h | lower quality, not recommended |

Add about 5 min per session to download and load the model. At the Colab Pro rate (about $0.10 per unit), a 100-page chapter costs roughly **$0.40–$0.80**. The draft table put the A100 at 1.5–5 CU/h, but it is about 12 CU/h. Rates change, so check Colab's *Resources* panel. These are estimates, not measurements. Step 6 prints the real seconds per page after the first few pages.

## Known limits

- The model has not been fine-tuned on these books. Heavy bleed-through, calligraphic titles and very small marginal print will read worse. Use the review queue.
- Column detection needs a visible gutter or a printed rule. Pages that mix poetry and prose in two columns may need `COLUMNS` set by hand.
- Positions in the text layer are approximate, at crop and line level, not word level. Search and copy work, but highlight boxes won't sit exactly on the printed words. Extracted text contains invisible RLM marks (U+200F) at the ends of runs.
- An Arabic quotation printed without harakat inside an Urdu sentence is normalised with that sentence. Set `NORMALIZE_UNCLEAR_LINES` or review such pages.

## Development

```bash
pip install numpy opencv-python-headless pillow pymupdf fonttools pytest
python -m pytest -q tests          # the text-layer tests download the Amiri font once, and are skipped offline
python build_notebook.py           # regenerate the notebook after editing zero_shot_ocr.py
```
