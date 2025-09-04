# eurorad_range_to_csv.py
# Scrape Eurorad regular cases and save each case to its own CSV.
#
# Modes:
#   1) Range (default): --start / --end
#   2) CSV IDs:         --csv / --case-id-col (only those Case IDs)
#   3) Union:           --csv ... --include-range (union of CSV IDs and range)
#
# Extras:
#   - tqdm progress bar by default (no per-case prints)
#   - --debug to log per-case details instead of tqdm
#   - --resume to skip IDs that already have an output CSV in --outdir
#     (checks completeness, not just existence)
#
# Examples:
#   pip install cloudscraper beautifulsoup4 tqdm
#   python eurorad_range_to_csv.py --start 18806 --end 19164 --outdir eurorad_csvs
#   python eurorad_range_to_csv.py --csv cases.csv --case-id-col "Case ID" --outdir eurorad_csvs
#   python eurorad_range_to_csv.py --csv cases.csv --include-range --start 18806 --end 18820 --outdir eurorad_csvs
#   python eurorad_range_to_csv.py --csv cases.csv --case-id-col "Case ID" --outdir eurorad_csvs --resume
#   python eurorad_range_to_csv.py --start 18806 --end 19164 --outdir eurorad_csvs --debug

import re
import sys
import csv
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Set, Tuple

import cloudscraper
from bs4 import BeautifulSoup, Tag, NavigableString
from tqdm import tqdm

HEADINGS_CANON = {
    "clinical history": "CLINICAL HISTORY",
    "imaging findings": "IMAGING FINDINGS",
    "discussion": "DISCUSSION",
    "differential diagnosis list": "DIFFERENTIAL DIAGNOSIS LIST",
    "final diagnosis": "FINAL DIAGNOSIS",
    "background": "BACKGROUND",
    "clinical perspective": "CLINICAL PERSPECTIVE",
    "imaging perspective": "IMAGING PERSPECTIVE",
    "outcome": "OUTCOME",
    "take home message / teaching points": "TEACHING POINTS",
    "take home message": "TEACHING POINTS",
    "teaching points": "TEACHING POINTS",
    "references": "REFERENCES",
}

STOP_HEADINGS = {
    "most active authors",
    "useful links",
    "brought to you by the european society of radiology",
    "follow us",
}

META_LABELS = {
    "section",
    "case type",
    "authors",
    "patient",
    "categories",
    "clinical history",
    "imaging findings",
    "discussion",
    "background",
    "differential diagnosis list",
    "final diagnosis",
    "outcome",
    "teaching points",
    "references",
}

DISALLOWED_SECTION_VALUES = {
    "home",
    "advanced search",
    "teaching cases",
    "quizzes",
    "faqs",
    "contact us",
    "history",
    "submit a case",
    "about us",
    "case",
    "cases",
}

LIKELY_SECTIONS = {
    "abdominal imaging",
    "breast imaging",
    "cardiovascular imaging",
    "chest imaging",
    "head and neck",
    "interventional radiology",
    "musculoskeletal",
    "neuroradiology",
    "nuclear medicine",
    "paediatric radiology",
    "urogenital imaging",
    "uroradiology & genital male imaging",
    "uroradiology and genital male imaging",
    "musculoskeletal system",
    "cardiovascular",
    "chest",
    "paediatric",
}

PRINT_CANDIDATES = ["?print=1", "/print", "?format=print"]
IGNORE_TAGS = {"figure", "figcaption", "aside", "footer", "table", "nav"}  # for main content parsing
NOISE_CLASSES = (
    "gallery", "figure", "fig", "swiper", "carousel", "thumb", "thumbnails",
    "footer", "site-footer", "site-nav", "breadcrumb"
)
NOISE_LINE_PATTERNS = [
    re.compile(r"^case\s+\d+\s+close$", re.I),
    re.compile(r"^(?:a(?:\s+b(?:\s+c)?)?|a b(?: c)?)$", re.I),
    re.compile(r"^\d+\s*x\s+", re.I),
    re.compile(r"\bOrigin:\s*Department of Radiology\b", re.I),
    re.compile(r"^useful links$", re.I),
    re.compile(r"^most active authors$", re.I),
    re.compile(r"^brought to you by the european society of radiology", re.I),
    re.compile(r"^follow us$", re.I),
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

# ---------------- fetch (reused session) ----------------

def make_scraper(timeout: int) -> cloudscraper.CloudScraper:
    s = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.eurorad.org/",
        "Connection": "close",
    })
    s.request_timeout = timeout
    return s

def is_bot_gate(html: str) -> bool:
    low = html.lower()
    return ("please wait while your request is being verified" in low
            or "<title>eurorad.org</title>" in low)

def fetch_html(sess: cloudscraper.CloudScraper, url: str, timeout: int, retries: int, backoff: float) -> Optional[str]:
    for attempt in range(retries):
        try:
            r = sess.get(url, timeout=timeout)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            html = r.text
            if not is_bot_gate(html):
                return html
            time.sleep(backoff * (attempt + 1))
        except KeyboardInterrupt:
            raise
        except Exception:
            time.sleep(backoff * (attempt + 1))
    return None

def try_fetch_variants(sess: cloudscraper.CloudScraper, url: str, timeout: int, retries: int, backoff: float) -> Optional[str]:
    html = fetch_html(sess, url, timeout, retries, backoff)
    if html and not is_bot_gate(html):
        return html
    for suf in PRINT_CANDIDATES:
        alt = url.rstrip("/") + suf
        html = fetch_html(sess, alt, timeout, retries, backoff)
        if html and not is_bot_gate(html):
            return html
    return html

# ---------------- parsing utils ----------------

def normalize_ws(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def canonicalize_heading(s: str) -> Optional[str]:
    key = re.sub(r"\s+", " ", (s or "").strip().lower())
    if key in HEADINGS_CANON:
        return HEADINGS_CANON[key]
    for k in HEADINGS_CANON:
        if key.startswith(k):
            return HEADINGS_CANON[k]
    return None

def looks_like_stop_heading(text: str) -> bool:
    key = re.sub(r"\s+", " ", (text or "").strip().lower())
    return any(key.startswith(h) for h in STOP_HEADINGS)

def has_noise_class(tag: Tag) -> bool:
    classes = tag.get("class") or []
    return any(any(noise in c.lower() for noise in NOISE_CLASSES) for c in classes)

def is_noise_line(line: str) -> bool:
    l = line.strip()
    if not l:
        return True
    return any(p.search(l) for p in NOISE_LINE_PATTERNS)

def dedupe_preserve_order(lines: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for ln in lines:
        key = ln.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(ln)
    return out

def clean_block_text(node: Tag) -> List[str]:
    if node.name in IGNORE_TAGS or has_noise_class(node):
        return []
    lines: List[str] = []
    blocks = node.find_all(["p", "div", "li"], recursive=True)
    if not blocks and node.name in ("p", "div", "li"):
        blocks = [node]
    for b in blocks:
        if b.name in IGNORE_TAGS or has_noise_class(b):
            continue
        txt = b.get_text(" ", strip=True)
        if not txt:
            continue
        for ln in re.split(r"\s{2,}|\n+", txt):
            ln = ln.strip()
            if not ln or is_noise_line(ln):
                continue
            lines.append(ln)
    return dedupe_preserve_order(lines)

def locate_content_root(soup: BeautifulSoup) -> Tag:
    for sel in ("main", "article", "section"):
        node = soup.find(sel)
        if node:
            return node
    pub = soup.find(string=re.compile(r"Published on", re.I))
    if pub and hasattr(pub, "parent") and isinstance(pub.parent, Tag):
        anc = pub.parent.find_parent(["article", "section", "div"])
        if anc:
            return anc
    return soup.body or soup

# ---------------- 'Section' extraction ----------------

def _text(x: str) -> str:
    return normalize_ws(x or "").strip()

def _in_nav_or_breadcrumb(tag: Tag) -> bool:
    for anc in tag.parents:
        if isinstance(anc, Tag):
            if anc.name == "nav":
                return True
            classes = anc.get("class") or []
            if any("breadcrumb" in c.lower() for c in classes):
                return True
    return False

def _valid_section(value: Optional[str]) -> bool:
    if not value:
        return False
    low = value.strip().lower()
    if low in DISALLOWED_SECTION_VALUES:
        return False
    if len(value) > 60:
        return False
    return True

def _find_dt_dd_value_anywhere(soup: BeautifulSoup, label_re: re.Pattern) -> Optional[str]:
    for dt in soup.find_all("dt"):
        if _in_nav_or_breadcrumb(dt):
            continue
        if label_re.fullmatch(_text(dt.get_text())):
            dd = dt.find_next_sibling("dd")
            if dd:
                val = _text(dd.get_text(" ", strip=True))
                if _valid_section(val):
                    return val
    return None

def _find_following_text_from_label(root: Tag, label_re: re.Pattern) -> Optional[str]:
    for node in root.find_all(string=label_re):
        parent = node.parent if isinstance(node, NavigableString) else root
        if not isinstance(parent, Tag) or _in_nav_or_breadcrumb(parent):
            continue
        for nxt in parent.next_elements:
            if isinstance(nxt, Tag) and _in_nav_or_breadcrumb(nxt):
                continue
            if isinstance(nxt, NavigableString):
                val = _text(str(nxt))
                if not val:
                    continue
                if val.lower() in META_LABELS:
                    return None
                if _valid_section(val):
                    return val
            elif isinstance(nxt, Tag):
                t = _text(nxt.get_text(" ", strip=True))
                if not t:
                    continue
                if t.lower() in META_LABELS:
                    return None
                if _valid_section(t):
                    return t
    return None

def _regex_from_flat_text(flat: str) -> Optional[str]:
    m = re.search(r"(?im)^\s*Section\s*(?::|-)?\s*(.+)$", flat)
    if m:
        cand = _text(m.group(1))
        if _valid_section(cand):
            return cand
    lines = [l.strip() for l in flat.split("\n")]
    for i, ln in enumerate(lines):
        if ln.lower() == "section":
            for j in range(i + 1, min(i + 8, len(lines))):
                cand = lines[j].strip()
                if not cand:
                    continue
                if cand.lower() in META_LABELS:
                    break
                if _valid_section(cand):
                    return cand
            break
    return None

def extract_section(soup: BeautifulSoup) -> Optional[str]:
    meta = soup.select_one('meta[property="article:section"], meta[name="article:section"]')
    if meta and meta.get("content"):
        val = _text(meta["content"])
        if _valid_section(val):
            return val

    label_re = re.compile(r"^\s*Section\s*$", re.I)
    val = _find_dt_dd_value_anywhere(soup, label_re)
    if _valid_section(val):
        return val

    root = locate_content_root(soup)
    val = _find_following_text_from_label(root, label_re)
    if _valid_section(val):
        return val

    flat_full = soup.get_text("\n", strip=True)
    val = _regex_from_flat_text(flat_full)
    if _valid_section(val):
        return val

    head = "\n".join(flat_full.split("\n")[:120]).lower()
    for sec in LIKELY_SECTIONS:
        if re.search(rf"(^|\b){re.escape(sec)}(\b|$)", head):
            return sec.title()

    return None

# ---------------- parse & scrape ----------------

def parse_case(html: str, url: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    data: Dict[str, str] = {"URL": url}

    title = ""
    og = soup.select_one('meta[property="og:title"]')
    if og and og.get("content"):
        title = og["content"].strip()
    if not title:
        h = soup.find(["h1", "h2"], string=True)
        if h:
            title = h.get_text(strip=True)
    if not title:
        ttag = soup.find("title")
        if ttag:
            title = ttag.get_text(strip=True)
    data["Title"] = title or ""

    body_txt = soup.get_text("\n", strip=True)
    m = re.search(r"Published on\s+(\d{2}\.\d{2}\.\d{4})", body_txt, flags=re.I)
    if m:
        data["Published on"] = m.group(1)

    m_eur = re.search(r"(10\.35100/eurorad/[^\s<>\)]+)", body_txt, flags=re.I)
    m_any = re.search(r"\b10\.\d{2,9}/[^\s<>\)]+", body_txt)
    if m_eur:
        data["DOI"] = m_eur.group(1)
    elif m_any:
        data["DOI"] = m_any.group(0)

    sec = extract_section(soup)
    if sec:
        data["Section"] = sec

    root = locate_content_root(soup)

    heading_tags = ("h1", "h2", "h3", "h4", "h5", "strong")
    sections: Dict[str, List[str]] = {}
    current_key: Optional[str] = None

    for el in root.descendants:
        if not isinstance(el, Tag):
            continue
        if el.name in {"footer", "nav"} or has_noise_class(el):
            current_key = None
            continue
        if el.name in heading_tags:
            htxt = el.get_text(" ", strip=True)
            if looks_like_stop_heading(htxt):
                current_key = None
                continue
            canon = canonicalize_heading(htxt)
            if canon:
                current_key = canon
                sections.setdefault(current_key, [])
                continue
        if current_key and el.name in {"p", "div", "ul", "ol", "li"}:
            lines = clean_block_text(el)
            if lines:
                sections[current_key].extend(lines)

    if not sections:
        txt = soup.get_text("\n", strip=True)
        labels = list(HEADINGS_CANON.keys())
        pat = r"(?is)(" + "|".join(re.escape(lbl) for lbl in labels) + r")\s*:?\s*(.+?)(?=(?:\n(?:"
        pat += "|".join(re.escape(lbl) for lbl in labels) + r")\b)|\Z)"
        for m in re.finditer(pat, txt):
            canon = canonicalize_heading(m.group(1))
            if canon:
                chunk = normalize_ws(m.group(2))
                lines = [ln for ln in re.split(r"\n+", chunk) if not is_noise_line(ln)]
                lines = dedupe_preserve_order(lines)
                if lines:
                    sections.setdefault(canon, []).extend(lines)

    for k, lines in list(sections.items()):
        lines = dedupe_preserve_order(lines)
        text = normalize_ws("\n\n".join(lines))
        if text:
            data[k] = text

    return data

def scrape_case(sess: cloudscraper.CloudScraper, url: str, timeout: int, retries: int, backoff: float) -> Optional[Dict[str, str]]:
    html = try_fetch_variants(sess, url, timeout, retries, backoff)
    if not html:
        return None
    return parse_case(html, url)

# ---------------- CSV output ----------------

def save_to_csv_vertical(data: Dict[str, str], csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Section", "Content"])
        order = [
            "Title", "Published on", "URL", "DOI",
            "Section",
            "CLINICAL HISTORY", "IMAGING FINDINGS", "DISCUSSION",
            "BACKGROUND", "CLINICAL PERSPECTIVE", "IMAGING PERSPECTIVE",
            "DIFFERENTIAL DIAGNOSIS LIST", "FINAL DIAGNOSIS",
            "OUTCOME", "TEACHING POINTS", "REFERENCES",
        ]
        emitted = set()
        for k in order:
            if k in data and k not in emitted:
                w.writerow([k, data[k]])
                emitted.add(k)
        for k in sorted(k for k in data.keys() if k not in emitted):
            w.writerow([k, data[k]])

# ---------------- resume helpers ----------------

_ID_FILE_RE = re.compile(r"^eurorad_(\d+)\.csv$", re.I)

def is_output_complete(path: Path) -> bool:
    """Minimal integrity check: has header and at least one data row (e.g., Title)."""
    try:
        if path.stat().st_size < 64:
            return False
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows or rows[0][:2] != ["Section", "Content"]:
            return False
        # Has at least one expected key row
        keys = {r[0] for r in rows[1:] if r}
        return any(k in keys for k in ("Title", "CLINICAL HISTORY", "IMAGING FINDINGS", "FINAL DIAGNOSIS"))
    except Exception:
        return False

def scan_completed_ids(outdir: Path) -> Set[int]:
    done: Set[int] = set()
    if not outdir.exists():
        return done
    for p in outdir.glob("eurorad_*.csv"):
        m = _ID_FILE_RE.match(p.name)
        if not m:
            continue
        cid = int(m.group(1))
        if is_output_complete(p):
            done.add(cid)
    return done

# ---------------- ID selection (CSV/range) ----------------

def _normalize_header(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "")).strip().lower()

def read_ids_from_csv(csv_path: Path, case_id_col: str) -> List[int]:
    requested = _normalize_header(case_id_col)
    ids: List[int] = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        norm_to_real = {_normalize_header(h): h for h in (reader.fieldnames or [])}
        key = None
        if requested in norm_to_real:
            key = norm_to_real[requested]
        else:
            def _canon(s: str) -> str:
                return re.sub(r"[^a-z0-9]+", "", _normalize_header(s))
            req_canon = _canon(case_id_col)
            for nh, real in norm_to_real.items():
                if _canon(nh) == req_canon:
                    key = real
                    break
        if not key:
            raise ValueError(f"Column '{case_id_col}' not found in CSV headers: {reader.fieldnames}")

        for row in reader:
            raw = (row.get(key) or "").strip()
            if not raw:
                continue
            m = re.search(r"\d+", raw)
            if not m:
                continue
            try:
                cid = int(m.group(0))
                ids.append(cid)
            except Exception:
                continue
    return ids

def build_id_list(
    start_id: int,
    end_id: int,
    csv_path: Optional[str],
    case_id_col: str,
    include_range: bool,
) -> List[int]:
    ids: Set[int] = set()
    if csv_path:
        csv_ids = read_ids_from_csv(Path(csv_path), case_id_col)
        if not include_range:
            return sorted(set(csv_ids))
        ids.update(csv_ids)
    if end_id < start_id:
        raise ValueError("--end must be >= --start")
    ids.update(range(start_id, end_id + 1))
    return sorted(ids)

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=18806, help="Start case ID (inclusive)")
    ap.add_argument("--end", type=int, default=19164, help="End case ID (inclusive)")
    ap.add_argument("--outdir", type=str, default=".", help="Output directory")
    ap.add_argument("--sleep", type=float, default=1.0, help="Seconds between requests")
    ap.add_argument("--csv", type=str, default=None, help="Path to CSV containing case IDs")
    ap.add_argument("--case-id-col", type=str, default="Case ID", help="Column name with case IDs")
    ap.add_argument("--include-range", action="store_true",
                    help="If set with --csv, scrape the union of CSV IDs and [--start, --end]. "
                         "By default, providing --csv scrapes ONLY those IDs.")
    ap.add_argument("--debug", action="store_true", help="Verbose per-case logging instead of tqdm")
    ap.add_argument("--resume", action="store_true",
                    help="Skip IDs that already have a completed output CSV in --outdir")
    ap.add_argument("--timeout", type=int, default=30, help="HTTP timeout (seconds)")
    ap.add_argument("--retries", type=int, default=3, help="HTTP retries per URL")
    ap.add_argument("--backoff", type=float, default=1.0, help="Retry backoff multiplier (seconds)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        all_ids = build_id_list(args.start, args.end, args.csv, args.case_id_col, args.include_range)
    except Exception as e:
        print(f"Error building ID list: {e}")
        sys.exit(2)

    # Resume filter
    skipped_existing = 0
    if args.resume:
        done_ids = scan_completed_ids(outdir)
        before = len(all_ids)
        ids = [cid for cid in all_ids if cid not in done_ids]
        skipped_existing = before - len(ids)
    else:
        ids = list(all_ids)

    if not ids:
        print("All requested IDs already processed (nothing to do).")
        return

    sess = make_scraper(args.timeout)

    total_ok, total_fail = 0, 0
    iterator = ids
    progress = None
    if not args.debug:
        progress = tqdm(total=len(ids), desc="Scraping Eurorad", unit="case")

    try:
        for cid in iterator:
            url = f"https://www.eurorad.org/case/{cid}"
            try:
                data = scrape_case(sess, url, args.timeout, args.retries, args.backoff)
                if not data:
                    if args.debug:
                        print(f"[{cid}] not available or blocked")
                    total_fail += 1
                else:
                    out = outdir / f"eurorad_{cid}.csv"
                    save_to_csv_vertical(data, out)
                    if args.debug:
                        captured = [k for k in ("CLINICAL HISTORY","IMAGING FINDINGS","DISCUSSION","FINAL DIAGNOSIS") if k in data]
                        sec = data.get("Section", "unknown")
                        print(f"[{cid}] saved -> {out}  section: {sec}  sections: {', '.join(captured) or 'none'}")
                    total_ok += 1
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if args.debug:
                    print(f"[{cid}] ERROR: {e.__class__.__name__}: {e}")
                total_fail += 1
            finally:
                if progress is not None:
                    progress.update(1)
            time.sleep(args.sleep)
    except KeyboardInterrupt:
        if progress is not None:
            progress.close()
        print("\nInterrupted by user.")
    finally:
        if progress is not None:
            progress.close()

    mode = "CSV-only" if args.csv and not args.include_range else ("CSV+Range" if args.csv else "Range")
    print(f"Done. Mode={mode} OK={total_ok} Fail={total_fail} "
          f"Requested={len(all_ids)} Skipped(existing)={skipped_existing} Processed_now={len(ids)}")

if __name__ == "__main__":
    main()
