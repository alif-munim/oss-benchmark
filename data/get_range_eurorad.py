# eurorad_range_to_csv.py
# Scrape Eurorad regular cases over an ID range and save each case to its own CSV.
# Captures the case "Section" (e.g., "Neuroradiology") robustly (ignores breadcrumbs like "Advanced Search").
#
# Usage:
#   pip install cloudscraper beautifulsoup4
#   python eurorad_range_to_csv.py --start 18806 --end 19164 --outdir eurorad_csvs
# Defaults: start=18806 end=19164 outdir=.

import re
import sys
import csv
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Iterable

import cloudscraper
from bs4 import BeautifulSoup, Tag, NavigableString

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

# Labels visible in the small meta block near the top of the page.
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

# Breadcrumb / nav strings we must never accept as the Section value.
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

# (Optional) whitelist of common Eurorad section names for sanity checks.
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
}

PRINT_CANDIDATES = ["?print=1", "/print", "?format=print"]
IGNORE_TAGS = {"figure", "figcaption", "aside", "footer", "table", "nav"}  # for long-text parsing
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

# ---------------- fetch ----------------

def make_scraper():
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
    return s

def is_bot_gate(html: str) -> bool:
    low = html.lower()
    return ("please wait while your request is being verified" in low
            or "<title>eurorad.org</title>" in low)

def fetch_html(url: str) -> Optional[str]:
    s = make_scraper()
    for attempt in range(3):
        try:
            r = s.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            html = r.text
            if not is_bot_gate(html):
                return html
            time.sleep(1.0 * (attempt + 1))
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return None

def try_fetch_variants(url: str) -> Optional[str]:
    html = fetch_html(url)
    if html and not is_bot_gate(html):
        return html
    for suf in PRINT_CANDIDATES:
        alt = url.rstrip("/") + suf
        html = fetch_html(alt)
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

def _valid_section(value: str) -> bool:
    if not value:
        return False
    low = value.strip().lower()
    if low in DISALLOWED_SECTION_VALUES:
        return False
    # Heuristic: Sections are short phrases, usually < 40 chars.
    if len(value) > 60:
        return False
    return True

def _find_dt_dd_value_anywhere(soup: BeautifulSoup, label_re: re.Pattern) -> Optional[str]:
    # <dl><dt>Section</dt><dd>Neuroradiology</dd>
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
    # Find a node with own-text 'Section', then walk forward for the first acceptable value.
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
                low = val.lower()
                if low in META_LABELS:
                    return None  # this label occurrence didn't have a value
                if _valid_section(val):
                    return val
            elif isinstance(nxt, Tag):
                t = _text(nxt.get_text(" ", strip=True))
                if not t:
                    continue
                low = t.lower()
                if low in META_LABELS:
                    return None
                if _valid_section(t):
                    return t
    return None

def _regex_from_flat_text(flat: str) -> Optional[str]:
    # Match "Section" followed by ":" or newline, then grab the next line.
    # Use MULTILINE to anchor at line starts and avoid earlier breadcrumbs.
    m = re.search(r"(?im)^\s*Section\s*(?::|-)?\s*(.+)$", flat)
    if m:
        cand = _text(m.group(1))
        if _valid_section(cand):
            return cand
    # If "Section" appears alone on a line, take the next non-empty line.
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
    """
    Order:
      1) <meta property="article:section"> if present and valid
      2) <dl>/<dt>Section</dt><dd>Value</dd> anywhere (excluding nav/breadcrumb)
      3) From the main content root: scan forward after a 'Section' label
      4) Regex over the FULL PAGE flat text (final fallback)
      5) If candidate not in LIKELY_SECTIONS but still passes validity checks, keep it (site adds new sections occasionally)
    """
    meta = soup.select_one('meta[property="article:section"], meta[name="article:section"]')
    if meta and meta.get("content"):
        val = _text(meta["content"])
        if _valid_section(val):
            return val

    label_re = re.compile(r"^\s*Section\s*$", re.I)

    # dt/dd anywhere (not only inside the root)
    val = _find_dt_dd_value_anywhere(soup, label_re)
    if _valid_section(val or ""):
        return val

    # scan from main/article root
    root = locate_content_root(soup)
    val = _find_following_text_from_label(root, label_re)
    if _valid_section(val or ""):
        return val

    # regex over FULL page text (not just root)
    flat_full = soup.get_text("\n", strip=True)
    val = _regex_from_flat_text(flat_full)
    if _valid_section(val or ""):
        return val

    # If nothing matched, try a soft guess: if any known section name appears early in the page, use it.
    head = "\n".join(flat_full.split("\n")[:120]).lower()
    for sec in LIKELY_SECTIONS:
        if re.search(rf"(^|\b){re.escape(sec)}(\b|$)", head):
            return sec.title()

    return None

# ---------------- parse ----------------

def parse_case(html: str, url: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    data: Dict[str, str] = {"URL": url}

    # Title
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

    # Published on (scan whole page so it works in print views too)
    body_txt = soup.get_text("\n", strip=True)
    m = re.search(r"Published on\s+(\d{2}\.\d{2}\.\d{4})", body_txt, flags=re.I)
    if m:
        data["Published on"] = m.group(1)

    # DOI
    m_eur = re.search(r"(10\.35100/eurorad/[^\s<>\)]+)", body_txt, flags=re.I)
    m_any = re.search(r"\b10\.\d{2,9}/[^\s<>\)]+", body_txt)
    if m_eur:
        data["DOI"] = m_eur.group(1)
    elif m_any:
        data["DOI"] = m_any.group(0)

    # Section
    sec = extract_section(soup)
    if sec:
        data["Section"] = sec

    root = locate_content_root(soup)

    # Content sections
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

def scrape_case(url: str) -> Optional[Dict[str, str]]:
    html = try_fetch_variants(url)
    if not html:
        return None
    return parse_case(html, url)

# ---------------- CSV ----------------

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

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=18806)
    ap.add_argument("--end", type=int, default=19164)
    ap.add_argument("--outdir", type=str, default=".")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between requests")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    start_id, end_id = int(args.start), int(args.end)
    if end_id < start_id:
        print("Error: --end must be >= --start")
        sys.exit(2)

    total_ok, total_fail = 0, 0
    for cid in range(start_id, end_id + 1):
        url = f"https://www.eurorad.org/case/{cid}"
        try:
            data = scrape_case(url)
            if not data:
                print(f"[{cid}] not available or blocked")
                total_fail += 1
            else:
                out = outdir / f"eurorad_{cid}.csv"
                save_to_csv_vertical(data, out)
                captured = [k for k in ("CLINICAL HISTORY","IMAGING FINDINGS","DISCUSSION","FINAL DIAGNOSIS") if k in data]
                sec = data.get("Section", "unknown")
                print(f"[{cid}] saved -> {out}  section: {sec}  sections: {', '.join(captured) or 'none'}")
                total_ok += 1
        except Exception as e:
            print(f"[{cid}] ERROR: {e.__class__.__name__}: {e}")
            total_fail += 1
        time.sleep(args.sleep)

    print(f"Done. OK={total_ok} Fail={total_fail} Range={start_id}-{end_id}")

if __name__ == "__main__":
    main()
