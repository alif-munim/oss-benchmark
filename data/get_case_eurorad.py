# get_case_eurorad.py
# Scrape a Eurorad case (regular case, not "teaching case") and save sections to CSV.
# Usage:
#   pip install cloudscraper beautifulsoup4
#   python get_case_eurorad.py https://www.eurorad.org/case/19164
#   # -> eurorad_case_19164.csv

import re
import sys
import csv
import time
from urllib.parse import urlparse
from typing import Dict, List, Optional, Iterable

import cloudscraper
from bs4 import BeautifulSoup, Tag

HEADINGS_CANON = {
    "clinical history": "CLINICAL HISTORY",
    "imaging findings": "IMAGING FINDINGS",
    "discussion": "DISCUSSION",
    "differential diagnosis list": "DIFFERENTIAL DIAGNOSIS LIST",
    "final diagnosis": "FINAL DIAGNOSIS",
    # common subheads sometimes present
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

PRINT_CANDIDATES = ["?print=1", "/print", "?format=print"]
IGNORE_TAGS = {"figure", "figcaption", "nav", "aside", "footer", "table"}
NOISE_CLASSES = (
    "gallery", "figure", "fig", "swiper", "carousel", "thumb", "thumbnails",
    "footer", "site-footer", "site-nav", "breadcrumb"
)
NOISE_LINE_PATTERNS = [
    re.compile(r"^case\s+\d+\s+close$", re.I),
    re.compile(r"^(?:a(?:\s+b(?:\s+c)?)?|a b(?: c)?)$", re.I),
    re.compile(r"^\d+\s*x\s+", re.I),  # "1 x T2-weighted ..."
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

# ---------------- parse ----------------

def parse_case(html: str, url: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    data: Dict[str, str] = {"URL": url}

    title = ""
    og = soup.select_one('meta[property="og:title"]')
    if og and og.get("content"):
        title = og["content"].strip()
    if not title:
        h1 = soup.find(["h1", "h2"], string=True)
        if h1:
            title = h1.get_text(strip=True)
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

def scrape_case(url: str) -> Dict[str, str]:
    html = try_fetch_variants(url)
    if not html:
        return {"Title": "(unavailable)", "URL": url}
    return parse_case(html, url)

# ---------------- CSV ----------------

def infer_case_id(url: str) -> Optional[str]:
    try:
        path = urlparse(url).path.strip("/").split("/")
        if len(path) >= 2 and path[0] == "case" and path[1].isdigit():
            return path[1]
    except Exception:
        pass
    return None

def save_to_csv_vertical(data: Dict[str, str], csv_path: str):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Section", "Content"])
        order = [
            "Title", "Published on", "URL", "DOI",
            "Section",  # optional top-level "Section" label if present on page
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
    if len(sys.argv) < 2:
        print("Usage: python get_case_eurorad.py https://www.eurorad.org/case/19164")
        sys.exit(1)

    for url in sys.argv[1:]:
        data = scrape_case(url)
        cid = infer_case_id(url) or "case"
        out = f"eurorad_{cid}.csv"
        save_to_csv_vertical(data, out)
        print(f"Saved -> {out}")

if __name__ == "__main__":
    main()
