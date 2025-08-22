import re
import csv
import time
import random
from typing import Dict, Optional, Iterable

import requests
from bs4 import BeautifulSoup, NavigableString
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from tqdm import tqdm

SECTIONS = [
    "Section",
    "CLINICAL HISTORY",
    "IMAGING FINDINGS",
    "DISCUSSION",
    "DIFFERENTIAL DIAGNOSIS LIST",
    "FINAL DIAGNOSIS",
]

BASE_URL = "https://www.eurorad.org/case/{}/teaching-case"
START_ID, END_ID = 18806, 18808
OUTPUT_CSV = "eurorad_cases_18806_19164.csv"

# Match any section header text (case-insensitive)
HDR_RE = re.compile(
    r"^\s*(Section|Clinical History|Imaging Findings|Discussion|Differential Diagnosis List|Final Diagnosis)\s*$",
    re.I,
)

def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5, connect=5, read=5, status=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16))
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    return s

def fetch_html(session: requests.Session, case_id: int) -> Optional[str]:
    try:
        r = session.get(BASE_URL.format(case_id), timeout=20)
        if r.status_code == 404: return None
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After", 30)))
            return None
        if r.status_code != 200: return None
        return r.text
    except requests.RequestException:
        return None

def norm(s: str) -> str:
    return " ".join((s or "").split()).strip()

def is_header_tag(el) -> bool:
    # Treat any element whose visible text matches HDR_RE as a header
    if not hasattr(el, "get_text"): return False
    txt = norm(el.get_text(" ", strip=True))
    return bool(HDR_RE.match(txt))

def iter_until_next_header(start_el: Iterable, stop_at_first=True) -> str:
    """
    Walk forward through document order from start_el, gathering text until the next header.
    Accepts an element; we iterate over its .next_elements (not just siblings).
    """
    parts = []
    for node in start_el:
        if hasattr(node, "get_text"):
            # If this node *itself* looks like a header, stop
            if is_header_tag(node):
                break
            name = getattr(node, "name", None)
            if name in ("p", "div", "section"):
                t = norm(node.get_text(" ", strip=True))
                if t: parts.append(t)
            elif name in ("ul", "ol"):
                for li in node.find_all("li"):
                    t = norm(li.get_text(" ", strip=True))
                    if t: parts.append(f"- {t}")
        elif isinstance(node, NavigableString):
            t = norm(str(node))
            if t: parts.append(t)
        # Stop early at first block boundary if we walked into a new major container
        if stop_at_first and getattr(node, "name", "") in ("h2", "h3", "h4"):
            # Covered by is_header_tag above; kept as a guard.
            break
    return "\n".join(parts).strip()

def find_header_element(soup: BeautifulSoup, header_text: str):
    # Find the element whose *visible* text equals header_text (case-insensitive)
    for tag in soup.find_all(True):  # any tag
        try:
            txt = norm(tag.get_text(" ", strip=True))
        except Exception:
            continue
        if txt.lower() == header_text.lower():
            return tag
    # Fallback: search by regex text nodes and return their parent
    m = soup.find(string=re.compile(rf"^\s*{re.escape(header_text)}\s*$", re.I))
    return m.parent if m else None

def parse_top_section_name(soup: BeautifulSoup) -> str:
    """
    On teaching-case pages the top block is:
        Section
        <SECTION NAME>
        Case Type
        ...
    We parse the text node immediately following the 'Section' label.
    """
    sec_label = find_header_element(soup, "Section")
    if not sec_label: return ""
    # Walk forward in document order to the next non-empty text which is NOT another header label
    for node in sec_label.next_elements:
        if is_header_tag(node):
            # if we hit another header before content, bail
            break
        if isinstance(node, NavigableString):
            t = norm(str(node))
            if t:
                # avoid generic words like "Case Type"
                if t.lower() not in {"case type"}:
                    return t
        elif hasattr(node, "get_text"):
            t = norm(node.get_text(" ", strip=True))
            if t and t.lower() not in {"case type"} and not is_header_tag(node):
                return t
    return ""

def scrape_case(session: requests.Session, case_id: int) -> Optional[Dict[str, str]]:
    html = fetch_html(session, case_id)
    if not html: return None
    soup = BeautifulSoup(html, "html.parser")

    # Build result row
    row: Dict[str, str] = {"case_id": str(case_id)}

    # 1) "Section" from the top metadata block (just the name like "Neuroradiology")
    row["Section"] = parse_top_section_name(soup)

    # 2) Content sections gathered by walking forward until next header
    for sec in ("CLINICAL HISTORY", "IMAGING FINDINGS", "DISCUSSION", "DIFFERENTIAL DIAGNOSIS LIST", "FINAL DIAGNOSIS"):
        hdr_el = find_header_element(soup, sec)
        if not hdr_el:
            row[sec] = ""
            continue
        # Start from the first element *after* the header element
        content = iter_until_next_header(hdr_el.next_elements)
        row[sec] = content

    return row

def polite_sleep():
    time.sleep(random.uniform(1.5, 3.5))

def main():
    session = make_session()
    fieldnames = ["case_id"] + SECTIONS
    total = END_ID - START_ID + 1

    html = fetch_html(session, 18806)
    print("status_ok=", bool(html), "has_clinical=", ("CLINICAL" in (html or "") or "Clinical" in (html or "")))
    print((html or "")[:1200])

if __name__ == "__main__":
    main()
