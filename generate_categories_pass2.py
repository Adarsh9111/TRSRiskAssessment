import os
import re
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    # New SDK style, mirrors extract_risks.py
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# ----------------------------
# Utilities
# ----------------------------

def _normalize(s: str) -> str:
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"\s+", " ", s.strip())
    return s


def verify_verbatim(quote: str, source: str) -> bool:
    nq = _normalize(quote)
    ns = _normalize(source)
    return nq in ns


def find_year_from_name(name: str) -> Optional[int]:
    """Extract a year from filenames like 'FY2024', 'FY24', 'ACFR23', or 4-digit years.

    Heuristics:
    - Prefer explicit 4-digit years 19xx/20xx.
    - Handle 'FY24'/'ACFR23' by mapping 00–89 -> 2000–2089, 90–99 -> 1990–1999.
    """
    # 4-digit year anywhere
    m4 = re.findall(r"((?:19|20)\d{2})", name)
    if m4:
        try:
            s = m4[-1]
            return int(s)
        except Exception:
            pass

    # FY two-digit like FY24
    m_fy2 = re.search(r"FY\s*[-_ ]?(\d{2})", name, flags=re.IGNORECASE)
    if m_fy2:
        yy = int(m_fy2.group(1))
        return 1900 + yy if yy >= 90 else 2000 + yy

    # ACFR/CAFR with two-digit like ACFR23
    m_acfr2 = re.search(r"(?:ACFR|CAFR)\s*[-_ ]?(\d{2})", name, flags=re.IGNORECASE)
    if m_acfr2:
        yy = int(m_acfr2.group(1))
        return 1900 + yy if yy >= 90 else 2000 + yy

    return None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # Skip malformed lines but keep going
                continue
    return rows


def group_snippets_by_year(input_dir: Path) -> Dict[int, List[Dict[str, Any]]]:
    """Load all JSONL files and group snippets by inferred year."""
    by_year: Dict[int, List[Dict[str, Any]]] = {}
    files = sorted([p for p in input_dir.glob("*.jsonl") if p.is_file()])
    for fp in files:
        rows = read_jsonl(fp)
        # Prefer year from row; else source_file; else filename
        file_year_hint = find_year_from_name(fp.name)
        for r in rows:
            # Try year in row
            y: Optional[int] = None
            v = r.get("year")
            if isinstance(v, int) and (1900 <= v <= 2100):
                y = v
            if y is None:
                src = r.get("source_file") or fp.name
                y = find_year_from_name(str(src))
            if y is None:
                y = file_year_hint

            if y is None:
                # Skip if year truly unknown
                continue

            quote = (r.get("quote") or "").strip()
            if not quote:
                continue
            item = {
                "id": r.get("id") or f"{fp.name}:{len(by_year.get(y, []))}",
                "text": quote,
                "section_hint": r.get("section_hint"),
                "source_file": r.get("source_file") or fp.name,
            }
            by_year.setdefault(y, []).append(item)
    return by_year


def pack_snippets_to_budget(snippets: List[Dict[str, Any]], budget_chars: int = 45000) -> Tuple[List[Dict[str, Any]], str]:
    """Pack as many snippets as possible into a JSON array string <= budget_chars.

    Returns (kept_snippets, json_string).
    """
    kept: List[Dict[str, Any]] = []
    # Favor diverse mix by interleaving short and long quotes
    ordered = sorted(snippets, key=lambda s: len(s.get("text", "")))
    # Interleave from both ends to mix lengths
    interleaved: List[Dict[str, Any]] = []
    i, j = 0, len(ordered) - 1
    while i <= j:
        if i == j:
            interleaved.append(ordered[i])
        else:
            interleaved.append(ordered[i])
            interleaved.append(ordered[j])
        i += 1
        j -= 1

    total_len = 2  # for opening/closing brackets
    parts: List[str] = ["["]
    first = True
    for s in interleaved:
        frag = {
            "id": s["id"],
            "text": s["text"],
            # Keep hint as a tie-breaker; safe for model to ignore
            "section_hint": s.get("section_hint"),
        }
        js = json.dumps(frag, ensure_ascii=False)
        add_len = len(js) + (0 if first else 1)  # comma if not first
        if total_len + add_len + 1 > budget_chars:  # +1 for closing bracket
            break
        if not first:
            parts.append(",")
        parts.append(js)
        total_len += add_len
        kept.append(s)
        first = False
    parts.append("]")
    return kept, "".join(parts)


def ensure_categories_response(raw: str) -> Dict[str, Any]:
    """Parse model output into {year?, categories: [...]}. Allows either object or list."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\n|\n```$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
    data: Any
    try:
        data = json.loads(cleaned)
    except Exception:
        # Try to find first JSON object or array
        m = re.search(r"\{.*\}|\[.*\]", cleaned, flags=re.DOTALL)
        if not m:
            raise ValueError("Model response was not valid JSON")
        data = json.loads(m.group(0))

    if isinstance(data, list):
        data = {"categories": data}
    if not isinstance(data, dict):
        raise ValueError("Model response must be a JSON object or array")

    cats = data.get("categories")
    if not isinstance(cats, list):
        raise ValueError("Missing 'categories' list in response")

    normd: List[Dict[str, Any]] = []
    for c in cats:
        if not isinstance(c, dict):
            continue
        name = (c.get("name") or c.get("category") or "").strip()
        definition = (c.get("definition") or c.get("description") or "").strip()
        rep_id = (c.get("representative_snippet_id") or c.get("snippet_id") or "").strip()
        rep_text = (c.get("representative_snippet") or c.get("snippet") or "").strip()
        if not name or not definition or not rep_text:
            continue
        normd.append({
            "name": name,
            "definition": definition,
            "representative_snippet_id": rep_id or None,
            "representative_snippet": rep_text,
        })
    data["categories"] = normd
    return data


SYSTEM_PROMPT = (
    "You are an expert analyst categorizing risk-related excerpts from public pension ACFR/CAFR reports."
    " Return STRICT JSON only. No markdown, no prose."
)


USER_PROMPT_TEMPLATE = (
    "You are given risk-related snippets for fiscal year {year}.\n"
    "Your task: produce EXACTLY 10 NON-OVERLAPPING risk categories that summarize the themes across the snippets.\n"
    "Constraints:\n"
    "- short category names (<= 5 words).\n"
    "- short definitions (<= 25 words).\n"
    "- choose 1 representative snippet for each category from the provided snippets ONLY, verbatim.\n"
    "- include the representative snippet's id if available.\n"
    "Output JSON with shape: {\"year\": <int>, \"categories\": [ {\"name\": <string>, \"definition\": <string>, \"representative_snippet_id\": <string or null>, \"representative_snippet\": <string> } x 10 ] }.\n"
    "Do not add extra fields.\n\n"
    "Snippets (JSON array of objects with id,text,section_hint):\n{snippets_json}\n"
)


def call_openai_categories(
    client: Any,
    year: int,
    snippets: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    max_retries: int = 3,
    retry_base: float = 1.5,
    char_budget: int = 100000,
) -> Dict[str, Any]:
    kept, snippets_json = pack_snippets_to_budget(snippets, budget_chars=char_budget)
    user_prompt = USER_PROMPT_TEMPLATE.replace("{year}", str(year)).replace("{snippets_json}", snippets_json)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            kwargs: Dict[str, Any] = {"model": "gpt-5-mini", "messages": messages}
            if temperature is not None:
                kwargs["temperature"] = temperature

            resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content or ""
            data = ensure_categories_response(raw)
            # Enforce exactly 10 categories: if not, perform a corrective retry once
            cats = data.get("categories", [])
            if isinstance(cats, list) and len(cats) == 10:
                # Optional: light verbatim check for representative snippets
                id_to_text = {s["id"]: s["text"] for s in kept if s.get("id")}
                for c in cats:
                    rep_text = c.get("representative_snippet") or ""
                    rep_id = c.get("representative_snippet_id") or None
                    if rep_id and rep_id in id_to_text:
                        if not verify_verbatim(rep_text, id_to_text[rep_id]):
                            # If mismatch, drop id to avoid misleading linkage
                            c["representative_snippet_id"] = None
                data["year"] = year
                return data

            # If first parse failed the 10-count requirement, try a corrective prompt once
            if attempt == 0:
                fix_prompt = (
                    "Your previous output did not have exactly 10 categories."
                    " Return JSON with exactly 10 categories as specified."
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": fix_prompt})
                # Continue loop to retry
                continue

            # Otherwise, coerce to 10 by trimming or skipping
            if isinstance(cats, list):
                data["categories"] = cats[:10]
                data["year"] = year
                return data

            # Fallback
            data = {"year": year, "categories": []}
            return data

        except Exception as e:
            err_text = str(e).lower()
            if temperature is not None and (
                "temperature" in err_text and (
                    "not supported" in err_text
                    or "unsupported" in err_text
                    or "unexpected" in err_text
                    or "invalid" in err_text
                )
            ):
                try:
                    resp = client.chat.completions.create(
                        model="gpt-5-mini",
                        messages=messages,
                    )
                    raw = resp.choices[0].message.content or ""
                    data = ensure_categories_response(raw)
                    cats = data.get("categories", [])
                    data["categories"] = cats[:10] if isinstance(cats, list) else []
                    data["year"] = year
                    return data
                except Exception as e2:
                    e = e2
            last_err = e
            time.sleep((retry_base ** attempt) + 0.1 * attempt)
    raise RuntimeError(f"OpenAI call failed after {max_retries} attempts: {last_err}")


def write_year_categories(output_root: Path, year: int, data: Dict[str, Any]) -> Path:
    cats_dir = output_root / "categories"
    cats_dir.mkdir(parents=True, exist_ok=True)
    out_path = cats_dir / f"{year}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"year": year, "categories": data.get("categories", [])}, f, ensure_ascii=False, indent=2)
    return out_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pass 2: Group snippets by year and generate 10 risk categories per year.")
    parser.add_argument("--input-dir", default="outputs", help="Folder with pass 1 .jsonl snippet files (default: outputs)")
    parser.add_argument("--output-dir", default="output_pass2", help="Folder to write pass 2 results; categories saved in <output-dir>/categories (default: output_pass2)")
    parser.add_argument("--char-budget", type=int, default=45000, help="Max characters of snippets to send per year")
    parser.add_argument("--max-years", type=int, default=0, help="Limit number of years processed (0 = all)")
    parser.add_argument("--temperature", type=float, default=None, help="Optional temperature parameter")

    args = parser.parse_args()
    in_dir = Path(args.input_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if OpenAI is None:
        raise RuntimeError("openai SDK not installed. Run: pip install -r requirements.txt")

    client = OpenAI()

    if not in_dir.exists() or not in_dir.is_dir():
        print(f"Input directory not found: {in_dir}")
        return

    by_year = group_snippets_by_year(in_dir)
    if not by_year:
        print(f"No snippets found in {in_dir}")
        return

    years = sorted(by_year.keys())
    if args.max_years and args.max_years > 0:
        years = years[: args.max_years]

    for y in years:
        print(f"Processing year: {y}  (snippets: {len(by_year[y])})")
        try:
            data = call_openai_categories(
                client,
                y,
                by_year[y],
                temperature=args.temperature,
                char_budget=args.char_budget,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            data = {"year": y, "categories": []}

        out_path = write_year_categories(out_root, y, data)
        print(f"  Wrote: {out_path}")


if __name__ == "__main__":
    main()
