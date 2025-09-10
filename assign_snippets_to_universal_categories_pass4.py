import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
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


def find_year_from_name(name: str) -> Optional[int]:
    """Try to extract a year from filenames like 'FY2024', 'ACFR23', '2014'."""
    # 4-digit year
    m4 = re.findall(r"((?:19|20)\d{2})", name)
    if m4:
        try:
            return int(m4[-1])
        except Exception:
            pass

    # FY two-digit like FY24
    m_fy2 = re.search(r"FY\s*[-_ ]?(\d{2})", name, flags=re.IGNORECASE)
    if m_fy2:
        yy = int(m_fy2.group(1))
        return 1900 + yy if yy >= 90 else 2000 + yy

    # ACFR/CAFR two-digit like ACFR23
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
                continue
    return rows


def load_global_categories(categories_path: Optional[Path]) -> List[Dict[str, Any]]:
    """Load 5 universal categories from output_pass3 or outputpass3.

    If categories_path is provided, use it. Otherwise try common defaults.
    Expected shape: {"categories": [{"name","definition","criteria":[...]}, ... 5 items ...]}
    """
    tried: List[Path] = []
    if categories_path is not None:
        tried.append(categories_path)
    else:
        tried.extend([
            Path("output_pass3") / "global_categories.json",
            Path("outputpass3") / "global_categories.json",
        ])

    for p in tried:
        if p.exists() and p.is_file():
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
            cats = data.get("categories") if isinstance(data, dict) else None
            if isinstance(cats, list) and len(cats) >= 1:
                norm: List[Dict[str, Any]] = []
                for c in cats:
                    if not isinstance(c, dict):
                        continue
                    name = (c.get("name") or c.get("category") or "").strip()
                    definition = (c.get("definition") or c.get("description") or "").strip()
                    criteria = c.get("criteria")
                    if isinstance(criteria, list):
                        crit = [str(x).strip() for x in criteria if str(x).strip()]
                    elif isinstance(criteria, str):
                        parts = re.split(r"[\n;]+", criteria)
                        crit = [p.strip(" -*•\t ") for p in parts if p.strip()]
                    else:
                        crit = []
                    if name and definition:
                        norm.append({"name": name, "definition": definition, "criteria": crit})
                return norm
    raise FileNotFoundError(
        f"Global categories file not found. Tried: {', '.join(str(x) for x in tried)}"
    )


def group_snippets_by_year(input_dir: Path) -> Dict[int, List[Dict[str, Any]]]:
    by_year: Dict[int, List[Dict[str, Any]]] = {}
    files = sorted([p for p in input_dir.glob("*.jsonl") if p.is_file()])
    for fp in files:
        rows = read_jsonl(fp)
        file_year_hint = find_year_from_name(fp.name)
        for r in rows:
            y = r.get("year") if isinstance(r.get("year"), int) else None
            if y is None:
                src = r.get("source_file") or fp.name
                y = find_year_from_name(str(src)) or file_year_hint
            if y is None:
                continue
            quote = (r.get("quote") or "").strip()
            if not quote:
                continue
            item = {
                "id": r.get("id") or f"{fp.name}:{len(by_year.get(y, []))}",
                "text": quote,
                "section_hint": r.get("section_hint"),
            }
            by_year.setdefault(y, []).append(item)
    return by_year


def pack_items(snippets: List[Dict[str, Any]], budget_chars: int = 45000) -> Tuple[List[Dict[str, Any]], str]:
    # Balance by interleaving short/long texts
    ordered = sorted(snippets, key=lambda s: len(s.get("text", "")))
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

    kept: List[Dict[str, Any]] = []
    total_len = 2
    parts: List[str] = ["["]
    first = True
    for s in interleaved:
        frag = {
            "id": s["id"],
            "text": s["text"],
            "section_hint": s.get("section_hint"),
        }
        js = json.dumps(frag, ensure_ascii=False)
        add_len = len(js) + (0 if first else 1)
        if total_len + add_len + 1 > budget_chars:
            break
        if not first:
            parts.append(",")
        parts.append(js)
        kept.append(s)
        total_len += add_len
        first = False
    parts.append("]")
    return kept, "".join(parts)


# ----------------------------
# Model prompting
# ----------------------------

SYSTEM_PROMPT = (
    "You are a precise classifier assigning verbatim risk-related snippets to exactly one of five universal risk categories."
    " Return STRICT JSON only. No markdown."
)


USER_PROMPT_TEMPLATE = (
    "Classify each snippet into one of the five categories below.\n"
    "Use only these category names verbatim. Do not invent categories.\n"
    "If a snippet does not fit any category reasonably, omit it.\n"
    "Output JSON object: {\"categories\": [ {\"name\": <category_name>, \"snippet_ids\": [<id>, ...] } x 5 ] }\n\n"
    "Categories (JSON array of {name, definition, criteria[]}):\n{categories_json}\n\n"
    "Snippets (JSON array of {id, text, section_hint}):\n{snippets_json}\n"
)


def ensure_assignments_response(raw: str, allowed_names: List[str]) -> Dict[str, List[str]]:
    """Return mapping of category_name -> list of snippet_ids (strings)."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\n|\n```$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()

    data: Any
    try:
        data = json.loads(cleaned)
    except Exception:
        m = re.search(r"\{.*\}|\[.*\]", cleaned, flags=re.DOTALL)
        if not m:
            raise ValueError("Model response not JSON")
        data = json.loads(m.group(0))

    mapping: Dict[str, List[str]] = {name: [] for name in allowed_names}

    # Preferred structure: {"categories":[{"name":..., "snippet_ids":[...]}, ...]}
    if isinstance(data, dict) and isinstance(data.get("categories"), list):
        for c in data["categories"]:
            if not isinstance(c, dict):
                continue
            name = (c.get("name") or c.get("category") or "").strip()
            ids = c.get("snippet_ids")
            if name in mapping and isinstance(ids, list):
                mapping[name].extend([str(x) for x in ids if str(x).strip()])
        return mapping

    # Alternate: list of assignments [{category, snippet_id}]
    if isinstance(data, list):
        for it in data:
            if not isinstance(it, dict):
                continue
            name = (it.get("category") or it.get("name") or "").strip()
            sid = it.get("snippet_id")
            if name in mapping and isinstance(sid, (str, int)):
                mapping[name].append(str(sid))
        return mapping

    # Alternate: object mapping name -> ids array
    if isinstance(data, dict):
        for k, v in data.items():
            if k in mapping and isinstance(v, list):
                mapping[k].extend([str(x) for x in v if str(x).strip()])
        return mapping

    return mapping


def call_openai_assign(
    client: Any,
    categories: List[Dict[str, Any]],
    snippets: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    max_retries: int = 3,
    retry_base: float = 1.5,
    char_budget: int = 45000,
) -> Dict[str, List[str]]:
    # Prepare inputs within budget
    cats_for_prompt = [{
        "name": c.get("name"),
        "definition": c.get("definition"),
        "criteria": c.get("criteria", []),
    } for c in categories]
    cats_json = json.dumps(cats_for_prompt, ensure_ascii=False)
    kept_snips, snips_json = pack_items(snippets, budget_chars=char_budget - len(cats_json) - 1000)

    allowed_names = [c.get("name") for c in categories if c.get("name")]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.replace("{categories_json}", cats_json).replace("{snippets_json}", snips_json)},
    ]

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            kwargs: Dict[str, Any] = {"model": "gpt-5-mini", "messages": messages}
            if temperature is not None:
                kwargs["temperature"] = temperature
            resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content or ""
            mapping = ensure_assignments_response(raw, allowed_names)
            # Sanity: ensure only allowed ids from kept_snips
            kept_ids = {s["id"] for s in kept_snips if s.get("id")}
            for k in list(mapping.keys()):
                mapping[k] = [sid for sid in mapping[k] if sid in kept_ids]
            return mapping

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
                    mapping = ensure_assignments_response(raw, allowed_names)
                    kept_ids = {s["id"] for s in kept_snips if s.get("id")}
                    for k in list(mapping.keys()):
                        mapping[k] = [sid for sid in mapping[k] if sid in kept_ids]
                    return mapping
                except Exception as e2:
                    e = e2
            last_err = e
            time.sleep((retry_base ** attempt) + 0.1 * attempt)
    raise RuntimeError(f"OpenAI call failed after {max_retries} attempts: {last_err}")


# ----------------------------
# CSV writing
# ----------------------------

def write_year_csv(output_dir: Path, year: int, mapping: Dict[str, List[str]], id_to_text: Dict[str, str]) -> Path:
    out_dir = output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{year}.csv"

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # Columns exactly as requested
        writer.writerow(["year", "category", "verbatim"])
        for category, ids in mapping.items():
            for sid in ids:
                text = id_to_text.get(sid, "")
                writer.writerow([year, category, text])
    return out_path


# ----------------------------
# Main
# ----------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pass 4: Assign snippets to five universal categories per year and write CSVs.")
    parser.add_argument("--input-dir", default="outputs", help="Folder with pass-1 .jsonl snippet files")
    parser.add_argument("--categories-file", default=None, help="Path to global categories JSON (defaults to output_pass3/global_categories.json or outputpass3/global_categories.json)")
    parser.add_argument("--output-dir", default="output_pass4", help="Folder to write per-year CSV files")
    parser.add_argument("--char-budget", type=int, default=45000, help="Max characters to send to the model per year")
    parser.add_argument("--temperature", type=float, default=None, help="Optional temperature parameter")
    parser.add_argument("--max-years", type=int, default=0, help="Limit number of years processed (0 = all)")

    args = parser.parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    categories_path = Path(args.categories_file) if args.categories_file else None

    if OpenAI is None:
        raise RuntimeError("openai SDK not installed. Run: pip install -r requirements.txt")

    categories = load_global_categories(categories_path)
    if len(categories) < 5:
        print(f"Warning: expected 5 global categories, found {len(categories)}")

    by_year = group_snippets_by_year(in_dir)
    if not by_year:
        print(f"No snippets found in {in_dir}")
        return

    years = sorted(by_year.keys())
    if args.max_years and args.max_years > 0:
        years = years[: args.max_years]

    client = OpenAI()

    for y in years:
        snippets = by_year[y]
        # Prepare id->text map for CSV writing
        id_to_text = {s["id"]: s["text"] for s in snippets if s.get("id")}
        print(f"Processing year {y} with {len(snippets)} snippets…")
        try:
            mapping = call_openai_assign(
                client,
                categories,
                snippets,
                temperature=args.temperature,
                char_budget=args.char_budget,
            )
        except Exception as e:
            print(f"  ERROR assigning categories for {y}: {e}")
            mapping = {c.get("name"): [] for c in categories if c.get("name")}

        out_path = write_year_csv(out_dir, y, mapping, id_to_text)
        print(f"  Wrote: {out_path}")


if __name__ == "__main__":
    main()

