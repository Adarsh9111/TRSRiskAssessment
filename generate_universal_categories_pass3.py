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


def read_year_categories(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        if "categories" not in data or not isinstance(data["categories"], list):
            return None
        return data
    except Exception:
        return None


def collect_all_categories(input_dir: Path) -> List[Dict[str, Any]]:
    """Flatten categories across years; keep minimal fields and year for context."""
    items: List[Dict[str, Any]] = []
    files = sorted([p for p in input_dir.glob("*.json") if p.is_file()])
    for fp in files:
        data = read_year_categories(fp)
        if not data:
            continue
        year = data.get("year")
        for c in data.get("categories", []) or []:
            name = (c.get("name") or "").strip()
            definition = (c.get("definition") or "").strip()
            rep = (c.get("representative_snippet") or "").strip()
            if not name or not definition or not rep:
                # Keep only fully specified items for best signal
                continue
            items.append({
                "year": year,
                "name": name,
                "definition": definition,
                "representative_snippet": rep,
            })
    return items


def pack_items_to_budget(items: List[Dict[str, Any]], budget_chars: int = 45000) -> Tuple[List[Dict[str, Any]], str]:
    """Pack items into JSON array string without exceeding char budget."""
    # Sort by representative snippet length to interleave and balance context
    ordered = sorted(items, key=lambda x: len(x.get("representative_snippet", "")))
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
    total_len = 2  # brackets
    parts: List[str] = ["["]
    first = True
    for it in interleaved:
        frag = {
            "year": it.get("year"),
            "name": it.get("name"),
            "definition": it.get("definition"),
            "representative_snippet": it.get("representative_snippet"),
        }
        js = json.dumps(frag, ensure_ascii=False)
        add_len = len(js) + (0 if first else 1)
        if total_len + add_len + 1 > budget_chars:
            break
        if not first:
            parts.append(",")
        parts.append(js)
        kept.append(it)
        total_len += add_len
        first = False
    parts.append("]")
    return kept, "".join(parts)


def ensure_global_categories_response(raw: str) -> Dict[str, Any]:
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

    if isinstance(data, list):
        data = {"categories": data}
    if not isinstance(data, dict):
        raise ValueError("Model response must be object or array")

    cats = data.get("categories")
    if not isinstance(cats, list):
        raise ValueError("Missing 'categories' list")

    norm: List[Dict[str, Any]] = []
    for c in cats:
        if not isinstance(c, dict):
            continue
        name = (c.get("name") or c.get("category") or "").strip()
        definition = (c.get("definition") or c.get("description") or "").strip()
        criteria = c.get("criteria")
        if isinstance(criteria, list):
            crit = [str(x).strip() for x in criteria if str(x).strip()][:6]
        elif isinstance(criteria, str):
            # Split bullets by newline or semicolon
            parts = re.split(r"[\n;]+", criteria)
            crit = [p.strip(" -*•\t ") for p in parts if p.strip()][:6]
        else:
            crit = []

        if not name or not definition:
            continue

        # Enforce max word counts gently
        name_words = len(re.findall(r"\w+", name))
        if name_words > 8:  # be tolerant if model overshoots
            # Keep first 8 words to avoid exploding
            name = " ".join(re.findall(r"\w+", name)[:8])

        norm.append({
            "name": name,
            "definition": definition,
            "criteria": crit[:6],
        })

    data["categories"] = norm
    return data


SYSTEM_PROMPT = (
    "You consolidate multiple years of pension risk categories into exactly five universal, orthogonal risk categories."
    " Return STRICT JSON only. No markdown or commentary."
)


USER_PROMPT_TEMPLATE = (
    "Here are the 10-category outputs from several years. From all of them, create exactly five global categories that are stable across years.\n"
    "For each global category, return: name (≤4 words), definition (≤25 words), criteria (3–6 short cues).\n"
    "Constraints: categories must be mutually exclusive (orthogonal), collectively exhaustive across inputs, and consistently applicable across all years.\n\n"
    "Input categories (JSON array of {year,name,definition,representative_snippet}):\n{items_json}\n\n"
    "Output JSON with shape: {\"categories\": [ {\"name\": <string>, \"definition\": <string>, \"criteria\": [<string>, ... 3-6 items] } x 5 ] }."
)


def call_openai_global_categories(
    client: Any,
    items: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    max_retries: int = 3,
    retry_base: float = 1.5,
    char_budget: int = 100000,
) -> Dict[str, Any]:
    kept, items_json = pack_items_to_budget(items, budget_chars=char_budget)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.replace("{items_json}", items_json)},
    ]

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            kwargs: Dict[str, Any] = {"model": "gpt-5-mini", "messages": messages}
            if temperature is not None:
                kwargs["temperature"] = temperature
            resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content or ""
            data = ensure_global_categories_response(raw)
            cats = data.get("categories", [])
            if isinstance(cats, list) and len(cats) == 5:
                return {"categories": cats}

            if attempt == 0:
                fix_prompt = (
                    "Your previous output did not have exactly 5 categories or the structure was off."
                    " Return JSON with exactly five categories as specified."
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": fix_prompt})
                continue

            # Coerce to 5 by slicing if needed
            if isinstance(cats, list):
                return {"categories": cats[:5]}
            return {"categories": []}

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
                    data = ensure_global_categories_response(raw)
                    cats = data.get("categories", [])
                    return {"categories": cats[:5] if isinstance(cats, list) else []}
                except Exception as e2:
                    e = e2
            last_err = e
            time.sleep((retry_base ** attempt) + 0.1 * attempt)
    raise RuntimeError(f"OpenAI call failed after {max_retries} attempts: {last_err}")


def write_global_categories(output_dir: Path, data: Dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "global_categories.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"categories": data.get("categories", [])}, f, ensure_ascii=False, indent=2)
    return out_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pass 3: Consolidate all years into five universal, orthogonal risk categories.")
    parser.add_argument("--input-dir", default="output_pass2/categories", help="Folder with per-year categories JSON files")
    parser.add_argument("--output-dir", default="output_pass3", help="Folder to write global categories JSON")
    parser.add_argument("--char-budget", type=int, default=45000, help="Max characters to send to the model")
    parser.add_argument("--temperature", type=float, default=None, help="Optional temperature parameter")

    args = parser.parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    if OpenAI is None:
        raise RuntimeError("openai SDK not installed. Run: pip install -r requirements.txt")

    if not in_dir.exists() or not in_dir.is_dir():
        print(f"Input directory not found: {in_dir}")
        return

    items = collect_all_categories(in_dir)
    if not items:
        print(f"No categories found in {in_dir}")
        return

    client = OpenAI()

    print(f"Collected {len(items)} categories across years. Sending to model…")
    try:
        data = call_openai_global_categories(
            client,
            items,
            temperature=args.temperature,
            char_budget=args.char_budget,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        data = {"categories": []}

    out_path = write_global_categories(out_dir, data)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

