import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    # New SDK style
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


SYSTEM_PROMPT = (
    "You are given an annual report Teachers Retirement System of State of Illinois. You have to extract strictly verbatim passages mentioning risks, challenges, concerns, or sustainability concerns in the public pension fund report. This is an exercise to extract all sentences that might indicate some sort of risk or challenge mentioned in the report."
    "Do not paraphrase or summarize. Return JSON only."
)


USER_PROMPT_TEMPLATE = (
    "From the text below, return passages of 1–4 sentences that  describe any type of risks the fund might be facing."
    "Output JSON array of items: { 'quote': <string>, 'section_hint': <string or null> }. "
    "If nothing relevant, return an empty array. Text: {chunk}"
)


def find_year_from_name(name: str) -> Optional[int]:
    """Try to extract a 4-digit year from filename like 'FY2024.txt' or 'CAFR_2019.txt'."""
    # Prefer 'FY2024' style if present; else any 20xx/19xx
    m = re.search(r"FY\s*[-_]?\s*(19|20)(\d{2})", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1) + m.group(2))
    m2 = re.findall(r"(19|20)\d{2}", name)
    if m2:
        return int(m2[-1])
    return None


def chunk_text(text: str, max_chars: int = 15000, overlap: int = 500) -> List[str]:
    """Split text into overlapping character chunks to fit model context."""
    if max_chars <= 0:
        return [text]
    chunks: List[str] = []
    i = 0
    n = len(text)
    if n == 0:
        return [""]
    while i < n:
        end = min(i + max_chars, n)
        chunk = text[i:end]
        chunks.append(chunk)
        if end >= n:
            break
        i = max(0, end - overlap)
    return chunks


def _normalize(s: str) -> str:
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"\s+", " ", s.strip())
    return s


def ensure_array_from_response(raw: str) -> List[Dict[str, Any]]:
    """Coerce model output into a list of {quote, section_hint} dicts."""
    cleaned = raw.strip()
    # Remove code fences if any
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\n|\n```$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()

    # Try direct JSON parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return _coerce_items(data)
        if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
            return _coerce_items(data["items"])  # some models wrap in an object
    except Exception:
        pass

    # Fallback: extract first JSON array substring
    m = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return _coerce_items(data)
        except Exception:
            pass

    raise ValueError("Model response was not valid JSON array of items")


def _coerce_items(items: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        quote = it.get("quote")
        section_hint = it.get("section_hint") if "section_hint" in it else None
        if quote is None:
            continue
        if section_hint is not None and not isinstance(section_hint, (str, type(None))):
            section_hint = str(section_hint)
        out.append({"quote": str(quote).strip(), "section_hint": section_hint})
    return out


def verify_verbatim(quote: str, source: str) -> bool:
    """Light check: ensure quote appears in source when normalized."""
    nq = _normalize(quote)
    ns = _normalize(source)
    return nq in ns


def call_openai(
    client: Any,
    chunk: str,
    temperature: Optional[float] = None,
    max_retries: int = 3,
    retry_base: float = 1.5,
) -> List[Dict[str, Any]]:
    user_prompt = USER_PROMPT_TEMPLATE.replace("{chunk}", chunk)
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            kwargs: Dict[str, Any] = {"model": "gpt-5-mini", "messages": messages}
            if temperature is not None:
                kwargs["temperature"] = temperature

            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            return ensure_array_from_response(content)
        except Exception as e:
            # If model complains about temperature, retry immediately without it once
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
                    content = resp.choices[0].message.content or ""
                    return ensure_array_from_response(content)
                except Exception as e2:
                    e = e2
            last_err = e
            # Exponential backoff
            time.sleep((retry_base ** attempt) + 0.1 * attempt)
    raise RuntimeError(f"OpenAI call failed after {max_retries} attempts: {last_err}")


def process_file(
    client: Any,
    txt_path: Path,
    output_dir: Path,
    chunk_chars: int = 15000,
    overlap: int = 500,
    temperature: Optional[float] = None,
) -> None:
    print(f"Processing: {txt_path.name}")
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_text(text, max_chars=chunk_chars, overlap=overlap)
    year = find_year_from_name(txt_path.name)
    base_stem = txt_path.stem  # keep 'FY2024' from 'FY2024.txt'
    out_path = output_dir / f"{base_stem}.jsonl"

    # Always create/overwrite output file, even if empty at the end
    # We will write incrementally for visibility
    with out_path.open("w", encoding="utf-8") as fw:
        total_written = 0
        for ci, chunk in enumerate(chunks, start=1):
            try:
                items = call_openai(client, chunk, temperature=temperature)
                # Cap to 50 passages per chunk as requested
                if len(items) > 50:
                    items = items[:50]
            except Exception as e:
                print(f"  Chunk {ci}: ERROR calling model: {e}")
                continue

            if not items:
                print(f"  Chunk {ci}: no items")
                continue

            valid_count = 0
            for si, item in enumerate(items, start=1):
                quote = (item.get("quote") or "").strip()
                section_hint = item.get("section_hint")
                if quote == "":
                    continue
                # Validate verbatim-ness lightly
                if not verify_verbatim(quote, chunk):
                    # Keep it but log a warning; user requested verbatim
                    print(f"    Warning: Quote may not be verbatim in chunk (c{ci}s{si}).")

                row = {
                    "id": f"{txt_path.name}_chunk{ci}_snip{si}",
                    "year": year,
                    "source_file": txt_path.name,
                    "quote": quote,
                    "section_hint": section_hint if section_hint is not None else None,
                }
                fw.write(json.dumps(row, ensure_ascii=False) + "\n")
                valid_count += 1
                total_written += 1

            print(f"  Chunk {ci}: wrote {valid_count} items")
            # Ensure visibility on disk for external readers (tail/cat)
            fw.flush()

    print(f"  Done -> {out_path} (total lines: {total_written})")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract risk passages from CAFR text files via OpenAI.")
    parser.add_argument("--input-dir", default="CAFR_text", help="Folder with .txt files")
    parser.add_argument("--output-dir", default="outputs", help="Folder to write .jsonl files")
    parser.add_argument("--chunk-chars", type=int, default=15000, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=500, help="Overlap between chunks in characters")
    parser.add_argument("--max-files", type=int, default=0, help="Process at most N files (0 = all)")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional temperature; if the model rejects this parameter, the script automatically retries without it.",
    )

    args = parser.parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if OpenAI is None:
        raise RuntimeError(
            "openai SDK not installed. Run: pip install -r requirements.txt"
        )

    client = OpenAI()

    txt_files = sorted([p for p in in_dir.glob("*.txt") if p.is_file()])
    if not txt_files:
        print(f"No .txt files found in {in_dir}")
        return

    limit = args.max_files if args.max_files and args.max_files > 0 else len(txt_files)
    for i, path in enumerate(txt_files[:limit], start=1):
        process_file(
            client,
            path,
            out_dir,
            chunk_chars=args.chunk_chars,
            overlap=args.overlap,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()
