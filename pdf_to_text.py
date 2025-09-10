import argparse
from pathlib import Path
from typing import Optional, Tuple


def _try_pdfminer(pdf_path: Path) -> Optional[str]:
    try:
        from pdfminer.high_level import extract_text  # type: ignore

        return extract_text(str(pdf_path))
    except Exception:
        return None


def _try_pypdf(pdf_path: Path) -> Optional[str]:
    try:
        try:
            # Prefer pypdf if available
            from pypdf import PdfReader  # type: ignore
        except Exception:
            # Fallback to PyPDF2 name if installed
            from PyPDF2 import PdfReader  # type: ignore

        reader = PdfReader(str(pdf_path))
        parts = []
        for page in reader.pages:
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            if txt:
                parts.append(txt)
        return "\n".join(parts)
    except Exception:
        return None


def extract_text_from_pdf(pdf_path: Path) -> Tuple[str, str]:
    """
    Attempt to extract text using pdfminer.six, falling back to pypdf/PyPDF2.

    Returns (text, method_used).
    Raises RuntimeError if extraction fails by all methods.
    """
    text = _try_pdfminer(pdf_path)
    if isinstance(text, str) and text.strip():
        return text, "pdfminer"

    text2 = _try_pypdf(pdf_path)
    if isinstance(text2, str) and text2.strip():
        return text2, "pypdf"

    # If both produced something but empty, prefer the non-None result
    if isinstance(text, str) and text != None:
        return text or "", "pdfminer"
    if isinstance(text2, str) and text2 != None:
        return text2 or "", "pypdf"

    raise RuntimeError("No PDF text extraction backends available or all failed")


def convert_pdfs(input_dir: Path, output_dir: Path, overwrite: bool = True) -> None:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found or not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {input_dir}")

    for pdf_path in pdf_files:
        txt_name = pdf_path.stem + ".txt"
        out_path = output_dir / txt_name
        if out_path.exists() and not overwrite:
            print(f"Skipping (exists): {out_path}")
            continue

        print(f"Converting: {pdf_path.name} -> {out_path.name}")
        try:
            text, method = extract_text_from_pdf(pdf_path)
        except Exception as e:
            print(f"  ERROR extracting {pdf_path.name}: {e}")
            continue

        try:
            out_path.write_text(text, encoding="utf-8", errors="replace")
            note = " (empty)" if not text.strip() else ""
            print(f"  Wrote {out_path.name} via {method}{note}")
        except Exception as e:
            print(f"  ERROR writing {out_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert all PDFs in a folder to text files.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="CAFR",
        help="Input folder containing PDF files (default: CAFR)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="CAFR_text",
        help="Output folder to write .txt files (default: CAFR_text)",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip writing if the output .txt already exists.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    convert_pdfs(input_dir, output_dir, overwrite=not args.no_overwrite)


if __name__ == "__main__":
    main()

