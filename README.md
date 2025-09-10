Risk Passage Extraction from CAFR Texts

Overview
- Loops through all `.txt` files in `CAFR_text/`.
- For each file, splits content into overlapping chunks and calls the OpenAI Chat Completions API (`gpt-5-mini`).
- Prompts the model to return up to 50 verbatim risk-related passages per chunk as a JSON array.
- Writes one JSONL file per input `.txt` file into `outputs/`, where each line contains: `id`, `year`, `source_file`, `quote`, `section_hint`.

Requirements
- Python 3.9+
- An OpenAI API key in env var `OPENAI_API_KEY`.
- Dependencies: `pip install -r requirements.txt`

Usage
1) Put your input `.txt` files under `CAFR_text/`.
2) Export your API key: `export OPENAI_API_KEY=sk-...`
3) Run the extractor:

   python extract_risks.py --input-dir CAFR_text --output-dir outputs \
       --chunk-chars 15000 --overlap 500

Notes
- Output files are named `<input_stem>.jsonl` under `outputs/`.
- `year` is parsed from filenames like `FY2024.txt` or `CAFR_2019.txt` when possible.
- The script prints progress per file and per chunk, and warns if a returned quote may not be strictly verbatim in the source chunk.
- If no relevant passages are found for a chunk, nothing is written for that chunk; the output file will still be created for the input file (possibly empty).

CLI Flags
- `--input-dir`: Directory containing `.txt` files (default `CAFR_text`).
- `--output-dir`: Directory to write `.jsonl` files (default `outputs`).
- `--chunk-chars`: Character size per chunk (default `15000`).
- `--overlap`: Overlap characters between chunks (default `500`).
- `--max-files`: Process at most N files (0 = all).

