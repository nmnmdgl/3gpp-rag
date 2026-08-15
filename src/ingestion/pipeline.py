import json
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from .docx_parser import parse_docx
from .chunker import make_chunks


EXPECTED_FILES = {
    "21905-j20.docx",
    "23501-k20.docx",
    "23502-k20.docx",
    "38300-fn0.docx",
}


def run(
    raw_dir="data/raw_data",
    output_dir="data/processed",
):
    raw = Path(raw_dir)
    out = Path(output_dir)

    out.mkdir(parents=True, exist_ok=True)

    files = sorted(raw.glob("*.docx"))

    if not files:
        raise FileNotFoundError(
            f"No DOCX files found in {raw.resolve()}"
        )

    found_names = {f.name for f in files}
    missing = EXPECTED_FILES - found_names

    if missing:
        raise FileNotFoundError(
            "Expected exactly the four assignment documents. "
            f"Missing: {sorted(missing)}"
        )

    all_chunks = []
    manifest = []

    for path in tqdm(files, desc="Parsing 3GPP documents"):
        metadata, blocks = parse_docx(str(path))
        chunks = make_chunks(metadata, blocks)

        manifest.append(asdict(metadata))
        all_chunks.extend(asdict(c) for c in chunks)

        parsed_path = (
            out / "parsed" / f"{path.stem}.json"
        )

        parsed_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        parsed_path.write_text(
            json.dumps(
                {
                    "metadata": asdict(metadata),
                    "blocks": [asdict(b) for b in blocks],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    chunks_dir = out / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    (chunks_dir / "chunks.json").write_text(
        json.dumps(
            all_chunks,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (out / "manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Documents: {len(manifest)}")
    print(f"Chunks: {len(all_chunks)}")
    print(f"Output: {out.resolve()}")


if __name__ == "__main__":
    run()
