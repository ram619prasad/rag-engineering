
import hashlib
import json
from pathlib import Path

from advanced_retrieval.loaders.download_documents import (
    DOCUMENTS,
    OUTPUT_DIR,
)


def freeze_corpus():
    records = []

    for filename, url in DOCUMENTS.items():
        path = OUTPUT_DIR / filename

        if not path.exists():
            raise FileNotFoundError(path)

        content = path.read_bytes()

        document_id = path.stem
        domain = document_id.split("_", 1)[0]

        records.append({
            "id": document_id,
            "file": filename,
            "domain": domain,
            "source": url,
            "sha256": hashlib.sha256(content).hexdigest(),
            "characters": len(
                content.decode("utf-8")
            ),
        })

    manifest = {
        "name": "engineering_knowledge_base",
        "version": "1.0",
        "documents": records,
    }

    output_path = OUTPUT_DIR.parent / "sources.json"

    output_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Documents frozen: {len(records)}")
    print(f"Manifest: {output_path}")


if __name__ == "__main__":
    freeze_corpus()
