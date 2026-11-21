from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "qasper"


def load_qasper() -> Dataset:
    if DATASET_PATH.exists():
        print("Loading QASPER from local storage...")
        dataset = load_from_disk(str(DATASET_PATH))
    else:
        print("Downloading QASPER from Hugging Face...")
        dataset = load_dataset("allenai/qasper", split="train")
        
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(DATASET_PATH))
        
        print("QASPER saved locally.")

    if not isinstance(dataset, Dataset):
        raise TypeError("Expected a Hugging Face Dataset")

    return dataset