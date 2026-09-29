import csv
from pathlib import Path


LOG_COLUMNS = ["epoch", "train_loss", "val_loss", "val_accuracy", "learning_rate"]


def append_epoch_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow({key: row[key] for key in LOG_COLUMNS})

