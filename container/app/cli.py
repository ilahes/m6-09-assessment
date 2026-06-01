import argparse
import csv
import json
import sys
from pathlib import Path

from detector import CatDetector

STUDENT_JSON = Path("/app/STUDENT.json")
INPUT_DIR    = Path("/data/input")
OUTPUT_DIR   = Path("/data/output")
EXTENSIONS   = {".jpg", ".jpeg", ".png"}


def cmd_info(_args):
    print(STUDENT_JSON.read_text())


def cmd_predict(_args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "predictions.csv"

    detector = CatDetector()
    images = sorted(
        p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in EXTENSIONS
    )

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_path", "xmin", "ymin", "xmax", "ymax", "confidence", "class"])

        for img_path in images:
            rel_str = img_path.relative_to(INPUT_DIR).as_posix()
            dets = detector.predict(str(img_path))
            if not dets:
                writer.writerow([rel_str, "", "", "", "", "", ""])
            else:
                for d in dets:
                    writer.writerow([
                        rel_str,
                        f"{d['xmin']:.2f}",
                        f"{d['ymin']:.2f}",
                        f"{d['xmax']:.2f}",
                        f"{d['ymax']:.2f}",
                        f"{d['confidence']:.4f}",
                        d["class"],
                    ])

    print(f"Processed {len(images)} image(s). Predictions saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Cat Detector CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info",    help="Print STUDENT.json")
    sub.add_parser("predict", help="Run inference on /data/input")

    args = parser.parse_args()
    if args.command == "info":
        cmd_info(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
