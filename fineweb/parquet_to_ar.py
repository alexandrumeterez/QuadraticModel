import os
from pathlib import Path

import pyarrow.parquet as pq
from array_record.python import array_record_module
from tqdm import tqdm


def convert(args):
    i, p, output_dir = args
    out_path = output_dir / f"fineweb_edu_100B_{i:05d}.arrayrecord"
    if out_path.exists():
        return
    texts = pq.read_table(p, columns=["text"])["text"].to_pylist()
    writer = array_record_module.ArrayRecordWriter(str(out_path), "group_size:1")
    for text in tqdm(texts, desc=p.name, position=i + 1, leave=False, mininterval=1.0):
        if text:
            writer.write(text.encode("utf-8"))
    writer.close()


def main():
    data_dir = Path(os.environ["DATA_DIR"])
    parquet_dir = data_dir / "fineweb_edu_100B_parquet" / "sample" / "100BT"
    output_dir = data_dir / "fineweb_edu_100B_arrayrecord"
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found under {parquet_dir}")
    args = [(i, p, output_dir) for i, p in enumerate(parquet_files)]

    for arg in tqdm(args, desc="parquet → arrayrecord", position=0):
        convert(arg)


if __name__ == "__main__":
    main()
