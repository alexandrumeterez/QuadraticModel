import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal

import grain
import numpy as np
from array_record.python.array_record_module import ArrayRecordReader
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


@dataclass(frozen=True)
class FineWeb:
    seq_len: int
    vocab_size: int

    @cached_property
    def tokenizer(self):
        path = Path("tokenizers") / f"bpe_{self.vocab_size}.json"
        if path.exists():
            print(f"Loading tokenizer from {path}")
            return Tokenizer.from_file(str(path))
        print(f"Training tokenizer with vocab_size={self.vocab_size}...")
        path.parent.mkdir(exist_ok=True)
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Digits(individual_digits=True),
                pre_tokenizers.ByteLevel(add_prefix_space=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<eot>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )

        def iter_shards():
            for shard in self.shards[:2]:
                reader = ArrayRecordReader(str(shard))
                for _ in range(reader.num_records()):
                    yield reader.read().decode("utf-8", errors="ignore")
                reader.close()

        tokenizer.train_from_iterator(iter_shards(), trainer=trainer)
        tokenizer.save(str(path))
        print(f"Saved {path} ({tokenizer.get_vocab_size()} tokens)")
        return tokenizer

    @cached_property
    def eot_token(self) -> int:
        return self.tokenizer.token_to_id("<eot>")

    @cached_property
    def token_bytes(self) -> np.ndarray:
        id_to_piece = {v: k for k, v in self.tokenizer.get_vocab().items()}
        return np.array(
            [
                len(id_to_piece.get(i, ""))
                for i in range(self.tokenizer.get_vocab_size())
            ],
            dtype=np.int32,
        )

    @cached_property
    def shards(self) -> list[Path]:
        data_dir = Path(os.environ["DATA_DIR"]) / "fineweb_edu_10B_arrayrecord"
        shards = sorted(data_dir.glob("*.arrayrecord"))
        if not shards:
            raise ValueError(f"No .arrayrecord shards found under {data_dir}")
        assert len(shards) == 14, f"Expected 14 shards, found {len(shards)}"
        return shards

    def build(
        self,
        split: Literal["train", "eval"],
        batch_size: int,
        seed: int,
        shuffle: bool,
        repeat: bool,
    ) -> grain.IterDataset:
        shards = self.shards[:-1] if split == "train" else self.shards[-1:]
        tokenizer = self.tokenizer
        token_bytes = self.token_bytes
        eot = self.eot_token

        def tokenize(rec):
            text = rec.decode("utf-8", errors="ignore")
            enc = np.asarray(tokenizer.encode(text).ids, dtype=np.int32)
            tokens = np.empty(enc.size + 1, dtype=np.int32)
            tokens[:-1] = enc
            tokens[-1] = eot
            return {"tokens": tokens, "bytes": token_bytes[tokens]}

        source = grain.sources.ArrayRecordDataSource(shards)
        ds = grain.MapDataset.source(source).seed(seed)
        if shuffle:
            ds = ds.shuffle()
        if repeat:
            ds = ds.repeat()
        ds = ds.map(tokenize).to_iter_dataset()
        ds = grain.experimental.ConcatThenSplitIterDataset(
            ds, length_struct={"tokens": self.seq_len + 1, "bytes": self.seq_len + 1}
        )

        def split_batch(batch):
            tokens, n_bytes = batch["tokens"], batch["bytes"]
            x, y = tokens[:-1], tokens[1:]
            n_bytes = n_bytes[1:]
            return (x, y), n_bytes

        ds = ds.map(split_batch)
        ds = ds.batch(batch_size, drop_remainder=True)
        return ds
