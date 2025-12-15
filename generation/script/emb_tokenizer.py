from .DiffusionLM import *
import torch
import sentencepiece as spm
from typing import List
import random


class SentencePieceTokenizer:
    def __init__(self, model_file: str, enable_sampling=True, alpha=0.15, nbest_size=10):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_file)
        self.enable_sampling = enable_sampling
        self.alpha = alpha
        self.nbest_size = nbest_size

    def __len__(self):
        return len(self.sp)

    @property
    def eos_id(self):
        return self.sp.eos_id()

    @property
    def pad_id(self):
        return self.sp.pad_id()

    def encode(self, text):
        return self.sp.Encode(text, enable_sampling=self.enable_sampling, 
                             alpha=self.alpha, nbest_size=self.nbest_size)

    def decode(self, encoded):
        return self.sp.Decode(encoded)


def reverse_complement(sequence: str) -> str:
    """DNA reverse complement."""
    complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 
                      'a': 't', 't': 'a', 'c': 'g', 'g': 'c',
                      'N': 'N', 'n': 'n'}
    return ''.join(complement_map.get(base, base) for base in reversed(sequence))


def augment_dna_sequence(sequence: str, augmentation_prob: float = 0.5) -> str:
    """DNA sequence augmentation with reverse complement and noise."""
    if random.random() < augmentation_prob:
        sequence = reverse_complement(sequence)
    
    if random.random() < 0.05 and len(sequence) > 10:
        pos = random.randint(0, len(sequence) - 1)
        bases = ['A', 'T', 'C', 'G']
        original = sequence[pos].upper()
        if original in bases:
            bases.remove(original)
            sequence = sequence[:pos] + random.choice(bases).lower() + sequence[pos+1:]
    
    return sequence


def get_line_offsets(path: str, chunk_size: int = 2 ** 20) -> List[int]:
    offsets = [0]
    with open(path, "rb") as file:
        chunk = file.readlines(chunk_size)
        while chunk:
            for line in chunk:
                offsets.append(offsets[-1] + len(line))
            print(f"Lines found: {len(offsets)}", end='\r')
            chunk = file.readlines(chunk_size)
    return offsets[:-1]


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, tokenizer: SentencePieceTokenizer, augment_factor: int = 1, 
                 use_dna_augmentation: bool = True, reverse_complement_prob: float = 0.5):
        self.path = path
        self.tokenizer = tokenizer
        self.augment_factor = augment_factor
        self.use_dna_augmentation = use_dna_augmentation
        self.reverse_complement_prob = reverse_complement_prob
        self.is_csv = path.endswith('.csv')
        if self.is_csv:
            import pandas as pd
            self.data = pd.read_csv(path)
            self.real_size = len(self.data)
        else:
            self.offsets = get_line_offsets(path)
            self.real_size = len(self.offsets)

    def __len__(self) -> int:
        return self.real_size * self.augment_factor

    def __getitem__(self, idx: int):
        real_idx = idx % self.real_size
        if self.is_csv:
            sequence = self.data.iloc[real_idx]['Sequence']
        else:
            with open(self.path, 'r', encoding='utf-8') as file:
                file.seek(self.offsets[real_idx])
                sequence = file.readline().strip('\n')
        
        if self.use_dna_augmentation:
            sequence = augment_dna_sequence(sequence, self.reverse_complement_prob)
        
        ids = self.tokenizer.encode(sequence)
        return ids


class Collate:
    def __init__(self, crop_length=-1, eos_id=-1, pad_id=-1, length_includes_pad=False):
        assert not (pad_id < 0 and length_includes_pad)
        self.crop_length = crop_length
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.length_includes_pad = length_includes_pad

    def generate_mask(self, length):
        return [False] * length

    def process_ids(self, ids):
        if self.eos_id >= 0:
            ids.append(self.eos_id)
        if 0 < self.crop_length < len(ids):
            ids = ids[:self.crop_length]
        conditional_mask = self.generate_mask(len(ids))
        return ids, len(ids), conditional_mask

    def __call__(self, batch):
        processed = list(map(self.process_ids, batch))
        ids, lengths, conditional_mask = zip(*processed)
        padded_lengths = [random.randint(length, max(lengths)) for length in lengths]
        lengths = torch.tensor(padded_lengths) if self.length_includes_pad else torch.tensor(lengths)
        ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.int64) for x in ids],
            batch_first=True,
            padding_value=self.pad_id
        )
        conditional_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.bool) for x in conditional_mask],
            batch_first=True,
            padding_value=False
        )
        return ids, lengths, conditional_mask
