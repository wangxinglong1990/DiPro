import sentencepiece as spm
import pandas as pd

# Extract sequences from CSV and save to text file
df = pd.read_csv('dataset/wxl_categorized.csv')
with open('dataset/promoter_sequences.txt', 'w', encoding='utf-8') as f:
    for seq in df['Sequence']:
        f.write(seq + '\n')

# Train SentencePiece model
spm.SentencePieceTrainer.Train(input='dataset/promoter_sequences.txt', model_prefix='m',
                               model_type="bpe", vocab_size=9, pad_id=3)
