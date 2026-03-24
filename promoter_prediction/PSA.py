import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

import PS_modules
import PA_modules
import PR_modules


EXPECTED_SEQ_LEN = 50


class denovo_seq():
    def __init__(self, seq):
        self.seq = seq

    def denovo(self):
        input_seq = self.seq
        code_dic = ['A', 'T', 'C', 'G']
        padding_site = random.randint(1, 9)
        padding = random.choice(code_dic) + random.choice(code_dic) + random.choice(code_dic) + random.choice(
            code_dic) + random.choice(code_dic) + random.choice(code_dic) + random.choice(code_dic) + random.choice(
            code_dic) + random.choice(code_dic)
        input_seq = padding[0:padding_site] + input_seq + padding[padding_site:9]

        for i in range(random.randint(1, 10)):
            mut_site = random.randint(1, 50)
            input_seq = input_seq[0:(mut_site - 1)] + random.choice(code_dic) + input_seq[mut_site:50]
        return input_seq


## using promoter J23119 as input, which contains 41 bp, the promoter is extended to 50 bp using random padding.
def create_seq(create_num):
    rand_seq = denovo_seq('AATTCTTGACAGCTAGCTCAGTCCTAGGTATAATGCTAGCA')
    all_seqs = []
    for i in range(create_num):
        all_seqs.append(rand_seq.denovo())
    all_seqs = np.array(all_seqs)
    all_seqs = all_seqs.reshape([len(all_seqs), ])
    np.save('denovo_seq%s.npy' % create_num, all_seqs)


def encode(seq):
    encoded_seq = np.zeros(len(seq) * 4, int)
    for j in range(len(seq)):
        if seq[j] == 'A' or seq[j] == 'a':
            encoded_seq[j * 4] = 1
            encoded_seq[j * 4 + 1] = 0
            encoded_seq[j * 4 + 2] = 0
            encoded_seq[j * 4 + 3] = 0

        elif seq[j] == 'C' or seq[j] == 'c':
            encoded_seq[j * 4] = 0
            encoded_seq[j * 4 + 1] = 1
            encoded_seq[j * 4 + 2] = 0
            encoded_seq[j * 4 + 3] = 0

        elif seq[j] == 'G' or seq[j] == 'g':
            encoded_seq[j * 4] = 0
            encoded_seq[j * 4 + 1] = 0
            encoded_seq[j * 4 + 2] = 1
            encoded_seq[j * 4 + 3] = 0

        elif seq[j] == 'T' or seq[j] == 't':
            encoded_seq[j * 4] = 0
            encoded_seq[j * 4 + 1] = 0
            encoded_seq[j * 4 + 2] = 0
            encoded_seq[j * 4 + 3] = 1

        else:
            encoded_seq[j * 4] = 0
            encoded_seq[j * 4 + 1] = 0
            encoded_seq[j * 4 + 2] = 0
            encoded_seq[j * 4 + 3] = 0
    encoded_seq = encoded_seq.reshape(len(seq), 4)
    return encoded_seq


def normalize_sequence(seq, target_len=EXPECTED_SEQ_LEN):
    seq = str(seq).strip().upper()
    seq = ''.join(ch for ch in seq if ch in {'A', 'C', 'G', 'T', 'N'})
    if len(seq) >= target_len:
        return seq[:target_len]
    return seq + ('N' * (target_len - len(seq)))


def get_input(seqs, seq_lenth):
    data = np.zeros((1, seq_lenth, 4))
    count = 0
    for i in seqs:
        count += 1
        if count == 1:
            single_seq = encode(i)
            single_seq = np.expand_dims(single_seq, axis=0)
            data = data + single_seq
            data = np.expand_dims(data, axis=0)
        if count != 1:
            single_seq = encode(i)
            single_seq = np.expand_dims(single_seq, axis=0)
            single_seq = np.expand_dims(single_seq, axis=0)
            data = np.concatenate((data, single_seq), axis=0)
    return data


def load_sequences_from_file(input_file):
    """
    Supports:
    1) CSV with `sequence` or `Sequence` column (DiPro output compatible)
    2) TXT/FASTA-like file: one sequence per line (header lines starting with '>' are ignored)
    3) Legacy alternating-line format (kept compatible)
    """
    file_path = Path(input_file)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
        if 'sequence' in df.columns:
            raw_seqs = df['sequence'].tolist()
        elif 'Sequence' in df.columns:
            raw_seqs = df['Sequence'].tolist()
        else:
            raise ValueError("CSV must contain a `sequence` or `Sequence` column.")
        return [normalize_sequence(s) for s in raw_seqs]

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    non_header_lines = [line for line in lines if not line.startswith('>')]

    # If legacy alternating lines are detected, preserve old behavior.
    legacy_like = len(non_header_lines) >= 2 and all(len(non_header_lines[i]) > 50 for i in range(0, len(non_header_lines), 2))
    if legacy_like:
        seqs = [non_header_lines[i] for i in range(1, len(non_header_lines), 2)]
    else:
        seqs = non_header_lines

    return [normalize_sequence(s) for s in seqs]


def load_models(device):
    base_dir = Path(__file__).resolve().parent
    model_classify = base_dir / 'PS_best_model.pth'
    model_strength = base_dir / 'PA_best_model.pth'
    model_r = base_dir / 'PR_best.pth'

    promoS_cp = torch.load(model_classify, map_location=device)
    promoA_cp = torch.load(model_strength, map_location=device)
    promoR_cp = torch.load(model_r, map_location=device)

    promoS = PS_modules.predictor().to(device)
    promoA = PA_modules.predictor().to(device)
    promoR = PR_modules.predictor().to(device)

    promoS.load_state_dict(promoS_cp['pred'])
    promoA.load_state_dict(promoA_cp['pred'])
    promoR.load_state_dict(promoR_cp['pred'])
    return promoS, promoA, promoR

##for generate sample
def predict(num, device='cpu'):
    ##generate data
    create_seq(create_num=int("%s"%num))
    all_seqs = np.load('denovo_seq%s.npy'%num)
    seqs = []
    seq_lenth = EXPECTED_SEQ_LEN
    for i in range(len(all_seqs)):
        seqs.append(all_seqs[i])
    ##load model
    seqs = [normalize_sequence(s) for s in seqs]
    device = torch.device('%s' % device)
    promoS, promoA, promoR = load_models(device)

    ##predict
    f = open('result_generated.txt', 'w', encoding='utf-8')
    test_seq = get_input(seqs=seqs, seq_lenth=seq_lenth)
    c = 0
    for i in test_seq:
        c += 1
        i = i.reshape(1, 1, 4, 50)
        i = torch.from_numpy(i)
        i = i.to(device)
        predict_promoS = promoS(i, 1)
        predict_promoA = promoA(i, 1)
        predict_promoR = promoR(i, 1)
        predict_promoS = predict_promoS.cpu().detach().numpy()
        predict_promoA = predict_promoA.cpu().detach().numpy()
        predict_promoR = predict_promoR.cpu().detach().numpy()
        predict_promoS = predict_promoS.reshape(1, )
        predict_promoA = predict_promoA.reshape(1, )
        predict_promoR = predict_promoR.reshape(1, )

        for j, k, l in zip(predict_promoR, predict_promoS, predict_promoA):
            f.write('%s %s %s %s\n' % (str(np.round(j)), str(np.round(k)), str(l), seqs[c - 1]))
    f.close()

##for input_sample
def predict_in(input_file='sample.txt', device='cpu', output_file=None):
    seqs = load_sequences_from_file(input_file)
    seq_lenth = EXPECTED_SEQ_LEN

    device = torch.device('%s' % device)
    promoS, promoA, promoR = load_models(device)

    ##predict
    if output_file is None:
        output_path = Path(input_file).with_name(f"result_{Path(input_file).name}.txt")
    else:
        output_path = Path(output_file)

    f = open(output_path, 'w', encoding='utf-8')
    test_seq = get_input(seqs=seqs, seq_lenth=seq_lenth)
    c = 0
    for i in test_seq:
        c += 1
        i = i.reshape(1, 1, 4, 50)
        i = torch.from_numpy(i)
        i = i.to(device)
        predict_promoS = promoS(i, 1)
        predict_promoA = promoA(i, 1)
        predict_promoR = promoR(i, 1)
        predict_promoS = predict_promoS.cpu().detach().numpy()
        predict_promoA = predict_promoA.cpu().detach().numpy()
        predict_promoR = predict_promoR.cpu().detach().numpy()
        predict_promoS = predict_promoS.reshape(1, )
        predict_promoA = predict_promoA.reshape(1, )
        predict_promoR = predict_promoR.reshape(1, )

        for j, k, l in zip(predict_promoR, predict_promoS, predict_promoA):
            f.write('%s %s %s %s\n' % (str(np.round(j)), str(np.round(k)), str(l), seqs[c - 1]))
    f.close()
    print(f"Saved prediction results to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sample_type', dest='sample_type', type=str, default='input_sample')
    parser.add_argument('-input_file', dest='input_file', type=str, default='sample.txt')
    parser.add_argument('-device', dest='device', type=str, default='cpu')
    parser.add_argument('-number_of_created_promoter', dest='create_num', type=int, default=1000)
    parser.add_argument('-output_file', dest='output_file', type=str, default=None)
    args = parser.parse_args()
    sample_type = "%s" % args.sample_type
    if sample_type == "input_sample":
        predict_in(input_file="%s" % args.input_file, device="%s" % args.device, output_file=args.output_file)
    if sample_type == "generate_sample":
        predict(num=int("%s" % args.create_num), device="%s" % args.device)