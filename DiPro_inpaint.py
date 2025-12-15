import argparse
import pandas as pd
import torch
from tqdm import tqdm
from generation.script.emb_tokenizer import SentencePieceTokenizer
from generation.script.DiffusionLM import DiffusionLM
from generation.script.ema import EMA


def parse_mask_string(mask_str):
    """Parse mask string. Supports: "0,5-10,15", "NNNNATCGNNN", "111100001111" """
    mask_str = mask_str.strip()
    
    if ',' in mask_str or '-' in mask_str:
        positions = set()
        parts = mask_str.split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                positions.update(range(start, end + 1))
            elif part:
                positions.add(int(part))
        return positions
    elif 'N' in mask_str.upper():
        return {i for i, c in enumerate(mask_str.upper()) if c == 'N'}
    else:
        return {i for i, c in enumerate(mask_str) if c == '1'}


def create_mask_tensor(mask_positions, seq_length):
    mask = torch.zeros(seq_length, dtype=torch.bool)
    for pos in mask_positions:
        if 0 <= pos < seq_length:
            mask[pos] = True
    return mask


def char_mask_to_token_mask(sequence: str, char_positions, tokenizer) -> set:
    """Convert character positions to token positions."""
    pieces = tokenizer.sp.EncodeAsPieces(sequence)
    token_positions = set()
    char_idx = 0
    for token_idx, piece in enumerate(pieces):
        piece_text = piece.replace('▁', '')
        piece_len = len(piece_text)
        masked = any((char_idx + k) in char_positions for k in range(piece_len))
        if masked:
            token_positions.add(token_idx)
        char_idx += piece_len
    return token_positions


def inpaint_batch(model, tokenizer, device, partial_seqs, masks, num_steps=500):
    """Batch sequence inpainting."""
    batch_size = len(partial_seqs)
    encoded_seqs = [tokenizer.encode(seq) for seq in partial_seqs]
    max_len = max(len(seq) for seq in encoded_seqs)
    
    padded_seqs = []
    padded_masks = []
    for raw_seq, token_ids, char_mask_pos in zip(partial_seqs, encoded_seqs, masks):
        padded = token_ids + [tokenizer.pad_id] * (max_len - len(token_ids))
        padded_seqs.append(padded)
        
        token_mask_pos = char_mask_to_token_mask(raw_seq, char_mask_pos, tokenizer)
        mask = create_mask_tensor(token_mask_pos, len(token_ids))
        padded_mask = torch.cat([mask, torch.zeros(max_len - len(token_ids), dtype=torch.bool)])
        padded_masks.append(padded_mask)
    
    ids_tensor = torch.tensor(padded_seqs, dtype=torch.long).to(device)
    mask_tensor = torch.stack(padded_masks).to(device)
    
    with torch.no_grad():
        completed_ids = model.inpaint(ids_tensor, mask_tensor, num_steps=num_steps)
    
    completed_seqs = []
    crop_length = 51
    for i, ids in enumerate(completed_ids.tolist()):
        actual_len = crop_length
        for j, token_id in enumerate(ids):
            if token_id == tokenizer.pad_id:
                actual_len = j
                break
        decoded = tokenizer.decode(ids[:actual_len])
        completed_seqs.append(decoded)
    
    return completed_seqs


def main():
    parser = argparse.ArgumentParser(description="DNA Sequence Inpainting")
    parser.add_argument('--checkpoint', default='generation/model/DiPro', help="Model checkpoint path")
    parser.add_argument('--input_csv', default='generation/partial_sequences.csv', help="Input CSV with 'sequence' and 'mask' columns")
    parser.add_argument('--output_csv', default='generation/completed_sequences.csv', help="Output CSV path")
    parser.add_argument('--spm_model', default='generation/script/m.model', help="SentencePiece model path")
    parser.add_argument('--batch_size', type=int, default=50, help="Batch size")
    parser.add_argument('--num_steps', type=int, default=500, help="Diffusion sampling steps")
    parser.add_argument('--use_ema', type=lambda x: (str(x).lower() == 'true'), default=True, help="Use EMA parameters")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    tokenizer = SentencePieceTokenizer(args.spm_model, enable_sampling=False)
    
    print(f"Loading model: {args.checkpoint}")
    model = DiffusionLM(
        num_embeddings=len(tokenizer),
        embedding_dim=512,
        model_dim=512,
        num_layers=12,
        dropout_prob=0.1,
        layerdrop_prob=0.0,
        crop_length=51,
        embedding_grad_scale=0.5,
        interpolate_temperature=0.8,
        label_smoothing=0.0
    )
    model.to(device)
    
    checkpoint_path = args.checkpoint + '_ema' if args.use_ema else args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    if args.use_ema and 'ema_state_dict' in checkpoint:
        ema = EMA(model)
        ema.load_state_dict(checkpoint['ema_state_dict'])
        ema.apply_shadow()
        print("EMA parameters loaded")
    
    model.eval()
    
    print(f"\nReading input: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    
    if 'sequence' not in df.columns or 'mask' not in df.columns:
        print("Error: CSV must contain 'sequence' and 'mask' columns")
        return
    
    partial_seqs = [str(s).lower() for s in df['sequence'].tolist()]
    mask_strs = df['mask'].tolist()
    
    print(f"Total sequences: {len(partial_seqs)}")
    print(f"Sampling steps: {args.num_steps}")
    print(f"Batch size: {args.batch_size}")
    
    print("\nParsing masks...")
    masks = [parse_mask_string(str(mask)) for mask in mask_strs]
    
    print("\nStarting inpainting...")
    all_completed = []
    num_batches = (len(partial_seqs) + args.batch_size - 1) // args.batch_size
    
    for i in tqdm(range(num_batches), desc="Progress"):
        start_idx = i * args.batch_size
        end_idx = min((i + 1) * args.batch_size, len(partial_seqs))
        
        batch_seqs = partial_seqs[start_idx:end_idx]
        batch_masks = masks[start_idx:end_idx]
        
        completed = inpaint_batch(model, tokenizer, device, batch_seqs, batch_masks, args.num_steps)
        all_completed.extend(completed)
    
    output_df = pd.DataFrame({
        'original_sequence': [s.upper() for s in partial_seqs],
        'mask': mask_strs,
        'completed_sequence': [s.upper() for s in all_completed]
    })
    
    output_df.to_csv(args.output_csv, index=False)
    print(f"\nInpainting complete!")
    print(f"Results saved to: {args.output_csv}")
    
    print("\nFirst 5 examples:")
    print("=" * 100)
    for i in range(min(5, len(output_df))):
        print(f"\nExample {i+1}:")
        print(f"Original: {output_df.iloc[i]['original_sequence']}")
        print(f"Mask:     {output_df.iloc[i]['mask']}")
        print(f"Completed: {output_df.iloc[i]['completed_sequence']}")
        
        orig = output_df.iloc[i]['original_sequence']
        comp = output_df.iloc[i]['completed_sequence']
        mask_pos = masks[i]
        highlighted = ""
        for j in range(min(len(orig), len(comp))):
            c = comp[j] if j < len(comp) else '-'
            if j in mask_pos:
                highlighted += f"[{c}]"
            else:
                highlighted += c
        print(f"Highlighted: {highlighted} ([] = inpainted)")
    print("=" * 100)


if __name__ == "__main__":
    main()
