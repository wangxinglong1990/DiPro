import pandas as pd
import argparse
import random


def create_template(num_samples=10, mask_mode='middle'):
    """Create inpainting template CSV."""
    train_df = pd.read_csv('dataset/wxl.csv')
    sequences = train_df['Sequence'].tolist()[:num_samples]
    
    data = []
    
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        
        if mask_mode == 'middle':
            start = seq_len // 3
            end = 2 * seq_len // 3
            mask_str = f"{start}-{end}"
            partial = seq[:start] + 'N' * (end - start + 1) + seq[end+1:]
        
        elif mask_mode == 'ends':
            mask_len = min(10, seq_len // 4)
            mask_str = f"0-{mask_len-1},{seq_len-mask_len}-{seq_len-1}"
            partial = 'N' * mask_len + seq[mask_len:seq_len-mask_len] + 'N' * mask_len
        
        elif mask_mode == 'random':
            positions = random.sample(range(seq_len), k=int(seq_len * 0.3))
            positions.sort()
            mask_str = ','.join(map(str, positions))
            partial = ''.join(c if i not in positions else 'N' for i, c in enumerate(seq))
        
        elif mask_mode == 'pattern':
            positions = list(range(0, seq_len, 3))
            mask_str = ','.join(map(str, positions))
            partial = ''.join(c if i not in positions else 'N' for i, c in enumerate(seq))
        
        else:
            mask_str = ''
            partial = seq
        
        data.append({
            'sequence': partial,
            'mask': mask_str,
            'original': seq
        })
    
    df = pd.DataFrame(data)
    return df


def main():
    parser = argparse.ArgumentParser(description='Create inpainting template')
    parser.add_argument('--output', default='generation/partial_sequences.csv', help='Output file path')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--mask_mode', default='middle', choices=['middle', 'ends', 'random', 'pattern'], help='Mask mode')
    args = parser.parse_args()
    
    print(f"Creating inpainting template...")
    print(f"  - Samples: {args.num_samples}")
    print(f"  - Mask mode: {args.mask_mode}")
    
    df = create_template(args.num_samples, args.mask_mode)
    
    df.to_csv(args.output, index=False)
    
    output_no_orig = args.output.replace('.csv', '_input.csv')
    df[['sequence', 'mask']].to_csv(output_no_orig, index=False)
    
    print(f"\nTemplate created:")
    print(f"  - Full version (with original): {args.output}")
    print(f"  - Input version (partial + mask): {output_no_orig}")
    
    print(f"\nFirst 3 examples:")
    for i in range(min(3, len(df))):
        print(f"\nExample {i+1}:")
        print(f"  Original: {df.iloc[i]['original']}")
        print(f"  Partial:  {df.iloc[i]['sequence']}")
        print(f"  Mask:     {df.iloc[i]['mask']}")


if __name__ == "__main__":
    main()
