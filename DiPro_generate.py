import argparse
import pandas as pd
from generation.script.emb_tokenizer import *
from generation.script.ema import EMA


def generate_in_batches(model, tokenizer, device, num_batches, batch_size, crop_length, num_steps=500):
    generated_sequences = []
    for _ in range(num_batches):
        with torch.no_grad():
            x_T = torch.randn((batch_size, crop_length, model.embedding_dim)).to(device)
            outputs = model(x_T, num_steps=num_steps).tolist()
            decoded_sequences = [tokenizer.decode(encoded) for encoded in outputs]
            generated_sequences.extend(decoded_sequences)
    return generated_sequences


def main():
    parser = argparse.ArgumentParser(description="Generator")
    parser.add_argument('--checkpoint', default='generation/model/DiPro', help="Path to checkpoint")
    parser.add_argument('--output_csv_path', default='generation/generated_sequences.csv', help="Output CSV path")
    parser.add_argument('--spm_model', default='generation/script/m.model', help="Path to SentencePiece model")
    parser.add_argument('--num_examples', type=int, default=500, help="Number of examples to generate")
    parser.add_argument('--crop_length', type=int, default=51, help="Token sequence length")
    parser.add_argument('--batch_size', type=int, default=100, help="Batch size for generation")
    parser.add_argument('--num_steps', type=int, default=500, help="Number of diffusion sampling steps")
    parser.add_argument('--use_ema', type=lambda x: (str(x).lower() == 'true'), default=True, help="Use EMA checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SentencePieceTokenizer(args.spm_model, enable_sampling=False)
    model = DiffusionLM(
        num_embeddings=len(tokenizer),
        embedding_dim=512,
        model_dim=512,
        num_layers=12,
        dropout_prob=0.1,
        layerdrop_prob=0.0,
        crop_length=args.crop_length,
        embedding_grad_scale=0.5,
        interpolate_temperature=0.8,
        label_smoothing=0.0
    )
    model.to(device)
    
    checkpoint_path = args.checkpoint + '_ema' if args.use_ema else args.checkpoint
    print(f"Loading Checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    if args.use_ema and 'ema_state_dict' in checkpoint:
        ema = EMA(model)
        ema.load_state_dict(checkpoint['ema_state_dict'])
        ema.apply_shadow()
        print(f"EMA parameters loaded")
    
    model.eval()
    print(f"Starting generation (sampling steps: {args.num_steps})...")

    num_batches = args.num_examples // args.batch_size
    remainder = args.num_examples % args.batch_size
    if remainder > 0:
        num_batches += 1

    generated_sequences = generate_in_batches(model, tokenizer, device, num_batches, args.batch_size,
                                              args.crop_length, num_steps=args.num_steps)

    df = pd.DataFrame({"sequence": generated_sequences[:args.num_examples]})
    df.to_csv(args.output_csv_path, index=False, encoding='utf-8')
    print(f"Successfully generated {len(generated_sequences[:args.num_examples])} sequences")
    print(f"Saved to: {args.output_csv_path}")


if __name__ == "__main__":
    main()
