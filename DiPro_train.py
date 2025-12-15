import argparse

from torch.utils.data import DataLoader
from tqdm import tqdm

from generation.script.utils import *
from generation.script.emb_tokenizer import SentencePieceTokenizer, TextDataset, Collate
from generation.script.DiffusionLM import DiffusionLM
from generation.script.ema import EMA


def main():
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument('--checkpoint', default='generation/model/DiPro', help="Path to save/load checkpoints")
    parser.add_argument('--data_path', default='dataset/wxl.csv', help="Path to the dataset")
    parser.add_argument('--spm_model', default='generation/script/m.model', help="Path to SentencePiece model")
    parser.add_argument('--num_examples', type=int, default=5, help="Number of examples to generate during training")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--crop_length', type=int, default=51, help="The number of tokens in the generated sequence")
    parser.add_argument('--epochs', type=int, default=500, help="The number of training cycles")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument('--save_every', type=int, default=200, help="Save checkpoint every N steps")
    parser.add_argument('--use_ema', type=lambda x: (str(x).lower() == 'true'), default=True, help="Use EMA for model parameters")
    parser.add_argument('--ema_decay', type=float, default=0.9999, help="EMA decay rate")
    parser.add_argument('--augment_factor', type=int, default=3, help="Data augmentation factor")
    parser.add_argument('--use_dna_augmentation', type=lambda x: (str(x).lower() == 'true'), default=True, help="Use DNA-specific augmentation")
    parser.add_argument('--reverse_complement_prob', type=float, default=0.5, help="Probability of applying reverse complement")
    parser.add_argument('--embedding_grad_scale', type=float, default=0.5, help="Embedding gradient scale")
    parser.add_argument('--interpolate_temperature', type=float, default=0.8, help="Temperature for interpolation softmax")
    parser.add_argument('--label_smoothing', type=float, default=0.1, help="Label smoothing for cross entropy loss")
    parser.add_argument('--learning_rate', type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument('--warmup_steps', type=int, default=2000, help="Learning rate warmup steps")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SentencePieceTokenizer(args.spm_model)
    
    print(f"=" * 60)
    print(f"Training Config:")
    print(f"  Dataset samples: ~11886")
    print(f"  Augment factor: {args.augment_factor}x")
    print(f"  Effective samples: ~{11886 * args.augment_factor}")
    if args.use_dna_augmentation:
        print(f"  DNA augmentation enabled:")
        print(f"    - Reverse complement prob: {args.reverse_complement_prob}")
        print(f"    - Sequencing error simulation: 5%")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"\nOptimization:")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Embedding grad scale: {args.embedding_grad_scale}")
    print(f"  Interpolate temperature: {args.interpolate_temperature}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"=" * 60)
    
    model = DiffusionLM(
        num_embeddings=len(tokenizer),
        embedding_dim=512,
        model_dim=512,
        num_layers=12,
        dropout_prob=0.1,
        layerdrop_prob=0.0,
        crop_length=args.crop_length,
        embedding_grad_scale=args.embedding_grad_scale,
        interpolate_temperature=args.interpolate_temperature,
        label_smoothing=args.label_smoothing
    )
    model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=0.01,
        eps=1e-8
    )
    
    global_step = load_checkpoint(model, optimizer, args.checkpoint)
    
    ema = None
    if args.use_ema:
        ema = EMA(model, decay=args.ema_decay)
        print(f"\nEMA enabled, decay={args.ema_decay}")
    else:
        print(f"\nEMA disabled (not recommended)")
    
    lr_lambda = lambda step: linear_decay_with_warmup(
        step, 
        max_learning_rate=args.learning_rate, 
        warmup_steps=args.warmup_steps, 
        hold_steps=10000,
        decay_steps=150000, 
        min_learning_rate=1e-6
    )
    
    data_loader = DataLoader(
        TextDataset(
            path=args.data_path, 
            tokenizer=tokenizer, 
            augment_factor=args.augment_factor,
            use_dna_augmentation=args.use_dna_augmentation,
            reverse_complement_prob=args.reverse_complement_prob
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=Collate(
            crop_length=args.crop_length,
            eos_id=tokenizer.eos_id,
            pad_id=tokenizer.pad_id,
            length_includes_pad=True
        )
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
    for ep in range(args.epochs):
        model.train()
        pbar = tqdm(data_loader, desc=f"Epoch: {ep}")
        optimizer.zero_grad()
        
        for idx, (ids, lengths, conditional_mask) in enumerate(pbar):
            ids, lengths, conditional_mask = ids.to(device), lengths.to(device), conditional_mask.to(device)
            
            with torch.amp.autocast('cuda'):
                loss, loss_diff, loss_reconstruction, accuracy = model.compute_loss(ids, lengths, conditional_mask)
                loss = loss / args.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (idx + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.param_groups[0]['lr'] = lr_lambda(global_step)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if ema is not None:
                    ema.update()
                
                global_step += 1
            
            pbar.set_postfix({
                "loss": (loss.item() * args.gradient_accumulation_steps),
                "mse": loss_diff.item(),
                "ce": loss_reconstruction.item(),
                "rec_acc": accuracy.item(),
                "lr": optimizer.param_groups[0]['lr'],
                "step": global_step
            })
            
            if global_step % args.save_every == 0:
                save_checkpoint(model, optimizer, global_step, args.checkpoint)
                if ema is not None:
                    checkpoint_ema = {
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ema_state_dict': ema.state_dict()
                    }
                    torch.save(checkpoint_ema, args.checkpoint + '_ema')
        
        model.eval()
        if ema is not None:
            ema.apply_shadow()
        
        with torch.no_grad():
            x_T = torch.randn((args.num_examples, args.crop_length, model.embedding_dim)).to(device)
            outputs = model(x_T, num_steps=200).tolist()
            print(f"\nEpoch {ep} generated samples:")
            [print(tokenizer.decode(encoded)) for encoded in outputs]
        
        if ema is not None:
            ema.restore()


if __name__ == "__main__":
    main()
