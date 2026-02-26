"""
Minimal training script that uses the skeleton transformer.
Run from language-models/exercises/:  python train_skeleton.py [args]
Reference: ../train_naive.py
"""
import sys
from pathlib import Path
# Allow importing from parent if needed (e.g. for datasets)
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import torch
from datasets import load_dataset
from transformer_skeleton import Transformer, TransformerConfig
from tqdm import tqdm
from transformers import GPT2Tokenizer
import argparse
import time
from pathlib import Path
from torch.utils.data import DataLoader

def collate_batch(batch, tokenizer):
    """Collate function to create batches from TinyStories dataset."""
    texts = [item['text'] for item in batch]
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    return encoded['input_ids'].cuda()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train skeleton transformer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=512, help='Sequence length')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--nlayers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--save', action='store_true', help='Save the model')
    parser.add_argument('--compile', action='store_true', help='Run compiler to optimize perf')
    parser.add_argument('--profile', action='store_true', help='Profile data loading and compute time')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    args = parser.parse_args()

    if args.verbose:
        print('Loading TinyStories dataset...')
    dataset = load_dataset("roneneldan/TinyStories", streaming=True)
    dataset = dataset['train'].shuffle(buffer_size=1000)

    if args.verbose:
        print('Loading GPT-2 tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    B, S, D, nlayers = args.batch_size, args.seq_len, args.hidden_dim, args.nlayers

    if args.verbose:
        print(f'Initializing skeleton model with {nlayers} layers, hidden dim {D}, vocab size {vocab_size}...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TransformerConfig(
        depth=nlayers,
        hidden_dim=D,
        vocab_size=vocab_size,
        max_seq_len=S,
        device=device,
        mtp=False
    )
    model = Transformer(config).to(device)

    if args.compile:
        print('Running torch.compile...')
        model = torch.compile(model)

    if args.verbose:
        print(f'Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters...')
        print(f'Initializing dataloader with batch size {B}, sequence length {S}...')

    dataloader = DataLoader(
        dataset.take(args.steps * args.batch_size),
        batch_size=B,
        collate_fn=lambda batch: collate_batch(batch, tokenizer)
    )

    if args.verbose:
        print('Setting up training...')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    data_time = 0
    compute_time = 0
    dataloader_iter = iter(dataloader)

    for step in range(args.steps):
        if args.profile:
            data_start = time.time()

        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        # TODO (optional): Construct labels for next-token prediction.
        # Labels should be batch shifted right: labels[:, :-1] = batch[:, 1:], last position can be 0 or ignored.
        labels = torch.empty_like(batch)
        labels[:, :-1] = batch[:, 1:]
        labels[:, -1] = 0

        if args.profile:
            data_time += time.time() - data_start
            compute_start = time.time()

        outputs = model(batch)

        B, S, V = outputs.shape
        outputs = outputs.reshape(-1, V).contiguous()
        labels = labels.reshape(-1).contiguous()

        # TODO (optional): Compute cross-entropy loss between outputs and labels.
        loss = loss_fn(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if step % 10 == 0:
            print(f'[Step {step}/{args.steps}]: Train loss {loss.item()} ...')
        opt.step()
        opt.zero_grad(set_to_none=True)

        del outputs, labels, loss

        if args.profile:
            compute_time += time.time() - compute_start

    if args.profile:
        print("\nProfiling results:")
        print(f"Total data loading time: {data_time:.2f}s")
        print(f"Total compute time: {compute_time:.2f}s")
        total = data_time + compute_time
        if total > 0:
            print(f"Data loading percentage: {100 * data_time / total:.1f}%")
            print(f"Compute percentage: {100 * compute_time / total:.1f}%")

    if args.save:
        save_dir = Path("saved_models")
        save_dir.mkdir(exist_ok=True)
        model_name = f'tinystories_skeleton_lr{args.lr}_bs{args.batch_size}_seq{args.seq_len}.pt'
        save_path = save_dir / model_name
        if args.verbose:
            print(f'Saving model to {save_path}...')
        torch.save(model.state_dict(), str(save_path))
