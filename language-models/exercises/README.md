# Language-models exercises

Educational skeletons for the decoder-only transformer and related notebooks. Implement the TODOs to learn each component; reference implementations are in the parent folder and remain unchanged.

## Purpose

- **transformer_skeleton.py** and **train_skeleton.py**: Learn each part of the transformer (embedding, positional encoding, layer norm, MLP, attention, full forward) by filling in the TODOs.
- **Notebook skeletons**: Learn BPE, KV caching, RoPE, and speculative decoding by implementing the marked cells in each notebook.

## Suggested order

### Transformer and training

1. **EmbeddingLayer** – lookup `embedding[x]`.
2. **PositionalEmbedding** – add `pos_embedding[:S]` to the sequence.
3. **LN** – mean, std, normalize, scale and shift.
4. **MLP** – up_proj → act → down_proj.
5. **Attention** – Q/K/V, scale, causal mask, softmax, attention @ V, output projection.
6. **TransformerLayer** – pre-norm + attention + residual; pre-norm + MLP + residual.
7. **Transformer** – full forward: emb → pos_emb → layers → unemb (with kv_cache handling when set).
8. Run **train_skeleton.py** (e.g. `python train_skeleton.py --steps 10 --verbose`).

### Notebooks

1. **bpe_skeleton.ipynb** – tokenization (merge, get_stats, train, encode, decode).
2. **KV_cache_skeleton.ipynb** – inference caching (cache update, use in attention).
3. **rope_skeleton.ipynb** – rotary position embeddings (sinusoidal, RoPE on Q/K).
4. **speculative_decoding_skeleton.ipynb** – draft + target verification and accept/reject.

## How to verify

- **Transformer**: Run the reference and skeleton on the same input and compare shapes; after implementing, compare outputs (e.g. small script that loads both `transformer` and `transformer_skeleton` and checks forward pass).
- **Training**: Once the skeleton forward passes are correct, training loss should decrease similarly to the reference.
- **Notebooks**: Compare your implementations to the reference notebooks (`../bpe.ipynb`, `../KV_cache.ipynb`, `../rope.ipynb`, `../speculative_decoding.ipynb`).

## Reference files

- [../transformer.py](../transformer.py) – full transformer implementation.
- [../train_naive.py](../train_naive.py) – full training script.
- [../bpe.ipynb](../bpe.ipynb), [../KV_cache.ipynb](../KV_cache.ipynb), [../rope.ipynb](../rope.ipynb), [../speculative_decoding.ipynb](../speculative_decoding.ipynb) – reference notebooks.

## Running the training skeleton

From this directory (`language-models/exercises/`):

```bash
python train_skeleton.py --steps 10 --verbose
```

Until the transformer_skeleton forward methods are implemented, outputs will be zeros and loss will not train meaningfully; implement the TODOs in `transformer_skeleton.py` first.
