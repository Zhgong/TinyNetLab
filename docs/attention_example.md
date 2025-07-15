# Positional Encoding and Default Attention Example

This short note explains the sinusoidal positional encoding used in `attention_demo.py` and shows the default attention weights for the sentence **"I love machine learning"** when the embedding dimension is 2.

## Sinusoidal positional encoding

For a sequence length `L` and embedding dimension `d`, the encoding is a matrix `PE[L, d]` where:

- **Rows** correspond to token positions (starting at 0).
- **Columns** alternate between sine and cosine functions.

The value at position `pos` and dimension `2i` is `sin(pos / 10000^{2i/d})` and the value at dimension `2i+1` is `cos(pos / 10000^{2i/d})`.

With the default settings (`L = 4`, `d = 2`) the encoding is:

| pos | dim0 (sin) | dim1 (cos) |
| --- | ----------- | ----------- |
| 0 | 0.00 | 1.00 |
| 1 | 0.84 | 0.54 |
| 2 | 0.91 | -0.42 |
| 3 | 0.14 | -0.99 |

## Default attention weights

Using random vectors seeded with `0` and identity projection matrices, the attention matrix for **"I love machine learning"** is:

| query\key | I | love | machine | learning |
| --- | --- | --- | --- | --- |
| **I** | 0.25 | 0.26 | 0.23 | 0.26 |
| **love** | 0.20 | 0.26 | 0.16 | 0.38 |
| **machine** | 0.24 | 0.21 | 0.35 | 0.20 |
| **learning** | 0.10 | 0.19 | 0.08 | 0.63 |

Each row sums to one. For example, the `love → learning` weight (~0.38) means that about 38 % of the output for the token **"love"** is drawn from **"learning"**. The final row shows `learning` attending mostly to itself (≈0.63).
