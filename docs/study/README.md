# TurboQuant Study

## Context

Source: [KV Cache Is Eating Your VRAM. Here's How Google Fixed It With TurboQuant](https://towardsdatascience.com/kv-cache-is-eating-your-vram-heres-how-google-fixed-it-with-turboquant/)
Paper: Zandieh et al. (2025). *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.* arXiv:2504.19874

## What TurboQuant Is

A two-stage KV cache compression framework that achieves 4.5–5x compression
(3.5–2.5 bits per channel) with near-zero accuracy loss. Proven to sit at
the theoretical optimum for its bit budget.

### Stage 1 — PolarQuant

1. **Randomized rotation** (y = R·x, R orthogonal) spreads outlier energy
   across all dimensions → isotropic distribution. Each coordinate follows
   Beta(1/2, (d−1)/2).
2. **Lloyd-Max quantization** with a precomputed codebook (depends only on
   head dimension d and bit-width b). Stores indexes (b−1 bits per dim).
3. Dequantize: codebook lookup → inverse rotation → K̂.
4. Residual: ε = K − K̂.

### Stage 2 — Residual Correction (QJL)

1. Random projection: sign(ε · S), where S is (d, d) random matrix.
   Johnson-Lindenstrauss lemma guarantees inner-product preservation.
2. Store: sign bits (1 bit per dim) + L2 norm ‖ε‖₂ (one scalar).
3. Reconstruct: K̃_QJL = (√(π/2) / d) · ‖ε‖₂ · Sᵀ · QJL.
4. Final: K̃ = K̂ + K̃_QJL.

### What sits in cache per token

| Component | Size | Purpose |
|-----------|------|---------|
| idx | b−1 bits × d | Lloyd-Max codebook indexes |
| QJL signs | 1 bit × d | Residual direction |
| ‖ε‖₂ | 1 scalar | Residual magnitude |

Total: b bits per dimension + negligible scalar overhead.

## The Idea: TurboQuant Instead of a Vector DB

### The problem with vector DBs

Traditional RAG stores document embeddings in a vector database (FAISS,
Pinecone, Qdrant, etc.), then retrieves top-k similar chunks at query time.
This means:

- A separate service to run, index, and maintain
- Embedding model dependency (choice of model affects retrieval quality)
- Chunking strategy is critical and fragile
- Retrieved chunks are context-free fragments — no accumulated knowledge
- Similarity search is approximate — misses semantic relationships that
  aren't captured by embedding distance

### What TurboQuant enables

TurboQuant's core insight is: **don't reconstruct the vector perfectly —
reconstruct what the attention mechanism actually needs** (the dot product).
This is directly relevant beyond KV cache:

1. **Longer context windows at fixed VRAM** — with 4.5–5x compression, a
   model that fits 32K context can now fit 128K+ in the same memory. If the
   knowledge base fits in context, you don't need retrieval at all.

2. **KV cache as the knowledge store** — instead of embedding documents
   into a vector DB and retrieving fragments, ingest the full knowledge
   into the wiki (llm-wiki DKR pattern), load relevant pages into context,
   and let TurboQuant-compressed KV cache hold it all. The attention
   mechanism does the "retrieval" natively — it already knows which tokens
   matter for the current query.

3. **No embedding model dependency** — the model's own attention weights
   determine relevance, not a separate embedding model that may have
   different semantic understanding.

4. **Accumulated knowledge vs. fragments** — the DKR pattern (process at
   ingest time, not query time) means the context contains synthesized
   knowledge pages, not raw chunks. TurboQuant makes it feasible to hold
   more of this synthesized knowledge in a single context window.

### The combined architecture

```
Source → llm-wiki ingest → synthesized wiki pages (DKR)
                                    ↓
                        Load relevant pages into context
                                    ↓
                    TurboQuant-compressed KV cache holds it all
                                    ↓
                    Attention does native "retrieval" over knowledge
```

vs. traditional RAG:

```
Source → chunk → embed → vector DB
                            ↓
Query → embed → similarity search → top-k chunks → context
                                                      ↓
                                              LLM generates answer
```

### When this works

- Knowledge base fits in extended context window (128K–1M tokens)
- Knowledge is pre-processed and synthesized (not raw documents)
- Single-user or low-concurrency scenarios (each session has its own KV cache)
- Quality matters more than latency (no retrieval step, but longer context)

### When vector DB still wins

- Knowledge base exceeds any feasible context window (millions of documents)
- High concurrency — KV cache per user doesn't scale the same way
- Need for exact citation / provenance at the chunk level
- Latency-sensitive with very large knowledge bases

## Key Insight

The fundamental shift is: **move intelligence from retrieval to attention**.
Instead of building a smart retrieval pipeline (embeddings, chunking,
reranking), build a smart knowledge base (DKR) and let the model's own
attention mechanism — now affordable via TurboQuant compression — do the
work of finding what's relevant.

This aligns with the llm-wiki philosophy: process at ingest time, not query
time. TurboQuant makes the "hold it all in context" part practical.

## Open Questions

- What is the actual VRAM budget for TurboQuant-compressed KV cache at
  128K tokens on consumer hardware (e.g. 24GB GPU)?
- How does TurboQuant interact with GQA (which most modern models already
  use)? Are the compression ratios additive?
- Is there a TurboQuant implementation available for vLLM or llama.cpp?
- At what knowledge base size does the "just put it in context" approach
  break down even with 5x compression?
- Could a hybrid approach work: TurboQuant for the "hot" knowledge in
  context, vector DB for the long tail?

## References

1. Zandieh et al. (2025). TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate. arXiv:2504.19874
2. Zandieh et al. (2024). QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead. AAAI 2025
3. Vaswani et al. (2017). Attention Is All You Need. NeurIPS 2017
4. Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP 2023
