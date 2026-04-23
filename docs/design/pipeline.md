# Pipeline: Search Query → Attention Goal → Answer

## Overview

The search query is the attention goal. It tells the model what to pay
attention to inside the compressed knowledge. No embedding, no reranking,
no retrieval pipeline — the query shapes attention directly.

```
User query
    │
    ▼
┌──────────────────────────────────────────────────┐
│  1. SEARCH — wiki_search (BM25, tantivy)         │
│     query → ranked page slugs + excerpts         │
└──────────────┬───────────────────────────────────┘
               │ top-k slugs
               ▼
┌──────────────────────────────────────────────────┐
│  2. LOAD — wiki_content_read (bulk)              │
│     slugs → full page content (frontmatter+body) │
└──────────────┬───────────────────────────────────┘
               │ pages as text
               ▼
┌──────────────────────────────────────────────────┐
│  3. PACK — build the context window              │
│                                                  │
│     ┌─────────────────────────────────────┐      │
│     │  System prompt                      │      │
│     │  ─────────────────────────────────  │      │
│     │  [page 1 — highest BM25 score]      │      │
│     │  [page 2]                           │      │
│     │  ...                                │      │
│     │  [page n — fits within budget]      │      │
│     │  ─────────────────────────────────  │      │
│     │  User query (= attention goal)      │      │
│     └─────────────────────────────────────┘      │
│                                                  │
│     Budget = max_context × compression_ratio     │
│     With TurboQuant 5x: 32K window → ~160K       │
│     effective tokens in KV cache                 │
└──────────────┬───────────────────────────────────┘
               │ packed prompt
               ▼
┌──────────────────────────────────────────────────┐
│  4. COMPRESS — build TurboQuant KV store         │
│                                                  │
│     Each page is projected into K and V vectors  │
│     using a frozen attention head (no full LLM): │
│     - K = W_k · page_tokens                     │
│     - V = W_v · page_tokens                     │
│                                                  │
│     Then TurboQuant compresses in-place:         │
│     - PolarQuant (rotation + Lloyd-Max idx)      │
│     - QJL residual (sign bits + norm scalar)     │
│                                                  │
│     This is a one-time cost per page version.    │
│     Store compressed KV alongside the page.      │
│     Re-compress only when the page changes.      │
└──────────────┬───────────────────────────────────┘
               │ compressed K, V per page
               ▼
┌──────────────────────────────────────────────────┐
│  5. ATTEND — attention kernel (no LLM)           │
│                                                  │
│     Q = W_q · query_tokens                       │
│                                                  │
│     For each loaded page's compressed K:         │
│       K̃ = decompress(idx, QJL, ‖ε‖₂)           │
│       scores = Q · K̃ᵀ                           │
│                                                  │
│     softmax(scores) → attention weights          │
│     output = weights · decompress(V)             │
│                                                  │
│     This is just matrix math. No generation,     │
│     no autoregressive decoding, no LLM runtime.  │
│     A single attention pass over compressed KV.  │
│                                                  │
│     The attention scores ARE the relevance       │
│     ranking — which pages (and which parts of    │
│     pages) matter for this query.                │
└──────────────┬───────────────────────────────────┘
               │ attention output + per-page scores
               ▼
┌──────────────────────────────────────────────────┐
│  6. RANK + GROUND — results with provenance      │
│                                                  │
│     Aggregate attention scores per page →        │
│     ranked list of relevant pages + passages.    │
│                                                  │
│     Each result carries its wiki:// URI,         │
│     frontmatter (sources, concepts, status),     │
│     and the specific tokens that scored highest. │
│                                                  │
│     No hallucinated citations — the pages were   │
│     literally in the attention window.           │
└──────────────────────────────────────────────────┘
```

## Why this doesn't need an LLM

Attention is Q·Kᵀ → softmax → weighted V. That's three matrix
operations. You need:

- Projection weights W_q, W_k, W_v (frozen, from any pretrained model)
- The TurboQuant decompression kernel
- A softmax

No autoregressive decoding. No token generation. No sampling. No
full model in memory. Just the attention head weights and the
compressed KV store.

The projection weights encode "what counts as similar" — they are
the learned relevance function. A single attention layer from a
pretrained model is a better similarity function than cosine distance
on embeddings, because it was trained on the actual attention task.

If you later want to generate an answer (not just rank), you can
feed the top-ranked pages into an LLM as a second step. But the
retrieval itself is LLM-free.

## The BM25 pre-filter

We still use BM25 (tantivy) as a pre-filter. Why not skip it and load
the entire wiki?

| Wiki size | Strategy |
|-----------|----------|
| < context budget | Load everything. No pre-filter needed. |
| 1–5x context budget | BM25 top-k. Generous k, let attention sort it out. |
| > 5x context budget | BM25 top-k + graph expansion (load linked pages). |
| >> context budget | Hybrid: BM25 pre-filter + vector DB for long tail. |

The pre-filter is coarse and fast. It doesn't need to be perfect —
its job is to get the right pages into the window. Attention does the
fine-grained relevance ranking.

## Graph expansion

After BM25 returns top-k pages, expand via the concept graph:

```
BM25 top-k slugs
    │
    ▼
wiki_suggest (graph neighborhood, 1-2 hops)
    │
    ▼
Add linked pages that fit within remaining budget
```

This catches pages that are semantically related but don't share query
terms. The graph edges (fed-by, depends-on, links-to) encode
relationships that BM25 misses.

## Token budget math

Given:
- Base context window: W tokens
- TurboQuant compression ratio: C (≈ 5x)
- Effective KV budget: W × C tokens
- System prompt overhead: S tokens
- Query overhead: Q tokens
- Available for knowledge: W × C − S − Q

Example (conservative):
- W = 32K, C = 5, S = 500, Q = 200
- Knowledge budget = 160,000 − 700 = ~159K tokens
- At ~1.3 tokens/word, that's ~122K words
- At ~500 words/page, that's ~244 wiki pages in a single context

Example (modern model):
- W = 128K, C = 5, S = 500, Q = 200
- Knowledge budget ≈ 639K tokens ≈ 490K words ≈ 980 pages

## What changes vs. traditional RAG

| Concern | RAG | This pipeline |
|---------|-----|---------------|
| Retrieval model | Separate embedding model | Frozen attention head weights |
| Relevance ranking | Cosine similarity (approximate) | Attention scores (near-lossless dot products) |
| Context fragments | Chunks, often missing context | Full synthesized pages |
| Provenance | Chunk ID → hope it maps back | wiki:// URI, frontmatter metadata |
| Knowledge freshness | Re-embed on update | Re-compress KV on page change |
| Infrastructure | Vector DB + embedding service | tantivy (BM25) + attention kernel + TurboQuant |
| LLM required | Yes (for generation) | No (attention kernel only). Optional for answer generation. |
| Failure mode | Wrong chunk retrieved | Wrong page loaded (but attention can ignore it) |

## Assumptions and risks

- **Need W_q, W_k, W_v weights from a pretrained model.** Any open
  model works — you only need one attention layer, not the full model.
- **TurboQuant compression/decompression kernel must exist.** Check
  vLLM, llama.cpp, or implement standalone from the paper.
- **Near-lossless is not lossless.** Validate ranking quality with and
  without compression on your knowledge base.
- **BM25 pre-filter can miss pages.** Mitigated by graph expansion and
  generous top-k.
- **Per-user query cost.** Q projection is per-query, but the
  compressed KV store is shared across users (read-only). Much better
  concurrency than per-user KV cache in a full LLM.
