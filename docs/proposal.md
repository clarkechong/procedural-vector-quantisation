# Research proposal and handover: Hardware-efficient computational codebooks and procedural neural network weights

**Date:** 26 November 2026  
**Status:** Working research proposal; preliminary numerical evidence, not a validated inference system  
**Initial model:** SmolLM2-135M (BF16 checkpoint)  
**Target architecture:** AMD Instinct MI300X / CDNA3 (HIP/ROCm, gfx942)

## 0. Purpose and current position

This document is intended to let a new human researcher or AI agent continue the project without needing the original conversations. It records the research motivation, the originating intuitions and how they evolved, the mathematical formulation, the experiments reported so far, the limitations already identified, and the proposed implementation and evaluation plan. It is deliberately more detailed than a conventional abstract or short proposal.

**Overarching objective:** Investigate whether neural network weights can be represented by automatically synthesised, hardware-efficient generating programs shared across many weight tiles, together with compact tile-specific parameters. Co-design the generator and its encoder under explicit information and execution budgets, exploiting algebraic structure to guide search, and eventually investigate training neural networks directly within such representations.

There are two connected, but independently valuable, research targets:

| | Practical target | Ambitious target |
|---|---|---|
| Representation | Computational vector quantisation (VQ) | Large-scale procedural weight generation |
| Bitrate | Competitive approximately 2 bits per weight (bpw) | Sub-1 bpw, potentially a small fraction of a bit per effective weight |
| Learning | Post-training compression first; calibration/fine-tuning if helpful | Compression-aware or end-to-end training |
| Primary question | Can a synthesised decoder achieve competitive quality and GPU performance? | Can a small amount of learned information support much more useful effective computation? |
| Status | Initial weight reconstruction demonstrated | Hypothesis, not demonstrated |

A 2-bit codec is not unprecedented: QuIP, QuIP#, AQLM, QTIP and Proteus are important competitors. A credible contribution would be a competitive **automatically synthesised** computational codebook, especially one whose encoder is mathematically structured and whose decoder is efficient on AMD GPUs. Do not claim novelty on the basis of bitrate alone. Do not discard this practical target if the more ambitious procedural representation fails.

The next immediate work is **target point-cloud analysis, rate–distortion references, and algebraic analysis of small generators**, with model-quality evaluation and GPU feasibility measurements proceeding in parallel. Avoid launching another very large brute-force search before these checks.

## 1. Where the research came from

### 1.1 GPU bottleneck: delivering useful operands, not just counting FLOPs

The wider research programme concerns exploiting sparsity and quantisation on modern accelerators. Previous sparse FFN and structured-sparsity experiments exposed the difficulty of translating fewer theoretical arithmetic operations into actual GPU speed. Sparse indexing, data preparation, register layout, memory staging and irregular utilisation can prevent a matrix unit from remaining busy.

The earlier sparse MFMA/SMFMA experiments were reported to achieve roughly 28% matrix utilisation in a particular configuration. This motivated investigating the producer side of the pipeline: how quickly and regularly can data be made ready in registers for matrix instructions? Treat the figure and the suspected memory/data-preparation bottleneck as historical observations, not a verified attribution for every kernel. Revisit the original profiles before making precise causal claims.

The key question was whether we could **generate the weights directly in registers** rather than fetch every weight from HBM, stage it in LDS if required, then load and rearrange it into matrix operands. In low-batch autoregressive decoding, weight traffic is particularly expensive because weights are often streamed with relatively little reuse per token. In prefill or large-batch GEMM, reuse increases and the trade-off changes; compute can become the bottleneck.

The intention is not to assume that arithmetic is free, nor that all dense kernels necessarily use LDS. The actual target is a fused pipeline that delivers useful MFMA operands faster or with less memory traffic than the best comparable implementation.

### 1.2 Algorithmic representation as a unifying perspective

Sparsity, quantisation, low rank, repetition and procedural generation can all be viewed as restrictions on how much independent information is required to describe a matrix. Kolmogorov complexity is a useful conceptual umbrella: the shortest program that reproduces a finite object can in principle be shorter than an explicit description. It is not a computable general optimisation objective, and the shortest program may be unusably slow.

A convenient general representation is

$$W_{ij}=G(\theta,i,j)+R_{ij},$$

where $G$ is a deterministic generator, $\theta$ contains stored information, and $R$ is an optional residual. Several generated components could also be superimposed. Every residual, coefficient, index and program constant must be counted when it contains model-specific information.

The practical search is over a restricted family of programs that are small, cheap, locally addressable, GPU-native, and require little working memory. The goal is not arbitrary lossless decompression or discovery of the shortest possible program.

**Important clarification from the originating discussion:** the original ambition was not merely to discover that trained matrices already contain recognizable periodic, low-rank, or pseudorandom patterns. The proposal is to explore many feasible algorithms and their latent spaces, seeking any computational mapping whose outputs happen to lie near the desired weights (or, later, near useful task solutions). A useful coincidence is enough; an interpretable pre-existing pattern is not required. Target geometry analysis is a way to guide that search, not an assumption that hidden patterns must exist.

### 1.3 Why SeedLM was a starting point

[SeedLM (ICLR 2025)](https://arxiv.org/html/2410.10714v2) demonstrates that one can generate a small pseudorandom basis from an LFSR seed and store projection coefficients instead of explicit weights. The paper fixes the basis-generation algorithm, chooses seeds for each block and fits the coefficients. Its principal model evaluations use 3- and 4-bit representations. Its custom FPGA inference results motivate a compute-for-memory trade, but are not evidence of performance on CDNA3.

The project asks: why restrict ourselves to a single LFSR, or even a preselected family of conventional PRNGs? Could a program synthesis engine discover better GPU-specific mappings? Could it jointly discover a way to encode weights into the mapping without exhaustively searching every seed? And could the same principle be pushed to much larger regions than ordinary quantisation blocks?

## 2. The geometric intuition: from PRN sequences to computational VQ

### 2.1 The ten points in three-dimensional space analogy

One of the user's key intuitions was to imagine a weight matrix as ten points in three-dimensional space. A pseudorandom generator, given a particular seed, generates ten of its own points. We search for an algorithm and a seed whose sequence aligns as closely as possible with the original sequence. Correspondence matters: matching an unordered point cloud is not the same as matching the weights in their correct positions.

Equivalently, those ten ordered 3D points constitute one point in 30-dimensional space. A complete seed-generated sequence is another single point in that space. Varying the seed produces a set of candidate points; varying the algorithm changes that set.

This viewpoint is especially useful once a matrix is tiled. For a matrix of shape $4096\times4096$ with nonoverlapping $8\times8$ tiles, there are 262,144 tiles, each a point in $\mathbb R^{64}$. Instead of reconstructing the entire matrix from one PRN sequence, we can seek a **single shared generator whose seed outputs collectively cover the cloud of all tiles**. Each tile gets its own short seed.

### 2.2 Formal definition

Let $N$ denote weights per tile, $T$ the number of tiles sharing a decoder, and $B$ the total tile-specific bit budget (including seed, scale, coefficient, selector and any other metadata, as applicable). Let $w_t\in\mathbb R^N$ be the vectorised tile. A deterministic generator can be written as a coordinate function or its vector counterpart:

$$G(s,i),\qquad g_G(s)=[G(s,0),\ldots,G(s,N-1)],\qquad s\in\{0,1\}^B.$$

A generator defines an implicit codebook / reachable set:

$$\mathcal C_G=\{g_G(s):s\in\{0,1\}^B\}.$$

There are at most $2^B$ distinct codewords, possibly fewer because of collisions. A fixed globally shared scale or fixed matrix-specific constant can be part of $G$, but any tile-varying scale must be part of the budget $B$.

For a fixed decoder, the ideal encoder solves

$$s_t^*=\arg\min_s D(w_t,g_G(s)).$$

The shared generator optimisation is

$$G^*=\arg\min_{G\in\mathcal G_{\rm feasible}}\sum_{t=1}^T \min_s D(w_t,g_G(s)).$$

Initially $D$ is squared reconstruction error; later it should include activation-weighted distortion or model loss. For search heuristics, optimise a calibrated validation objective, never the final held-out evaluation split.

### 2.3 The connection to vector quantisation

This is ordinary vector quantisation (VQ) at the level of the encoding problem, except that the codebook is represented by a program instead of an explicit array.

| Conventional VQ | Computational VQ proposed here |
|---|---|
| Store vectors $c_s$ | Generate vectors $g_G(s)$ from code |
| Learn codeword locations directly | Adjust instruction composition, constants and computational structure |
| Store a codeword index per tile | Store a seed per tile (same role as index) |
| Decode by table access | Decode by executing a shared program |
| Encoding by nearest-neighbour search or structured methods | Same underlying problem, potentially with a generator-specific algebraic solver |

A conventional VQ codebook is a reference for representational quality. It is not necessarily a fair runtime baseline if it requires a large lookup table. Computational VQ is already a research area (especially QTIP and Proteus); the proposed emphasis is **automated program synthesis, mathematical guidance, encoder co-design, and the AMD-specific execution cost**.

### 2.4 Coverage is not the same as span or orthogonality

The intuitive question was whether a generator's effective basis could cover all tile points, perhaps by providing orthogonal directions. Orthogonality is useful for reducing redundancy and may make encoding easier, but it is not sufficient. An orthogonal basis of $N$ vectors spans $\mathbb R^N$ only if arbitrary linear combinations are allowed. Selecting one of those vectors does not make all points reachable. And storing independent coefficients consumes bits.

The relevant error is distance to the actual **reachable codeword set**, not merely distance to its span. Full affine rank, uniform marginal distributions, a high algebraic degree, or a large number of noncolliding seeds are not guarantees of good reconstruction. The points must be distributed in useful places under the relevant distortion metric.

Useful geometric quantities include average nearest-codeword distortion, tail distortion, and covering radius over a bounded target set. A worst-case covering radius over all of unbounded $\mathbb R^N$ would be infinite for a finite codebook and is not a sensible target.

## 3. Reverse the search: fix the seed budget and search programs

The user's next intuition was to reverse the usual PRNG-based approach. Instead of increasing seed length to improve coverage, fix a strict seed budget and make the algorithm the main search variable. If we target 0.25 bpw, for example, a 64-weight tile has only 16 bits of tile-specific information. We should not solve a poor reconstruction result by silently expanding that budget.

Our previous synthesis already used a shared program and fixed seeds, but only a very small program budget. The new direction is to systematically vary program complexity and sharing while holding the tile-specific information constant.

Let $\mathcal G$ denote the feasible program family. Its collective outputs are

$$\mathcal M=\bigcup_{G\in\mathcal G}\mathcal C_G.$$

Even a tiny operation dictionary yields an enormous combinatorial program space. With 11 operation types, there are $11^{100}\approx 1.38\times10^{104}$ syntactic opcode sequences of length 100 alone, ignoring operand choice, constants, arities and dependencies. Many are invalid, redundant or semantically identical. This is effectively inexhaustible, not literally infinite at a fixed finite instruction and word-size budget. If unbounded program lengths are admitted the syntactic space is countably infinite.

A critical distinction: the union across generators is useful for **choosing a shared decoder**, but once one decoder is fixed, each tile can access only that decoder's codebook. The two optimisation orders are not interchangeable:

$$\sum_t\min_{G_t,s_t}D(w_t,g_{G_t}(s_t))\quad\text{vs.}\quad\min_G\sum_t\min_{s_t}D(w_t,g_G(s_t)).$$

The first may fit far better, but requires a way to communicate a different program or selector for each tile. The second amortises the code description. The union of many programs does not confer all their codewords on each tile for free.

## 4. Information theory and the code/data trade-off

### 4.1 Two amortisation factors

The user immediately raised the key objection: perhaps we are simply transferring the information from the seed to the instruction stream. A 64-bit machine instruction occupies the same storage as four FP16 weights. A long unique program for every tile can easily be worse than storing the weights directly.

Let $C$ be the bits required for the shared program and constants, $M$ other shared metadata bits, $B$ tile-local bits, and $T$ the number of $N$-weight tiles sharing it. Then

$$R_{\rm total}=\frac{C+M+TB}{TN}=\frac{C+M}{TN}+\frac BN.$$

There are two storage amortisation factors: **tiles per generator ($T$)** and **weights per seed ($N$)**. In addition, there is a separate computational amortisation problem: executed work per useful output weight. A small code footprint does not imply cheap execution.

Example: assume a 100-instruction generator occupying 8 bytes per instruction, or 6,400 bits, plus a 16-bit descriptor per 64 weights. This is an illustrative code size, not a claim about all AMD instruction encodings.

| Tiles sharing one program | Program bpw | Descriptor bpw | Total bpw (no other metadata) |
|---:|---:|---:|---:|
| 1 | 100 | 0.25 | 100.25 |
| 16 | 6.25 | 0.25 | 6.50 |
| 64 | 1.5625 | 0.25 | 1.8125 |
| 256 | 0.390625 | 0.25 | 0.640625 |
| 1,024 | 0.097656 | 0.25 | 0.347656 |
| 65,536 | 0.001526 | 0.25 | 0.251526 |

For a $4096\times4096$ matrix, an 800-byte shared program is negligible relative to its 32 MiB BF16 representation. This is not proof that the codebook can represent the matrix at acceptable quality. It only demonstrates the amortisation mechanism.

One generator per matrix is a sensible initial sharing granularity; compare one per model, layer, region and a small dictionary. A dictionary of $K$ generators chosen per tile normally requires $\lceil\log_2 K\rceil$ selector bits (or an alternative explicitly counted encoding); those bits reduce the available seed budget. All generator binaries/constants are also counted. If the choice is determined by a fixed coordinate rule, no per-tile selector is needed, but that rule is part of the shared program.

### 4.2 The relationship to SparseRT and compiler specialisation

[SparseRT (PACT 2020)](https://arxiv.org/html/2008.11849v1) specialises GPU computations for known sparse weight patterns and values. [SparGen](https://www.microsoft.com/en-us/research/publication/a-new-approach-to-deep-learning-model-sparsity-via-tensor-with-sparsity-attribute/) and [SparTA (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/zheng-ningxin) provide related context on compiler-based sparse specialisation.

The conceptual spectrum is:

```text
Generic dense kernel       Fixed quantisation decoder       Specialised procedural decoder       Fully weight-specialised code
Explicit weight data  -->  Packed weight data          -->  Shared program + compact seeds  -->  Constants/patterns in program
```

The proposal generalises the idea of hardcoding weights into instructions by introducing adjustable specialisation: the program may encode information common to many tiles, while compact descriptors supply the remaining variation. Unlike exact sparse code generation, our approach may be lossy and may add arithmetic rather than remove zero operations. The desired destination is a matrix-engine-friendly operand stream, not necessarily the same execution strategy as SparseRT. Do not claim that SparseRT itself uses this codec or that our approach is automatically superior.

### 4.3 Fundamental rate–distortion limit

No deterministic program creates independent information. With a fixed shared decoder and $B$ tile bits, there are at most $2^B$ possible reconstructions. For arbitrary binary data or arbitrary high-precision weight vectors, the overwhelming majority cannot be exactly represented at very low bitrates.

For an IID zero-mean Gaussian source with variance $\sigma^2$, the squared-error rate–distortion bound, normalised by the variance, is

$$D_{\rm Gaussian}(R)\geq 2^{-2R},\qquad R=B/N.$$

This is an information-theoretic lower bound on expected distortion for the stated source and rate, not an achievable finite-block guarantee. If the decoder is fitted to the very source samples being encoded, its model-specific description must be counted or the bound interpreted on independent draws. Real neural weights need not satisfy the IID Gaussian assumptions.

For a 16-bit tile budget:

| Weights per tile | Payload bpw | Gaussian distortion lower bound |
|---:|---:|---:|
| 8 | 2 | 0.0625 |
| 16 | 1 | 0.25 |
| 32 | 0.5 | 0.50 |
| 64 | 0.25 | 0.7071 |
| 128 | 0.125 | 0.8409 |
| 256 | 0.0625 | 0.9170 |
| 1,024 | 0.015625 | 0.9786 |

This is a warning about attempting to reconstruct arbitrary pretrained weights with tiny seeds. Increasing the shared generator's complexity may improve an inefficient codebook but cannot beat the applicable information bound. For a large matrix of nearly independent weights, using a handful of bits to specify an entire tile is unlikely to preserve its original values closely.

The ambitious project therefore depends on one or more of: nontrivial source structure; sharing beyond independent tiles; a different distortion objective that better reflects functionality; or training a network whose useful weights lie within a cheap representable family. The last does not violate entropy laws: the task might not require independent control of every effective weight.

## 5. Algebraic framework: what does instruction composition actually buy?

### 5.1 Start with the user's AND generator

The original example was a one-bit seed with an AND operation and an index:

$$G(s,i)=s\mathbin{\&}i=s(i\bmod 2).$$

For seed 0, all outputs are zero. For seed 1, the sequence is `01010101...` (ones at odd indices). For a tile of eight weights, the entire sequence is a scalar multiple of $[0,1,0,1,0,1,0,1]$. There are exactly two reachable points on a line in eight-dimensional space.

For a two-bit seed, $s=s_0+2s_1$, write $i_0,i_1$ for the low index bits. Then

$$G(s,i)=s_0 i_0+2s_1 i_1,$$

$$g_G(s)=s_0[0,1,0,1,0,1,0,1]+s_1[0,0,2,2,0,0,2,2].$$

We have two basis vectors and four reachable points. The sequence necessarily repeats every four positions, regardless of seed. This is a limitation that can be proven without numerical search.

For binary operands $a,b$ the following real-polynomial identities are exact:

| Operation | Real polynomial on binary inputs |
|---|---|
| AND | $ab$ |
| OR | $a+b-ab$ |
| XOR | $a+b-2ab$ |
| NOT | $1-a$ |

Composition can introduce interaction terms. For example,

$$ (s_0\land i_0)\oplus(s_1\land i_1)=s_0i_0+s_1i_1-2s_0s_1i_0i_1.$$

The new term may add another real-valued basis direction, but its coefficient is constrained to the product $s_0s_1$. We cannot choose that coefficient independently of the two seed bits.

### 5.2 Exact polynomial representation and dimension bound

Every deterministic numeric-valued generator on a finite binary seed domain admits a unique real multilinear expansion:

$$g_G(s)=\sum_{S\subseteq\{0,\dots,B-1\}}v_S\prod_{j\in S}s_j=A\phi(s).$$

Here each $v_S\in\mathbb R^N$ is a vector coefficient, and $\phi$ is the vector of square-free monomials including the constant 1. Coefficients can be recovered by subset Möbius inversion on the table of all seed outputs. Equivalently use Walsh characters on the Boolean hypercube; the Fourier/Walsh expansion is useful for variance and interaction-order analysis. These exact expansions may be exponentially large; they are analytical tools, not necessarily the actual decoder.

If the decoded numerical output has real polynomial degree at most $d$, then

$$K(B,d)=\sum_{k=1}^d\binom Bk,\qquad\dim_{\rm aff}(\mathcal C_G)\leq\min(N,K(B,d)).$$

For $B=16$ and $N=1024$:

| Maximum degree | Nonconstant terms | Maximum affine dimension |
|---:|---:|---:|
| 1 | 16 | 16 |
| 2 | 136 | 136 |
| 3 | 696 | 696 |
| 4 | 2,516 | 1,024 |

The user asked whether some combinations of instructions would *never* be expressive enough, rather than merely empirically bad. This bound offers such a statement, provided the degree bound is actually established for the **whole decoded numerical function**. Full affine dimension does not imply dense coverage, good rate–distortion or many independently controllable variables.

For a target distribution of mean $\mu$ and covariance eigenvalues $\lambda_1\geq\cdots\geq\lambda_N$, the optimal affine $K$-dimensional approximation has expected squared error $\sum_{j>K}\lambda_j$. Any generator confined to such an affine space has at least this error, even if we relax its discrete seeds into unrestricted points in that space. Therefore with the pooled-energy NMSE convention:

$$D_{\rm subspace}\geq\frac{\sum_{j>K}\lambda_j}{\mathbb E\|W\|^2}.$$

For an isotropic zero-mean source this reduces to $1-K/N$ for $K\leq N$. With the Gaussian assumptions both bounds apply, so take their maximum. On real data, the empirical covariance-based bound can be evaluated directly (with appropriate uncertainty and split handling). If a proposed low-degree generator cannot meet a specified distortion even in this optimistic relaxation, there is no point searching that family for that objective.

### 5.3 Boolean versus real algebra

We must not confuse Boolean algebraic normal form (ANF) over $\mathbb F_2$ with the real polynomial expansion above. For Boolean bit functions, XOR is addition in $\mathbb F_2$; AND is multiplication; NOT adds 1. A parity bit is degree one over $\mathbb F_2$ but its numerical representation over the reals can have high degree.

| Restricted operation family | Proven or potential limitation |
|---|---|
| Real affine operations (no rounding, products of variables or nonlinear output mapping) | Remain affine regardless of program length |
| XOR, NOT, fixed shifts/rotations, constants | Affine in input bits over $\mathbb F_2$; cannot implement AND of two independent bits |
| AND, fixed shifts, constants (without NOT/OR) | Each output bit is a conjunction of selected input bits or a constant |
| AND with XOR and NOT | Functionally complete for finite Boolean mappings, with adequate wiring and output construction |
| Word addition, multiplication, variable shifts, popcount | May introduce nonlinear bit dependencies; exact behaviour depends on word width, carries, truncation and output conversion |

The existing bitwise instruction dictionary is therefore not universally limited to linear combinations in the original variables. It can express arbitrary finite Boolean mappings with enough resources. However, universal Boolean expressivity says nothing about the number of instructions needed, hardware efficiency, or compression. Many mappings require enormous descriptions. A finite word ISA also requires explicitly accounting for how input bits are made available and output words assembled.

### 5.4 Analyse both seed and index dependence

For an $N$-position tile, let $i$ be represented by $\lceil\log_2 N\rceil$ index bits. A joint expansion has terms

$$G(s,i)=\sum_{S,U}a_{S,U}\prod_{j\in S}s_j\prod_{k\in U}i_k.$$

Seed-only terms can control shared magnitudes/offsets. Index-only terms define a fixed pattern. Mixed terms enable changes in the pattern itself as the seed changes. If the real-valued function has no mixed terms, it is separable as $f(s)+h(i)$, so seeds change only a uniform offset. If a program cannot depend on certain index bits, there are guaranteed identical output positions; for power-of-two domains this can produce periodicity.

Track seed/index dependencies statically where possible. Do not assume that longer programs increase rank or that higher degree automatically improves coverage. High interaction degree may make encoding harder without providing useful geometry.

### 5.5 A constructive example: cheap orthogonal sequences and a fast encoder

A parity-based generator provides a useful sanity check:

```cpp
// For B seed bits and N = 2^B positions.
int G(uint32_t seed, uint32_t index) {
    return 1 - 2 * (popcount(seed & index) & 1);
}
```

Its vectors are the Walsh–Hadamard characters, $G(s,i)=(-1)^{\langle s,i\rangle_{\mathbb F_2}}$. For $N=2^B$ and distinct seeds they are mutually orthogonal with squared norm $N$. The nearest fixed-amplitude codeword is the one with maximum correlation; a fast Walsh–Hadamard transform computes all correlations in $O(N\log N)$ instead of $O(N^2)$. If a sign or scale is allowed, encode it under the bit budget and modify the score accordingly.

This illustrates the desired generator–encoder relationship, not a good general-purpose codec by itself. Selecting one orthogonal vector is restrictive; it does not allow arbitrary linear combinations of the entire basis. Note also that hardware cost of the C-like operations needs checking in compiled GPU assembly.

### 5.6 General encoding objective and algebraic structure

For an affine binary-seed generator $g(s)=As+b$, squared error expands to

$$E(s)=s^TA^TAs-2s^TA^T(w-b)+\|w-b\|^2.$$

This is a binary quadratic optimisation problem. If $A$ has orthogonal columns, each bit can be chosen independently: choose $s_j=1$ iff $2a_j^T(w-b)>\|a_j\|^2$ (ties can be arbitrary). Encoding then takes $O(BN)$, rather than $O(2^B N)$ for direct exhaustive scoring. Nonorthogonal columns introduce interactions and the general problem can be difficult.

For nonlinear generators, substituting $\phi(s)$ yields a higher-order binary optimisation problem. A sparse interaction graph, bounded treewidth, additive blocks or trellis structure may permit dynamic programming or other structured algorithms. The exact polynomial form alone does not guarantee tractability. A strongly mixing PRNG may be a good codebook but offer no useful local search direction. This is the motivation for **co-designing the generator and encoder**, rather than designing a decoder first and assuming efficient encoding will follow.

## 6. Hardware feasibility: is instruction bandwidth really different from weight bandwidth?

### 6.1 Instruction fetching and the code/data distinction

The user questioned whether replacing dense data accesses with instruction streaming changes anything if both compete for the same memory bandwidth. The answer depends on where the information resides and how it is reused.

MI300X has separate instruction, scalar and vector L1 caches. The lower cache/memory system is shared, so instruction misses can compete with weight traffic. A cached generator, however, can execute many times without refetching its code from HBM. One wavefront instruction also operates on many lanes, with different seeds and/or indices as operands. Instruction size must not be confused with the number of numerical lane operations performed. Conversely, a unique large kernel for every tile risks code-cache thrashing and transfers the original bandwidth problem to the instruction path. Interpreting unique encoded programs at runtime also turns them into data loads and introduces interpreter overhead.

Published MI300X specifications (see [AMD architecture](https://rocm.docs.amd.com/en/docs-6.2.0/conceptual/gpu-arch/mi300.html), [ROCm hardware specifications](https://rocm.docs.amd.com/en/docs-6.3.2/reference/gpu-arch-specs.html), [product page](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)):

| Resource | MI300X |
|---|---|
| XCDs and active CUs | 8 XCDs; 304 active CUs (38 per XCD) |
| Wavefront size | 64 lanes |
| LDS | 64 KiB per CU |
| L1 vector cache | 32 KiB per CU |
| L1 instruction cache | 64 KiB shared by two CUs |
| L1 scalar cache | 16 KiB shared by two CUs |
| L2 | 4 MiB per XCD; 32 MiB aggregate |
| HBM bandwidth | 5.3 TB/s peak |
| BF16 matrix throughput | Approximately 1.3 PFLOP/s peak |

These are nominal resources, not achieved rates. A generator that fits in the instruction cache can still compete for issue slots, registers, vector execution resources, and lower-level cache bandwidth. Actual code size includes the enclosing kernel as well as the generator. XCD placement, cache reuse and data layout need controlled measurement.

### 6.2 Pipelining: latency hiding does not imply sufficient throughput

The user noted that a generator changes a parallel weight retrieval into a sequence of instructions and wondered if it could be latency-hidden. We need not generate the entire logical GEMM tile before starting matrix operations; we need to have the register fragment required by the next MFMA ready. Software pipelining or independent wavefronts may overlap generation of a subsequent fragment with matrix work on a previous one. This depends on scheduling, register allocation, hardware resource sharing, and dependence chains.

```text
Conventional:  HBM/cache -> (optional LDS) -> VGPR operand -> MFMA
Procedural:    compact data -> VGPR/SGPR -> generator -> VGPR operand -> MFMA

Potential overlapped execution (schematic):
               fragment 0    fragment 1    fragment 2
Generator:     [produce 0]   [produce 1]   [produce 2]
MFMA:                        [consume 0]   [consume 1]   [consume 2]
```

If generation produces fragments more slowly than MFMA consumes them, the pipeline will stall regardless of latency hiding. LDS is already a fast on-chip resource. Replacing a well-coalesced LDS read with 100 dependent vector instructions per weight is unlikely to pay off in isolation. The primary opportunity is avoiding HBM traffic or avoiding an expensive staging/unpacking/layout pipeline. Bypassing LDS may also save space and improve occupancy; alternatively, extra VGPRs can reduce occupancy and cause spills. Both effects need measurement.

The ideal generator should produce weights in the lane-specific register layout consumed by the chosen MFMA instruction, rather than generate a dense row-major intermediate and pay an extra shuffle. Logical compression tiles and MFMA fragments are distinct concepts; their layouts should be co-designed but not conflated.

### 6.3 A simple compute/memory envelope

Assume, purely for an illustrative first-order model, 304 CUs with 64 stream processors per CU, a 2.1 GHz clock, and one useful 32-bit integer lane operation per stream processor per cycle. This gives a nominal $P_{\rm gen}\approx40.86$ trillion lane operations per second. This is **not** a measured CDNA3 integer throughput figure or a universal ISA guarantee. Operation classes, packed instructions, issue limits and achieved clock may differ.

At 5.3 TB/s, the illustrative budget is about 7.71 lane operations per byte of streamed weights. Ignoring decoder overhead and competing traffic:

| Reference weight payload | Bytes per weight | Weight-only HBM ceiling (trillion weights/s) | Illustrative arithmetic budget within streaming time (ops/weight) |
|---|---:|---:|---:|
| BF16 | 2 | 2.65 | 15.42 |
| INT8 | 1 | 5.30 | 7.71 |
| 4-bit | 0.5 | 10.60 | 3.85 |
| 2-bit | 0.25 | 21.20 | 1.93 |

For a generator with $c$ effective operations per weight and stored bytes per weight $b$, an optimistic weight-supply roof is

$$Q\leq\min\left(\frac{BW_{\rm HBM}}{b},\frac{P_{\rm gen}}{c}\right).$$

This assumes perfect overlap, ideal traffic and no other costs. It is an upper-bound-style model, not a speed prediction. If weights are regenerated for multiple consumers, account for that work; if metadata is reread by many CUs, account for that traffic. The actual comparison is between full fused kernels with matching mathematical outputs, precision and batch sizes; the baseline also incurs dequantisation and memory-layout costs. MI300X does not natively have the same FP4 matrix instructions as later CDNA4 hardware, so compare packed INT4/FP4 storage with the actual conversion path to supported MFMA types.

The crucial observation is that reducing bpw beyond the point at which generation is the bottleneck cannot improve this particular throughput ceiling. A 100-operation generator is only attractive if the computation is genuinely amortised across many outputs, not if it executes 100 dependent operations for each weight.

### 6.4 Multi-output synthesis rather than independent coordinate evaluation

A program may be short in source but expensive in compiled instructions. Conversely, a longer program may reuse seed-dependent intermediate values across multiple outputs, perform packed arithmetic, and map naturally to SIMD lanes. We should explicitly synthesise a multi-output computational DAG and separate:

- Seed-only setup (potentially shared over indices or lanes).
- Coordinate-only calculations (potentially implicit or reused).
- Seed–coordinate interactions that actually generate variation.
- Output conversion, packing, register rearrangement and communication.

For example, the best small program in an earlier experiment computes `(seed * 0x9e3779b9u)` independently of index. That multiplication could be hoisted if multiple outputs share it and the lane mapping permits. A scalar expression does not establish that sharing is possible on the GPU; count broadcasts/shuffles and register lifetimes too.

In the long term it is also worth considering **operator generation** rather than weight generation. If $W$ has a structure such as $AB$, the application can be computed as $A(Bx)$ without materialising $W$. Butterfly/Hadamard-type transforms offer other examples. This is a distinct and more restrictive research route; do not assume arbitrary procedural generators permit fast direct multiplication.

## 7. Evidence and experimental history

**Provenance:** The numbers in this section are from previously supplied internal experiment reports and conversation records. The original repository, raw arrays, logs, scripts and profiler traces were not independently inspected when preparing this handover. Recover those artifacts and reproduce the reported baselines before relying on them for a publication. The exact checkpoint revision and dataset manifests should be obtained from the original runs. No model-level accuracy or MI300X performance result has yet been established for our synthesised codec.

### 7.1 Initial procedural dictionary pursuit

Initial experiments used the SmolLM2-135M BF16 checkpoint, 21 sampled attention/FFN projection matrices from layers 0, 15 and 29, tile scales between $4\times4$ and $64\times64$, and full-row or row-segment representations. The approach attempted to fit weight blocks using mixtures of procedural atoms. Original and shuffled controls were compared to assess whether positional structure was being exploited.

Reported 2-bpw pooled reconstruction NMSE:

| Tile configuration | Procedural result | Relevant controls / observations |
|---|---:|---|
| $4\times4$ | 0.6591 | Sparse reference 0.6780 |
| $8\times8$ | 0.6848 | Global shuffle 0.6896; within shuffle 0.6914; Lloyd–Max 0.3814 |
| $16\times16$ | 0.8599 | Shuffle results 0.8603/0.8607; scalar reference 0.4026 |
| $32\times32$ | 0.9645 | Little useful reconstruction |
| $64\times64$ | 0.9909 | Nearly zero-predictor error |
| Full row | 0.954 | Poor fit |

A larger pursuit on $16\times16$ at 4 bpw reached only 0.8359 versus scalar 0.0423. A planted procedural control was reconstructed with near-zero error, showing that the method could recover at least some deliberately generated structure. A shared FFN permutation changed held-out $16\times16$ NMSE only from 0.8596 to 0.8577. Whole-matrix 16-atom reconstruction was 0.99983 versus 0.99990 for global shuffle.

Interpretation: the tested atoms did not reveal substantial exploitable positional structure and scaled poorly to large blocks. This does **not** prove absence of useful structure in neural weights or invalidate the original search-over-algorithms hypothesis. It motivates using stronger implicit codebooks and treating shuffle/random controls as central experiments.

### 7.2 SeedLM reproduction and modifications

The reported reproduction used the same 21 matrices with disjoint contiguous held-out spans. SeedLM's nonzero 16-bit LFSR state defines a $C\times P$ basis; coefficients are fitted jointly using a pseudoinverse and then represented with low-precision values and a shared exponent. Exhaustive enumeration covers 65,535 nonzero seeds, not every combination of quantised coefficients.

| Reproduction configuration | Weights $C$ | Basis columns $P$ | Seed bits | Coefficient bits total | Exponent bits | Total |
|---|---:|---:|---:|---:|---:|---:|
| 3 bpw | 12 | 4 | 16 | 16 | 4 | 36 bits |
| 4 bpw | 8 | 3 | 16 | 12 | 4 | 32 bits |

Reported NMSE as seed enumeration increased:

| Seeds searched | 3 bpw | 4 bpw |
|---:|---:|---:|
| 1 | 0.7324 | 0.7146 |
| 4,095 | 0.1474 | 0.0707 |
| 65,535 | 0.0952 | 0.0370 |

At 4 bpw the global shuffle, within shuffle, and Gaussian controls gave 0.0385, 0.0363 and 0.0367 respectively. With floating-point coefficients (an unbudgeted oracle), NMSE went from 0.6289 with one seed to 0.0091 with exhaustive search. For one random rank-3 subspace in eight dimensions the isotropic expected projection residual is $1-3/8=0.625$, consistent with the one-seed oracle.

This is an important result for the original intuition: a large search across pseudorandom subspaces can find good coincidences even for Gaussian inputs. It is not evidence that the original weights are themselves generated by a PRNG.

Further reported coefficient/exponent refinements:

| Method | 3 bpw NMSE | 4 bpw NMSE |
|---|---:|---:|
| Published-recipe reproduction | 0.09518 | 0.03698 |
| Range-fitting exponent | 0.05258 | 0.01547 |
| LFSR with quantisation-aware refinement | 0.05031 | 0.01454 |
| Xorshift with refinement | 0.04733 | 0.01389 |
| SplitMix with refinement | 0.04854 | 0.01385 |

The exponent change was reported to preserve the LFSR decoder and payload. The original authors' exact implementation is not known from the conversation; do not claim a definitive improvement over their published model-level results on the basis of this reproduction. Greedy fitting was worse than joint fitting, and the shared-basis approach outperformed the particular independent-seed construction tested.

A screen of 430 LFSR configurations reported the following selected points:

| bpw | Weights/block | Seed bits | Coefficients | NMSE |
|---:|---:|---:|---|---:|
| 0.969 (rounded) | 32 | 12 | 5 × 3-bit | 0.4557 |
| 2 | 12 | 8 | 4 × 3-bit | 0.1997 |
| 3 | 8 | 8 | 4 × 3-bit | 0.0652 |
| 4 | 8 | 8 | 5 × 4-bit | 0.0218 |

These rates also included a four-bit exponent. The 4-bpw scalar comparator in that screen was 0.0413. These are exploratory selected configurations, not independently audited final model results; different experiment splits and fitting conditions explain why a 2-bpw result here need not match the next experiment's matched SeedLM baseline.

### 7.3 Automated coefficient-free instruction synthesis

The next experiment attempted to synthesise cheap coordinate generators directly rather than storing basis coefficients. The setup was:

- Eight weights per block; a 12-bit seed plus signed 4-bit exponent (16 total bits, 2 bpw).
- A single shared generator across the sampled matrices.
- Unsigned 32-bit operations: `ADD SUB MUL XOR AND OR NOT SHL SHR ROTL POPCOUNT`.
- Inputs: seed, index, a fixed set of 20 literal constants and previously computed DAG nodes. Shift/rotation counts use the low five bits. Arithmetic wraps modulo $2^{32}$ where applicable.
- Final value: low output byte interpreted as signed int8, scaled by $2^e$ with $e\in[-8,7]$.
- Exhaustive seed and scale scoring for each candidate; the exact nearest power-of-two scale selection used the correct quadratic criterion and was checked against brute force (midpoint between adjacent scales $t$ and $2t$ is $1.5t$ for a positive ideal scale).
- Reported disjoint block sets: 10,752 calibration, 42 generator-search, 1,344 validation and 2,688 final hold-out blocks, using disjoint rows/spans. Verify split construction in the original scripts. Metric was pooled residual squared energy divided by pooled original squared energy.

The one- and two-instruction grammars were enumerated exhaustively within the finite grammar. Three-instruction search was bounded: approximately 240,000 visited structures out of an estimated 113.3 million live structural DAGs, 21,508 scored, about 51 seconds on an M1 system. The two-instruction run visited 324,148 structures, scored 5,019, and took about 12 seconds. Semantic fingerprints were reportedly verified across all seeds before discarding equivalent candidates. Revalidate that equivalence is appropriate for the exact output width, index domain and architecture cost; two numerically equivalent programs may have different hardware costs.

| Best candidate | NMSE (2 bpw) | Reported distinct scaled codewords | Abstract cost per 8-weight block |
|---|---:|---:|---:|
| `SHL(seed,index)` | 0.55595 | 4,081 | 33 |
| `ROTL(seed, SHR(seed,index))` | 0.23622 | 60,332 | 41 |
| `x=SHR(MUL(seed,0x9e3779b9),index); SHR(x,x)` | 0.19910 | 65,473 | 49 |

The best three-operation generator in C-like pseudocode:

```cpp
uint32_t raw(uint32_t seed, uint32_t index) {
    uint32_t x = (seed * 0x9e3779b9u) >> (index & 31u);
    return x >> (x & 31u);
}
// Interpret low byte as signed int8, then multiply by 2^e.
```

The signed byte conversion must be explicitly defined in cross-platform reference implementations. The multiplication wraps; masked shift counts are essential for a faithful interpretation of the grammar. The reported three-op candidate has dependency depth 3, peak live intermediates 2, 4,093 unique unscaled seed outputs and effective rank 7.95 out of 8. Verify the exact effective-rank definition in the original code. With a 16-bit seed plus 4-bit exponent (2.5 bpw) the same frozen generator reportedly achieved 0.10997 NMSE.

Matched 2-bpw results from this experiment:

| Method | Weight NMSE |
|---|---:|
| Direct LFSR | 0.38908 |
| Direct Xorshift | 0.26730 |
| Direct SplitMix | 0.28549 |
| Calibrated LFSR | 0.28050 |
| Calibrated Xorshift | 0.19408 |
| Calibrated SplitMix | 0.18807 |
| Matched SeedLM variant | 0.21658 |
| Synthesised three-op program | 0.19910 |
| Lloyd–Max | 0.17292 |
| Explicit Gaussian VQ | 0.16861 |
| Trained VQ | 0.16449 |

The calibrated PRNG baselines selected an arithmetic output shift (0–5) on validation. The trained VQ reference used a reported 32 KiB codebook. Confirm its storage format and the distinction between codebook entry index and total tile metadata; a table is not automatically budget-equivalent to a program at small sharing counts.

Reported paired matrix-resampling confidence intervals for the three-op generator's NMSE difference (generator minus comparator) were approximately:

| Comparator | Interval |
|---|---|
| SeedLM | [-0.03753, +0.01497] |
| Calibrated Xorshift | [-0.01492, +0.03620] |
| Calibrated SplitMix | [-0.00742, +0.03925] |

These intervals include zero. The synthesis experiment demonstrated automatic discovery of useful tiny computational codebooks, **not statistical superiority over calibrated PRNGs, state-of-the-art reconstruction, or inference acceleration**. The abstract operation counts include conversion/scaling assumptions, not actual GPU instruction counts. Fair output calibration across all shortlisted programs, multi-output synthesis, and compiled hardware measurements remain outstanding.

### 7.4 What the results do and do not establish

- Finding good seeds can be surprisingly powerful even without source-specific patterns. SeedLM and random codebooks illustrate this.
- The prior synthesis infrastructure is a good starting point. It explores a finite hardware-inspired grammar and produces reproducible codewords for a fixed format.
- There is a gap to stronger VQ and calibrated PRNG baselines at 2 bpw. Neither rank near 8 nor many unique codewords was sufficient to close it.
- Large independent tiles performed poorly in the tested post-training experiments. This motivates stronger rate–distortion references and compression-aware training, not unlimited expansion of the seed or program budget.
- All internal results so far are primarily weight reconstruction metrics, not proof that a compressed neural network works or that a fused kernel is faster.

## 8. What counts as a successful reconstruction?

The user explicitly asked whether NMSE describes numerical weights rather than actual model performance. Yes: our reported pooled NMSE is

$$\mathrm{NMSE}_{W}=\frac{\sum_t\|w_t-\hat w_t\|_2^2}{\sum_t\|w_t\|_2^2}.$$

It compares signed numerical values, not just their absolute magnitudes. NMSE 0.2 means the residual has 20% of the original squared energy; it does not mean 20% of weights are wrong or that model accuracy has fallen by 20%. NMSE 1 corresponds to zeroing all weights under this convention.

Example: original weights $w=[1,1]$, input $x=[10,0]$, output 10. Reconstructions $[0.5,1]$ and $[1,0.5]$ both have NMSE 0.125, but outputs 5 and 10 respectively. Identical weight error can have radically different functional consequences.

For a linear layer $y=Wx$ and calibration activation matrix $X$, define

$$D_{\rm act}(W,\widehat W)=\|(W-\widehat W)X\|_F^2,$$

$$\mathrm{NMSE}_{\rm act}=\frac{D_{\rm act}}{\|WX\|_F^2}.$$

Equivalently, the unnormalised error is $\operatorname{tr}((W-\widehat W)XX^T(W-\widehat W)^T)$. This highlights directions that matter to real inputs. Note that off-diagonal activation correlations can couple independently encoded tiles; independent tile-local scoring may only approximate the full-layer objective. A block-diagonal Hessian or covariance approximation is one possible starting point, with full-layer error checked afterward.

For subsequent evaluation, report separately:

| Quantity | Why it matters |
|---|---|
| Weight NMSE, tail errors and errors per matrix | Numerical approximation and failure modes |
| Activation error / logits discrepancy | Impact on the actual computations |
| Perplexity on held-out text | Language model functionality |
| Downstream task accuracy, if scale permits | Functional generalisation |
| Actual payload and total serialized model size | Information and capacity cost |
| Encoder time and peak memory | Offline practical feasibility |
| Fused inference time, bandwidth and utilisation | Realised execution benefit |

Use the existing synthesised generator for an initial model-level evaluation **before** another major synthesis campaign. Replace one projection, one complete FFN, then all target FFN and attention projections; keep other modules identical and explicitly record what remains uncompressed. Measure BF16, a conventional 4-bit baseline and the matched 2-bit methods on the same data and compression scope. Do not use the evaluation set to choose the generator or calibration constants.

## 9. Related work and positioning

These are initial references, not a completed systematic literature review. The closest papers must be read in full and their implementations studied before claiming novelty. Links below point to paper or project pages verified during preparation of this document.

| Reference | Relevance / question to investigate |
|---|---|
| [SeedLM: Compressing LLM Weights into Seeds of Pseudo-Random Generators (ICLR 2025)](https://arxiv.org/abs/2410.10714) | Fixed LFSR, per-block seed and coefficients; what is the best achievable trade-off if we replace that basis generator with a synthesised GPU-native program? See [method and results](https://arxiv.org/html/2410.10714v2). |
| [QuIP: 2-Bit Quantization of Large Language Models With Guarantees (NeurIPS 2023)](https://arxiv.org/abs/2307.13304) | Incoherence processing and adaptive rounding; strong reminder that geometry of weights and of task-sensitive directions can be transformed before encoding. |
| [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks (ICML 2024)](https://proceedings.mlr.press/v235/tseng24a.html) | Lattice VQ, randomised Hadamard preprocessing and fine-tuning; compare structured codebooks against unrestricted PRNG codebooks. |
| [AQLM: Extreme Compression of Large Language Models via Additive Quantization (ICML 2024)](https://proceedings.mlr.press/v235/egiazarian24a.html) | Additive multi-codebook representation; compare composing generated vectors and efficient encoding strategies. |
| [QTIP: Quantization with Trellises and Incoherence Processing (NeurIPS 2024)](https://arxiv.org/html/2406.11235) | Particularly close prior work: hardware-friendly bitshift trellis, computed codebooks, lookup-free variants and dynamic-programming encoding. |
| [Proteus: Lookup-Free Trellis-Coded Quantization by Lattice-Breaking Compute Codes for 2-Bit LLMs (ICML 2026)](https://proceedings.mlr.press/v306/yang26ah.html) | Particularly close prior work: integer compute codes, overlap-local geometry and per-layer calibration; [official implementation](https://github.com/YZW-explorer/Proteus). |
| [SparseRT (PACT 2020)](https://arxiv.org/abs/2008.11849), [SparGen](https://www.microsoft.com/en-us/research/publication/a-new-approach-to-deep-learning-model-sparsity-via-tensor-with-sparsity-attribute/), [SparTA (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/zheng-ningxin) | Code specialisation, the code/data partition, and the hardware implications of instruction footprint. |
| [AMD CDNA3 / MI300 architecture](https://rocm.docs.amd.com/en/docs-6.2.0/conceptual/gpu-arch/mi300.html) | Register layouts, cache and compute architecture; basis for the hardware-specific constraints. |

### 9.1 Why trellises deserve immediate attention

A conventional independent-block codebook consumes all $B$ bits anew for every $N$ weights. QTIP-style trellis quantisation instead permits consecutive decoder states to share bits. For example, an $L$-bit state can shift in $k$ new bits at each step and compute an output from the resulting state. This has a long effective context but consumes only $k$ fresh bits per step, plus boundary conditions and any other metadata.

A trellis with $2^L$ states and $2^k$ outgoing transitions per state has dynamic-programming structure. For a suitable additive distortion, a Viterbi-like procedure can find a best valid path without explicitly enumerating every full sequence. The complexity still depends on state size; overlap is not free information because valid adjacent states are constrained. Tail-biting, segment boundaries and random access must be handled explicitly.

We should investigate synthesis of the computation inside such a trellis and compare it to the original independent-block design. A generator with good marginal Gaussianity may still have poor joint statistics for overlapping states; Proteus is especially relevant to this issue. The existing polynomial framework could be extended to analyse those local joint statistics. Do not assume our first candidate must beat trellis methods; they may provide a better structural foundation than independent seeds.

A conventional PRNG with an arbitrary seed is only one point in the broader design space. Future literature work should also include procedural model compression, random bases (e.g. PRANC/NOLA as discussed in SeedLM's related work), neural network intrinsic dimensionality, hypernetworks, structured operators, Boolean circuit complexity, Boolean Fourier analysis, program synthesis and rate–distortion theory. Verify specific papers, claims and code before adopting them.

## 10. The immediate research programme

The following work packages are proposed in order of dependency, not a rigid serial schedule. In particular, hardware feasibility and model-quality checks can proceed while target geometry and mathematical analysis are being developed.

### WP0 — Recover and freeze the existing experimental baseline

**Deliverable:** A reproducible baseline manifest, plus a small deterministic test that reproduces the prior three-op codebook and NMSE calculation.

1. Locate the original experiment repository, checkpoint revision, matrix selection, tile layout and exact train/search/validation/hold-out manifests. This document does not supply a verified repository path.
2. Recover the existing LFSR, PRNG, VQ and synthesis scripts; do not rewrite them without reason. Preserve raw results and configuration files.
3. Verify the uint32 instruction semantics, signed-byte conversion and exponent convention against known codewords. Test equivalence and scale-fitting against exhaustive small examples.
4. Establish disjoint calibration, search, validation and final test sets. Avoid data leakage from adjacent/repeated weights and from using test data to optimise scale or programs. Preserve prior holdout results; allocate a new untouched test split for new design choices if the previous holdout has already influenced the next experiments.
5. Record model ID/hash, matrix names, shapes, tile ordering, random seeds, precision, software versions and execution environment. Save both per-matrix and pooled statistics.

### WP1 — Characterise the target weight point cloud

**Research question:** Can target-space geometry tell us which computational codebooks are worth constructing, and can it identify when we are simply fighting an information limit?

Use SmolLM2-135M and the existing 21-matrix sample initially. Start with a deliberately small pilot (8, 16, 32, 64 weights per tile, both contiguous row segments and compatible rectangular tiles). Expand to 256 and 1,024 weights, different layouts and additional models only after the pipeline works. Analyse projection types separately; pooling matrices of different scales can create misleading apparent clusters.

For each matrix/layout prepare:

- Original contiguous or rectangular tiles, with fixed flattening convention.
- Within-matrix shuffled weights before retiling: preserves the marginal distribution, disrupts positional dependencies. A shuffle within each tile instead preserves its norm and marginal multiset while disturbing local positions; distinguish these controls.
- A global shuffle across matrices where appropriate, with awareness that it mixes distributions and scales.
- Synthetic IID Gaussian weights with matched mean/variance, and optionally independent samples from the empirical marginal to separate non-Gaussian marginals from spatial correlations.
- Raw, matrix-normalised, and unit-norm tile versions (unit-norm analysis is diagnostic, not permission to use a free per-tile scale). Handle zero tiles explicitly.

Compute the following:

| Statistic | Intended use |
|---|---|
| Mean, variance, norms, tails, outliers | Decide whether fixed scaling, group scaling, or explicit magnitude bits matter |
| Covariance and cumulative PCA spectrum | Derive the minimum affine span required for a specified expected reconstruction error |
| Effective rank / participation ratio | Detect concentration of energy in few dimensions; record the exact definition |
| Pairwise cosine similarity and directional statistics | Distinguish magnitude variation from useful directional coverage |
| Within-tile position correlations, lag spectra, higher-order dependencies | Guide index interactions, basis selection, shifts/rotations and layout hypotheses |
| Nearest-neighbour distances, clustering and intrinsic-dimension estimates | Test whether local geometry suggests structured or hierarchical seeds |
| Cross-tile correlation, adjacent tile similarity | Test whether descriptor or state sharing can help |

For the PCA analysis, report $K_\epsilon$, the smallest rank that captures a specified fraction of centred variance (e.g. 90% or 95%), and separately report the mean energy. Fit PCA on calibration tiles; measure its affine projection error on held-out tiles. Estimate uncertainty with matrix-level or block-level resampling. High-dimensional distances concentrate, so compare to controls at the same dimension and sample size. Intrinsic-dimension estimators can be unreliable when samples are sparse or noisy; do not treat one estimate as proof of a manifold.

**How this is used:** If a target requires at least $K$ dimensions for a reconstruction tolerance, prune generator families whose proven maximal affine dimension is smaller. If direction distributions are close to IID Gaussian, prioritise codeword shaping and strong universal computational codebooks rather than elaborate index-specific pattern recovery. If anisotropy or norm variability is reproducible, test matched structure or explicit budgeted scaling. If tiles cluster, try hierarchical seed allocation or a small portfolio. These are hypotheses to evaluate, not rules that guarantee good VQ.

**Output:** `target_geometry/` results containing per-matrix covariance/eigenvalue data, norm and cosine distributions, original-vs-control plots, PCA projection bounds, and a concise report stating which observed differences are actually reproducible. Include negative results.

### WP2 — Establish the empirical rate–distortion frontier

**Research question:** At a given fixed information budget, how much of the error comes from an inadequate codebook versus the inherent difficulty of the source?

First use a total per-tile budget of 16 bits, matching the previous 12-bit seed + 4-bit exponent setup. Then include 8- and 12-bit budgets. A split between seed and exponent is a hyperparameter, not additional storage. Allow a no-exponent variant with fixed shared scale. The following payload rates are examples:

| Weights/tile | 8 total bits | 12 total bits | 16 total bits |
|---:|---:|---:|---:|
| 8 | 1 | 1.5 | 2 |
| 16 | 0.5 | 0.75 | 1 |
| 32 | 0.25 | 0.375 | 0.5 |
| 64 | 0.125 | 0.1875 | 0.25 |
| 256 | 0.03125 | 0.046875 | 0.0625 |
| 1,024 | 0.0078125 | 0.01171875 | 0.015625 |

Compare zero/mean prediction, scalar quantisation at feasible matched budgets, random Gaussian codebooks, trained explicit VQ (k-means or stronger methods), calibrated conventional PRNGs, SeedLM variants and the existing synthesised program. A scalar codec at below one bit per weight needs a genuinely matched representation (e.g. grouping or sparsity); do not pretend standard scalar quantisation directly supports fractional bits per weight. Keep scale/metadata accounting explicit and separate payload bpw from total effective bpw.

A trained explicit VQ table is an intentionally flexible reference, not a proven optimal codebook. If all $2^{16}$ entries are used for 64 weights and FP16 table entries, the table is 8 MiB; for 1,024 weights it is 128 MiB. Training it requires enough independent tiles to avoid overfitting. Do not begin by running naive full k-means on 65,536 centres for every matrix/tile combination. Start with 8- or 12-bit codebooks or the actual 12-bit seed + 4-bit scaling structure, batch distance evaluations, use sufficient calibration samples, and scale up only where useful. Record the codebook storage, training compute, held-out distortion and exact scale format.

Evaluate original, shuffled and synthetic sources with matched splits. Compare to the Gaussian theoretical lower bound only where its assumptions apply; it is not a universal empirical bound for all neural weights. Use multiple VQ initialisations and report uncertainty. Where a full oracle is infeasible, explicitly label the available reference as approximate.

**Decision logic:** A large gap between strong explicit VQ and our generator indicates room for improving the program family/search. Similar poor results suggest a limitation of the tested bitrate, source geometry or objective, although an inadequate VQ optimiser can also be responsible. Similar good results support the computational codebook approach. A synthesised codec beating a trained VQ table can reflect suboptimal VQ training or generalisation, not a violation of optimal VQ theory.

An optional follow-up is to use a learned VQ codebook as a teacher: approximate its geometry using a program, weighted by cluster usage or task importance. Codeword ordering is arbitrary, so do not force seed number $j$ to match centroid number $j$. Use a permutation-invariant or nearest-codeword objective and ultimately refine against actual target tiles. Matching distribution moments alone is not enough.

**Output:** A rate–distortion table/plot for every evaluated source, plus payload/total storage accounting and comparisons with the prior 2-bpw numbers. Include the fixed-seed block-size sweep even when the results deteriorate, because that is a central test of the larger procedural hypothesis.

### WP3 — Build the algebraic generator analyser

**Research question:** Can we derive constraints or predictive geometric properties from program structure, instead of scoring every candidate as a black box?

Begin with 2–8 seed bits and small index domains. Use a deterministic fixed-width interpreter for the existing grammar. For a program with $B$ seed bits and $N$ coordinates, enumerate a reference codebook matrix $C\in\mathbb R^{2^B\times N}$. Include the full decoded numerical mapping (sign, output truncation, scale); if analysing just the seed core, label that separately.

Implement two complementary analyses:

1. **Exact enumerated algebra.** For each output bit, compute its Boolean ANF by a GF(2) subset transform; for numeric outputs, compute real multilinear coefficients by subset Möbius inversion. Optionally compute Walsh coefficients by a fast transform for more stable energy-by-order summaries. Verify reconstruction of all codewords. Exact enumeration is a tool for small domains, not the eventual scalable analyser.
2. **Symbolic or static analysis.** Propagate bit dependencies, constant bits, affine/non-affine classifications, conservative degree bounds, known equalities and index periodicity through the instruction DAG. Do not treat machine integer ADD/MUL as real affine operations; carries, masking, variable shifts and conversion matter. Use verified bit-vector semantics, potentially an SMT solver for equivalence or bounded properties when cheap enough.

For each program record actual affine rank (centre the codewords before SVD), numerical covariance spectrum, ANF degree per output bit, real polynomial degree, coefficient energy by interaction order, basis Gram matrix, collision rate, index dependencies, and codeword geometry. Measure seed locality by Hamming-neighbour distances and the correlation of distance to a target under bit flips; pure avalanche is not necessarily desirable. Measure codeword scale/norm distributions, nearest-neighbour distances and actual coverage of held-out targets.

**Required unit tests:**

| Test generator | Expected result |
|---|---|
| `G(s,i)=0` | One distinct codeword, affine rank zero |
| One-bit `s & i`, eight indices | Two codewords, rank one, period two |
| Two-bit `s & i`, eight indices | Four codewords, rank two, period four |
| XOR/NOT/fixed shifts | Boolean ANF degree at most one in the inputs |
| Two independent seed bits ANDed | Boolean and real degree two for that bit |
| Parity / Walsh–Hadamard example, $N=2^B$ | Pairwise orthogonal codewords, Boolean parity affine, real numeric interaction degree can be high |
| Known orthogonal affine basis | Independent-bit encoder matches exhaustive nearest codeword |
| Existing 3-op generator | Interpreter matches original reference, including signed conversion and scale |

Test random compositions against exhaustive truth tables. Correctly distinguish an affine span dimension from rank of an uncentred codebook; e.g. $N$ orthogonal vectors have ordinary linear rank $N$ but affine dimension $N-1$. Ensure numeric tolerances are documented. A symbolic pruning rule must be a proved necessary condition, not merely a correlation from a small sample.

**Output:** A reusable analyser that accepts a DAG, seed width, index domain and numeric output format and returns exact small-domain properties and sound static upper/lower bounds where possible.

### WP4 — Connect geometry to synthesis and improve the search

**Research question:** Which algebraic/geometric properties predict rate–distortion quality, hardware cost and encoding tractability?

Sample thousands of small programs from the finite grammar with diverse instruction families and cost levels. Reuse existing exhaustive enumeration and semantic deduplication. Record the generator features from WP3 and score each against the targets from WP1 using the same output calibration policy. A representative small target subset can screen candidates, but final ranking must be checked on independent validation data; keep the final evaluation set untouched.

Compare the predictive value of covariance alignment, affine dimension, degree, orthogonality, collision count, directional/norm distributions and locality. Check correlations within cost/bitrate strata and across matrices. For example, a covariance mismatch may correlate with error simply because one candidate's scale is wrong; remove or account for that confound. Use actual distortion as the final objective, and apply algebraic bounds as safe pruning only when justified.

The next generator search should include the following improvements:

- Calibrate the output mapping symmetrically for every shortlisted generator and comparator (scale, shift, bias where legal), with the same shared or local bit accounting.
- Search multiple outputs jointly with shared intermediate values; include a coordinate generator as a baseline. Prefer actual compile cost or a measured surrogate over an arbitrary source instruction limit.
- Try beam search and evolutionary mutations of opcodes, constants, operand choices and DAG edges; compare to random search and prior exhaustive grammar runs under matched evaluation budget/wall-clock. Record visited, unique, evaluated and retained programs.
- Retain multiple Pareto candidates rather than one lowest-NMSE program. Cost includes execution, code footprint, dependency depth, live registers, packing, and storage.
- Test portfolios of complementary generators at fixed total bits including selector. Compare to one decoder with a larger seed set of the same total size.
- Sweep sharing granularity and tile layout. Compare shared per-model and per-matrix generators first, followed by region-specific variants only if quality justifies them.
- Explore trellis-state generators and overlap-local quality metrics, based on the QTIP/Proteus literature.

Program complexity should be a design variable, but arbitrary 100-op-per-weight programs are not a viable default. A longer program may only be considered if the measured amortised cost per generated weight remains low. Compare candidate families with the hardware feasibility envelope from WP-H.

A multi-objective formulation is preferred to prematurely choosing arbitrary penalty weights:

$$\text{Pareto minimise }\left(D, R_{\rm total}, t_{\rm decode}, t_{\rm encode}\right)$$

subject to correctness, an explicit program-sharing policy and target hardware constraints. First optimise reconstruction proxies cheaply, then measure actual fused decode/inference time. A lower-NMSE candidate can be rejected if it is too slow or unencodable.

**Output:** A measured comparison of guided and unguided synthesis, a list of safe algebraic pruning rules, a set of useful (and failed) geometric predictors, and a Pareto frontier of generators suitable for functional/hardware testing.

### WP5 — Co-design an efficient encoder

**Research question:** Can a generator's algebraic structure be exploited to find acceptable seeds without searching the complete seed space for every target?

There are two different offline searches: synthesising $G$ and choosing $s$ given $G$. Neither should be confused with the fast online decoder. With 8–16 bits, exhaustive seed scoring is practical as a ground-truth reference and can sometimes be acceptable offline. Reuse a generated codebook during encoding and score many tiles with batched dot products; this temporary encoder-side table does not imply runtime inference lookup overhead. Its memory/time still belongs in the encoder budget.

Investigate methods appropriate to the generator structure:

| Structure | Candidate encoder |
|---|---|
| Orthogonal affine basis | Independent bit selection / rounding |
| Walsh structure | Fast transform |
| Additive or hierarchical basis | Matching pursuit, beam search, coarse-to-fine optimisation |
| Sparse low-order interaction graph | Binary optimisation, dynamic programming when structural conditions hold |
| Trellis with additive distortion | Viterbi-style dynamic programming |
| Arbitrary discrete codebook | Exact or approximate nearest-neighbour index, batched exhaustive scoring |
| Smooth continuous latents | Gradient optimisation followed by quantisation or quantisation-aware training |

Test local seed search and bit-flip hill climbing, but expect them to fail for strongly mixing hash-like functions unless useful locality can be established. A solver may target an approximate answer: compare its distortion relative to the exact nearest seed, its speed, setup cost, memory use and worst-case failures. Do not trade away inference quality merely to make offline encoding faster if offline search was already practical. The broader goal is scalable synthesis and encoding as the structure becomes more expressive, not avoidance of brute force at any cost.

**Output:** Encoders paired with generators, measured against exhaustive reference solutions on held-out tiles; a clear account of which algebraic properties enabled the speedup.

### WP-F — Functional model evaluation (parallel priority)

Use the existing 2-bpw winner and matched comparators immediately, rather than waiting for WP4. Start with offline reconstruction into ordinary dense tensors, allowing evaluation with a standard inference framework. This measures representation quality in isolation; it is **not** a compressed-memory or speed benchmark.

1. Establish BF16 perplexity on a fixed public held-out text split, with an explicit tokenizer, sequence length and scoring protocol. Use a separate calibration corpus if required.
2. Test one projection, one complete FFN, all FFN projections, then all selected linear layers. Record layer-specific activation error, logits discrepancy and perplexity. Include scale-only and conventional quantisation controls.
3. Compare weight-only encoding with activation-aware encoding and, later, a short matched-budget fine-tuning phase. Keep the same model architecture, compressed modules and rate accounting across comparisons.
4. Repeat on a larger model only once the experiment harness is sound. SmolLM2-135M may respond differently to compression from larger models; do not generalise one result to all LLMs.

**Gate:** No claim of a competitive 2-bit codec until model-level quality has been compared against strong 2-bit baselines and an optimised 4-bit reference. Define acceptable perplexity and accuracy degradation for the application before judging results; the original discussion did not establish a numerical tolerance. Use full Pareto comparisons rather than a single invented threshold.

### WP-H — GPU feasibility and fused execution (parallel priority)

Use HIP/ROCm and MI300X; start with synthetic, deterministic inputs so weight representability does not confound hardware results. Reuse the earlier MFMA experiment infrastructure if available. Produce a generator cost model from measured kernels rather than relying solely on nominal throughput.

Implement comparable kernels for:

- Dense BF16 HBM-to-register loading and conventional HBM/LDS/register staging; test cache-hot and cold cases separately.
- Packed INT4 (and optionally FP4 storage) conversion to MFMA-supported values, plus packed INT2.
- Register-only generation from compact descriptors using LFSR, Xorshift, SplitMix, the existing synthesised generator and structured multi-output examples.
- Generation with independent operations vs equal-count dependent chains; vary reusable intermediates, packing and direct MFMA fragment layout.
- Generator and MFMA fused in one kernel, comparing identical GEMM/GEMV shapes, mathematical operands and precision. Use register-resident generation with no unnecessary write of dense weights to HBM.

Measure dependency depth and compiled ISA, instruction count and encoding size, VGPR/SGPR use, occupancy and spills, LDS use, achieved bandwidth, vector arithmetic and matrix activity, and kernel/end-to-end time. Use available ROCm profiling counters with documented definitions; not every desired counter may be supported. Record launch overhead, warmup, repetitions and variance. Confirm correct outputs against a reference. Ensure the compiler has not eliminated the computation or constant-folded the synthetic workload. Check actual physical memory traffic where possible instead of equating payload size with HBM bytes.

Sweep batch sizes 1, 8, 32 and 64 and representative projection shapes. Batch-1 decode may use a different optimal kernel than a larger GEMM; do not force a bad dense baseline to use MFMA where it is inappropriate. Use tuned libraries/kernels (e.g. rocBLAS/hipBLASLt or a suitable quantised implementation) alongside matched microbenchmarks. Vary instruction footprint and cache reuse, including generator sharing across many tiles. Investigate how many outputs a shared setup can really produce without excessive cross-lane communication. Separate a wavefront instruction count from scalar lane operations and useful weights produced.

The first hardware result should be a **feasibility envelope**: measured generated weights/s and fused latency as a function of decoder cost, payload, shape, register pressure and reuse. Use it to constrain synthesis. A longer generator is only useful if its actual amortised throughput meets the target. Be explicit about what can and cannot overlap in the schedule; latency hiding cannot rescue a producer that lacks steady-state throughput.

**Gate:** A representation that is much smaller but slower than a tuned 4-bit baseline may still be useful for fitting models in memory, but it has not achieved the original acceleration target. Report both capacity and speed as distinct benefits.

## 11. Longer-term direction: train within a procedural weight space

### 11.1 The representability and trainability hypotheses are different

The user asked whether the project ultimately relies on finding a generator and a seed set that conform to the gradient descent process of conventional neural networks. This separates two necessary questions:

1. **Existence:** Does there exist a network within the procedural family that achieves acceptable task loss?
2. **Optimisation:** Can we find such a network with a tractable training process?

Let $\mathcal L(W)$ be task loss and $\mathcal W_\epsilon=\{W:\mathcal L(W)\leq\epsilon\}$. A feasible generated network family must intersect this low-loss set. We do not need it to approximate the precise weight configuration of a particular dense checkpoint. Two very different weight matrices can serve the same task.

This is why the Gaussian reconstruction bound does not itself rule out useful neural networks at fractional bits per effective weight. It limits reconstruction of the specified random source, not the minimal description of a task-solving model. The existence of a good task-solving procedural network still needs to be demonstrated.

A smooth generator with continuous trainable parameters $\theta$ can use conventional backpropagation:

$$\nabla_\theta\mathcal L=J_G(\theta)^T\nabla_W\mathcal L.$$

However, a purely integer/hash-based generator may have no useful notion of a small seed update. Changing one seed bit may alter many weights in unrelated ways. Its encoding search may succeed offline while gradient-based training fails. Discrete seed search, continuous relaxations, straight-through estimators, or a hybrid of continuous and discrete updates are potential solutions, not established guarantees.

If $N$ effective weights are generated from $p$ continuous trainable parameters, the local Jacobian has rank at most $p$. The generated matrix itself can be full rank while its parameter space has far fewer independent locally controllable directions. Do not conflate these concepts. Full matrix rank does not demonstrate equal capacity to a dense network.

### 11.2 Candidate procedural parameterisations

| Representation | Purpose / comparison |
|---|---|
| $W_{ij}=\sum_k a_k G(s_k,i,j)$ | Fixed generated basis and trainable continuous coefficients; SeedLM-like or random-basis baseline |
| $W_{ij}=G(a_i,i,j)$ | Per-row parameters |
| $W_{ij}=G(b_j,i,j)$ | Per-column parameters |
| $W_{ij}=G(a_i,b_j,i,j)$ | Shared nonlinear row/column parameters |
| $W_{ij}=G(a_{\rm global},a_i,b_j,i,j)$ | Hierarchical parameter sharing |
| $W_{ij}=G(\theta,i,j)+R_{ij}$ | Procedural backbone plus a budgeted sparse or structured residual |

For example, a $4096\times4096$ matrix with one 32-bit parameter per row and column uses $8192\times32$ bits = 32 KiB of such parameters, or 0.015625 bpw excluding shared overhead. That is 256 times less than its ideal 8 MiB FP4 representation. The parameters may be reused in cache/registers, but the actual memory traffic depends on the kernel. This is a storage illustration, not a demonstrated model-quality result.

A conventional outer product $W_{ij}=a_i b_j$ has low rank. A nonlinear alternative such as $W_{ij}=\sin(a_i b_j+c_{ij})$, with a deterministic inexpensive coordinate-dependent $c_{ij}$, may be full rank under similar parameter counts. The sine example is a conceptual demonstration, not a proposed efficient GPU decoder. Search cheaper nonlinearities later and compare with low-rank and structured operators. Full rank alone is not a performance argument.

### 11.3 Training methods to investigate

- Fixed random seeds, trainable coefficients: easy gradients but limited span and potentially expensive basis combination.
- Differentiable nonlinear generators: train continuously, then quantise parameters and refine, or use quantisation-aware training.
- Alternating discrete/continuous optimisation: update differentiable parameters by backpropagation; periodically change seeds or programs using guided discrete search.
- Projected gradient methods: take dense weight steps, then project into the representable codebook; can stagnate if the codebook is too coarse. A dense shadow model or straight-through estimator may help but adds training cost.
- Distillation: optimise the student generated network for output/task similarity rather than the teacher's individual weight values.
- Eventually synthesise generators for **learnability** rather than only nearest-codeword reconstruction. The outer objective becomes validation loss after a fixed inner training procedure. This is expensive; do not run a massive nested training search before testing simple hand-designed generators.

### 11.4 Small, controlled learnability experiment

Train a simple MLP on MNIST or Fashion-MNIST (then a small transformer) using:

1. A large dense network as a functional reference.
2. A smaller dense network with matched stored parameter bits.
3. A low-rank layer with matched storage and trainable parameter count.
4. A fixed generated-basis layer.
5. A nonlinear shared procedural layer.
6. A procedural layer with a budgeted residual if useful.

Compare held-out loss/accuracy, stored bits, effective weights, independently trainable parameters, training time, inference compute, and ultimately GPU latency. Run multiple initialisation/training seeds. Start with continuous latents to isolate the capacity question, then quantise and measure the actual representation. A procedural network must offer some advantage over simply using a smaller dense model or a structured operator at the same storage budget. Also compare task quality at matched compute where feasible: increasing effective weights can increase computation even when storage is small.

If these experiments establish a viable parameterisation, incorporate the algebraic and hardware constraints from the earlier work packages and move towards transformer projections. If not, report what failed: representability, optimisation, quantisation, or hardware efficiency.

## 12. Suggested organisation and interfaces

This layout is a proposed organisational scheme, not a verified existing repository tree. Prefer reusing the actual existing structure if it is suitable.

```text
experiments/
  manifests/             # checkpoint, matrix names, splits, seeds
  target_geometry/       # covariance, PCA, norms, neighbour statistics
  rate_distortion/       # explicit VQ, PRNGs, SeedLM, scalar controls
  generator_algebra/     # bit-vector semantics, ANF, real polynomial, Walsh analysis
  synthesis/             # existing enumeration, beam/evolutionary search, multi-output DAG
  encoding/              # exhaustive reference, structured/approximate methods
  evaluation/            # reconstruction, activation error, perplexity
  hardware/              # HIP kernels, ISA inspection, profiles
  training/              # differentiable procedural layers and matched controls
  results/               # versioned configs, raw tables, plots and reports
```

A minimal common experiment interface should distinguish format, program, encoder and cost:

```python
# Conceptual interfaces; adapt to the existing implementation.
format = TileFormat(
    weights_per_tile=N,
    seed_bits=b_seed,
    scale_bits=b_scale,
    selector_bits=b_selector,
    index_layout=layout,
    output_format=output_format,
)
assert format.local_bits == b_seed + b_scale + b_selector

program = GeneratorDAG(nodes, output_nodes, constants)
codebook = generate_reference_codebook(program, format)
properties = analyse_generator(program, format, codebook)

encoded = encode_tiles(tiles, program, format, method="exhaustive")
reconstructed = decode_reference(encoded, program, format)
assert decoded_format_is_bit_exact(reconstructed, program, format, encoded)

metrics = evaluate(tiles, reconstructed, activation_data=None)
report = account_storage_and_cost(encoded, program, format, metrics)
```

Do not interpret the pseudocode as implemented APIs. The reference decoder must precisely specify unsigned wrapping arithmetic, masked shifts, signed interpretation, scale and any rounding. Include deterministic known-answer tests and compare CPU and GPU output for the same encoded data.

Every result row should include enough information to reconstruct the experiment, e.g. model/checkpoint hash, matrix name, source/control type, tile layout, sample counts and split, seed/metadata widths, generator ID and DAG, shared program/constant bytes, codebook entry count, payload and total bpw, reconstruction metric, encode/decode timing and hardware environment. Store codeword collisions, rank and cost features in the same manifest if they were evaluated. Preserve raw metrics, not just formatted tables.

## 13. Success criteria, decision gates and risks

| Question | Evidence needed | Response if it fails |
|---|---|---|
| Can target geometry guide our search? | Reproducible differences from matched controls, or mathematically valid necessary bounds | Retain negative findings; prioritise generic codebook shaping and actual distortion instead of pattern hunting |
| Can algebra improve synthesis? | Safe pruning or measurable improvement in best held-out distortion per search budget | Use the analyser for diagnosis only; retain random/exhaustive search controls |
| Is a fixed budget adequate? | Strong VQ reference and model-level quality | Adjust tile size, use another sharing structure or train within the representation; do not hide extra bits |
| Is encoding tractable? | Speed/memory vs exact encoding, with controlled distortion gap | Keep exhaustive batched encoding if practical, or adopt structured generators/trellises |
| Is a 2-bit codec competitive? | Comparable model loss/accuracy to strong 2-bit methods, and ideally close to a 4-bit baseline, at explicitly accounted rates | Improve calibration/representation or report the quality–cost trade-off; do not use NMSE as a substitute |
| Does it accelerate inference? | Fused kernel and end-to-end performance vs tuned baselines | Reduce generation work, increase useful reuse, change layout, or restrict the claim to memory capacity |
| Can sub-1-bpw models learn? | Task performance vs matched-storage and matched-compute baselines | Analyse capacity vs optimisation; retain 2-bit codec as an independent research track |

Key risks are information scarcity, inappropriate codeword geometry, search overfitting, misleading source-level instruction counts, register pressure, instruction-cache contention, lack of seed locality, and poor transfer from a small LLM to larger models. Another major risk is pursuing a clever mathematical proxy that does not correlate with actual compression or model quality. The safeguards are matched controls, held-out data, strong baselines, exact storage accounting, architecture-specific measurement and explicit decision gates.

## 14. Immediate handover: what to do first

The first implementation session should be small and decisive. It should not attempt all work packages simultaneously.

1. **Recover the existing repository and reproduce the known codec.** Use the exact three-op program and its 12-bit seed + 4-bit exponent format. Verify semantics and reproduce at least a subset of the reported comparisons. If the original code is unavailable, reconstruct a clearly labelled new reference implementation and do not claim exact reproduction.
2. **Run a pilot target-geometry experiment.** Use the existing sampled matrices, 8/16/32/64-weight tiles, original/shuffled/IID controls, covariance spectra, norms and directional statistics. Determine whether the generator needs high affine dimensionality and whether any reproducible positional structure exists.
3. **Run the fixed-budget rate–distortion pilot.** Keep 16 total bits per tile; compare the existing generator, calibrated PRNGs and an explicit VQ reference across those tile sizes. Use smaller codebooks or the existing exponent split if full 65,536-entry VQ is too expensive. Report held-out NMSE, codebook cost and Gaussian references where appropriate.
4. **Implement the exact algebra analyser for 2–8-bit seeds.** Start with AND, parity/Walsh, affine generators and the existing grammar; verify degree, rank and dependency predictions against enumeration. This can proceed while the geometry pilot runs.
5. **In parallel, begin functional and GPU checks.** Evaluate the existing codec's layer outputs and perplexity; establish synthetic MI300X generator throughput and fused MFMA baselines when hardware is available. Do not wait for an optimised codec to establish these constraints.
6. **Review the pilot evidence before expanding the search.** Identify which mathematical predictors are worth pursuing, which generator families are provably too restrictive, how much room remains relative to VQ, and what actual execution budget is available. Then implement multi-output and guided synthesis, followed by structured encoding.

The next report should answer the following specific questions rather than merely present more reconstruction numbers:

- How does the original target geometry differ from shuffled and random controls, if at all?
- How many affine dimensions are necessary for a desired reconstruction quality?
- How does the fixed-budget quality gap between explicit and computational VQ change with tile size?
- Which algebraic constraints can we verify without enumerating all seeds, and which are merely empirical proxies?
- Are the previously discovered codebooks actually useful to the model?
- How many real GPU operations per useful generated weight can we afford, and what is the appropriate sharing/layout policy?

## 15. Summary of the originating intuitions

For continuity, the following captures the user's questions and their implications. These are the conceptual links that should guide future work, not assumptions to accept uncritically.

| Originating intuition / question | Consequence for the research |
|---|---|
| Could we generate weights instead of loading them, to keep matrix engines fed? | Evaluate direct VGPR operand generation and fused MFMA, not just standalone decompression. |
| Is sparsity, low rank or repetition just a special case of a short algorithmic description? | Search a restricted feasible program space; separate description complexity from runtime cost. |
| Do we really need to find natural patterns, or can a generator just happen to coincide with the target? | Search representable outputs across feasible algorithms and seeds; retain random controls to distinguish coincidence from source structure. |
| Is it like aligning ten 3D points generated by a PRNG to the original ten points? | Treat ordered weight sequences as single high-dimensional vectors; optimise corresponding coordinates. |
| Is the operation/program space effectively infinite? | It is combinatorially enormous even with a small ISA; need guided synthesis and equivalence pruning. |
| Should we fix seed length and explore algorithms instead of enlarging the seed? | Hold tile-local information constant; alter shared codebook geometry under a hardware budget. |
| Does this just move information from data into instructions? A 64-bit instruction is four FP16 weights. | Count all model-dependent program bits; separate instruction and data traffic on-chip, and require sharing to amortise the program. |
| Can sequential instruction streams hide latency better than dense weight loads? | Possible overlap, but not free throughput; test dependence depth, issue, cache, register pressure and fused supply rate. |
| Is this a generalisation of SparseRT with adjustable specialisation? | Explicitly vary tiles per program and weights per seed, and distinguish storage amortisation from computation amortisation. |
| What can we infer from a one-bit AND generator producing `010101...`? | Derive seed/index dependence, effective polynomial bases and provable periodicity. |
| Does algebra imply that some instruction combinations can never work? | Use field-specific invariants, degree and affine dimension bounds, and source covariance to rule out restricted families. |
| If every tile is a point, should one generator's basis cover the cloud of all tiles? | Computational VQ: optimise reachable codewords, not just rank or basis span. |
| Why characterise the target point cloud first? | Guide synthesis, establish necessary expressivity, decide whether source geometry or the bitrate is limiting, and use VQ references as controls/teachers. |
| Can we find acceptable seeds without brute force? | Co-design generator structure and encoder; retain exact search as a reference and exploit orthogonality, trellises or other structure where justified. |
| Does NMSE measure reconstruction rather than neural network performance? | Evaluate activation error, perplexity and downstream tasks separately. |
| Does a good solution need to conform to gradient descent? | Separate representability from trainability; explore differentiable and hybrid procedural parameterisations. |
| Is a 2-bit codec valuable even if the grander goal fails? | Yes, if competitive in functionality and execution; existing 2-bit methods mean the novelty must lie in quality, synthesis, encoding or hardware efficiency. |

## 16. Final research framing

The project is best understood as **hardware-constrained synthesis of computational codebooks, with mathematically guided generator–encoder co-design**, motivated by replacing expensive movement of neural network weights with reusable computation.

For the post-training problem, we want a program that generates a limited set of useful codewords from a fixed tile-local information budget. Its program is shared widely enough to amortise model-specific description, and its execution must be cheap enough to sustain the relevant matrix pipeline. We want to use the geometry of the target tiles and the algebra of the generating instructions to guide that synthesis and, where possible, construct efficient encoders.

For the longer-term problem, we want to determine whether neural networks can be trained to solve their tasks with far fewer independently stored parameters than effective weights. If this can be done without costly reconstruction or operator evaluation, it may substantially reduce inference memory traffic. If it cannot, the limits and the practical 2-bit computational codec remain valuable results.

The guiding principle is that each experiment should advance one of these objectives or establish a limitation that justifies changing direction. No increase in program complexity, apparent rank, seed space, or reduction in nominal bpw is itself a success unless it survives storage accounting, model-quality evaluation and the relevant hardware comparison.
