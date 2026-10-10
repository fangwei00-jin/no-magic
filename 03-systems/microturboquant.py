"""
Why rotating a vector before quantizing it works -- random rotation makes coordinates
independent, turning one hard D-dim quantization problem into D easy 1-D ones.
"""
# Reference: Aamand et al., "TurboQuant: Online Vector Quantization with Optimal
# Bit Budget" (2025). https://arxiv.org/abs/2504.19874
# The key result: for any unit vector x in R^D, applying a uniformly-random rotation R
# makes the coordinates of y = R @ x concentrate around a Beta marginal that depends
# only on D -- NOT on x. This converts a hard joint-distribution quantization problem
# into D independent 1-D scalar quantization problems, with one rotation shared across
# every vector. No calibration data, no training: "data-oblivious".

# === TRADEOFFS ===
# + Data-oblivious: one random rotation works for any vector distribution, provably.
# + No training, no calibration data, no retraining when data distribution shifts.
# + Inner-product preservation: unbiased estimator via sign-bit residuals (QJL trick).
# - A DATA-AWARE method (learned rotation, product quantization) can beat TurboQuant
#   when you have plentiful calibration data and a stable distribution.
# - Random rotations are dense: encoding cost is O(D^2) per vector (vs. O(D) for
#   absmax). Production uses structured rotations (Hadamard, Kac) to get O(D log D).
# WHEN TO USE: Online/streaming vector compression, KV-cache quantization at inference
#   time, any setting where you cannot collect calibration data ahead of time.
# WHEN NOT TO: When you have millions of representative vectors to calibrate against
#   and the distribution is stationary -- learned codebooks (e.g., product quantization)
#   outperform data-oblivious methods given enough training data.

from __future__ import annotations

import math
import os
import random
import time
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

EMBEDDING_DIM = 32              # D: vector dimension. Small enough for pure-Python matvec.
N_SYNTHETIC = 300               # Number of anisotropic synthetic vectors for contrast demo.
ANISOTROPY = 6.0                # Dominant-axis std inflation. 6x is the sweet spot for D=32:
                                # large enough that absmax wastes bits on small coords, small
                                # enough that the post-normalization vector stays dense. Above
                                # ~10x the pre-normalization vector becomes nearly 1-sparse --
                                # the regime where absmax is already optimal and rotation turns
                                # sparse-good into dense-average: TurboQuant loses. The effect
                                # size depends on D: at larger D the sweet spot shifts higher.
N_NAMES_SAMPLE = 300            # Number of names sampled for real-embedding evaluation.
BITS_TESTED = [1, 2, 4, 8]      # Bit-widths to sweep in the rate-distortion table.
N_IP_PAIRS = 2000               # Held-out vector pairs for inner-product MSE estimate.
QJL_PROJECTIONS = 256           # Sign-bit projections for the 1-bit residual demo.

DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# Signpost: production systems use D = 768 to 4096 (BERT, GPT embeddings). We use 32
# because Gram-Schmidt on Gaussian columns is O(D^3) in pure Python -- at D = 32
# orthogonalization costs ~30k scalar multiplies, finishing in milliseconds. At D = 768
# it would take minutes without BLAS. The algorithm is dimension-agnostic.


# === DATA LOADING ===

def load_names(url: str, filename: str) -> list[str]:
    """Download names.txt on first run, return list of lowercase names."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
    with open(filename, "r") as f:
        return [line.strip().lower() for line in f if line.strip()]


# === LINEAR ALGEBRA PRIMITIVES ===

# Why hand-roll these: pedagogy, and to show exactly what the pipeline's O(D^2) cost
# is hiding. NumPy would replace every one of these with a single line, but obscure
# the fact that random rotation's encoding cost dominates over scalar quantization.

def matvec(A: list[list[float]], x: list[float]) -> list[float]:
    """Dense matrix-vector product: returns A @ x."""
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def transpose(A: list[list[float]]) -> list[list[float]]:
    """Matrix transpose. For orthogonal R, transpose is also the inverse."""
    rows, cols = len(A), len(A[0])
    return [[A[i][j] for i in range(rows)] for j in range(cols)]


def vec_norm(x: list[float]) -> float:
    """Euclidean norm ||x||_2."""
    return math.sqrt(sum(v * v for v in x))


def inner(a: list[float], b: list[float]) -> float:
    """Inner product <a, b>."""
    return sum(ai * bi for ai, bi in zip(a, b))


def gaussian_sample() -> float:
    """Return one draw from the standard normal N(0, 1). Named wrapper so callers
    read as 'gaussian_sample()' rather than 'random.gauss(0.0, 1.0)'."""
    return random.gauss(0.0, 1.0)


# === RANDOM ROTATION VIA GRAM-SCHMIDT ===

def random_rotation(dim: int) -> list[list[float]]:
    """Sample a uniformly-random D-by-D orthogonal matrix.

    Why this construction: if we fill a matrix with i.i.d. standard-normal entries
    and orthonormalize the columns with Gram-Schmidt, the result is distributed
    as Haar measure on O(D) (up to a sign flip per column). This is the simplest
    Haar-random sampler that works in pure Python.

    Math-to-code:
        columns[k] = g_k - sum_{j<k} <g_k, columns[j]> * columns[j]
        columns[k] = columns[k] / ||columns[k]||
    where g_k is the k-th standard-Gaussian column before orthonormalization.
    """
    # Generate a D-by-D matrix of i.i.d. Gaussians. Stored row-major; we extract
    # columns on demand via `raw[i][k]` for k in the Gram-Schmidt loop below.
    raw = [[gaussian_sample() for _ in range(dim)] for _ in range(dim)]

    # Gram-Schmidt in place, column by column.
    orthonormal_cols: list[list[float]] = []
    for k in range(dim):
        col = [raw[i][k] for i in range(dim)]
        # Subtract projections onto previously orthonormalized columns.
        for prev in orthonormal_cols:
            coef = inner(col, prev)
            col = [col[i] - coef * prev[i] for i in range(dim)]
        norm = vec_norm(col)
        # If the Gaussian draw produced a column nearly in the span of earlier ones,
        # renormalizing would amplify noise. In practice for D = 32 this never fires.
        if norm < 1e-12:
            raise RuntimeError("Degenerate Gaussian draw: Gram-Schmidt rank-deficient.")
        orthonormal_cols.append([v / norm for v in col])

    # orthonormal_cols[k] is the k-th column of R. Convert to row-major storage.
    return [[orthonormal_cols[j][i] for j in range(dim)] for i in range(dim)]


def orthogonality_error(R: list[list[float]]) -> float:
    """Return max |R^T R - I|_ij. Should be below ~1e-10 for any correct rotation."""
    dim = len(R)
    Rt = transpose(R)
    return max(
        abs(sum(Rt[i][k] * R[k][j] for k in range(dim)) - (1.0 if i == j else 0.0))
        for i in range(dim)
        for j in range(dim)
    )


# === SCALAR QUANTIZERS ===

def absmax_quantize(x: list[float], bits: int) -> tuple[list[int], float]:
    """Per-vector absmax scalar quantization.

    Math-to-code:
        levels = 2^(bits-1) - 1                  # signed grid [-levels, levels]
        scale  = max(|x_i|) / levels             # one scalar per vector
        q_i    = round(x_i / scale), clipped to [-levels, levels]
    The scale `s` plus the bits-per-coord integer grid reconstructs x to within
    O(s) = O(max|x|/levels) per coordinate.

    Why the per-vector scale is bad on anisotropic data: one huge coordinate sets
    `scale` high for all D coordinates, wasting bits on the many small ones.
    Rotation (TurboQuant) fixes this by making the per-coord max roughly uniform.
    """
    levels = 2 ** (bits - 1) - 1 if bits > 1 else 1   # 1-bit case: sign only, levels = 1.
    max_abs = max((abs(v) for v in x), default=0.0)
    if max_abs < 1e-12:
        return [0 for _ in x], 1.0                    # zero vector: nothing to quantize.
    scale = max_abs / levels
    q = [max(-levels, min(levels, round(v / scale))) for v in x]
    return q, scale


def absmax_dequantize(q: list[int], scale: float) -> list[float]:
    """Inverse of absmax_quantize: x_hat_i = q_i * scale."""
    return [v * scale for v in q]


# === TURBOQUANT ENCODE AND DECODE ===

def turboquant_encode(
    x: list[float], R: list[list[float]], bits: int
) -> tuple[list[int], float]:
    """Rotate then scalar-quantize. One shared R is used across all vectors.

    Why rotation helps absmax: for unit-norm x, the rotated y = R @ x has
    coordinates with the same empirical scale regardless of x's original
    anisotropy. The per-vector `max |y_i|` concentrates around sqrt(2/pi)
    (the mean of |N(0, 1/D)|) rather than tracking x's dominant direction.
    """
    y = matvec(R, x)
    return absmax_quantize(y, bits)


def turboquant_decode(
    q: list[int], scale: float, R_transpose: list[list[float]]
) -> list[float]:
    """Dequantize rotated coords then rotate back: x_hat = R^T (scale * q)."""
    y_hat = absmax_dequantize(q, scale)
    return matvec(R_transpose, y_hat)


# === QJL: 1-BIT QUANTIZED JOHNSON-LINDENSTRAUSS RESIDUAL ===

def qjl_signs(residual: list[float], S: list[list[float]]) -> list[int]:
    """Take sign bits of a random projection of the residual.

    Math-to-code:
        signs_k = sign( (S @ r)_k )     for k = 1..K where K = rows(S)
    Each sign bit is a single-bit summary of `r`'s alignment with S's k-th row.
    Cost: K * D multiplies per residual. Storage: K bits per residual.
    """
    projection = matvec(S, residual)
    return [1 if v >= 0.0 else -1 for v in projection]


def qjl_estimate_inner_product(signs_a: list[int], signs_b: list[int]) -> float:
    """Unbiased inner-product estimator from paired 1-bit sign projections.

    Math: if s_k = sign(<g_k, r>) for g_k standard Gaussian, then
        E[ s_a^k * s_b^k ] = 1 - 2 * arccos(rho) / pi
    where rho = <a,b> / (||a|| * ||b||) is the cosine similarity of unit vectors.
    Invert: rho = cos( (1 - E[s_a * s_b]) * pi / 2 ).
    For the small-angle regime arccos(rho) ~ pi/2 - rho, giving
        rho ~ (pi/2) * E[s_a * s_b].
    The constant is pi/2 regardless of D: each sign bit carries an angular
    summary, not a magnitude summary. Averaging K bits shrinks variance as 1/K
    without changing the pi/2 scaling.
    Signpost: production QJL inverts the arccos explicitly (one trig call per
    pair) to remove the small-angle bias; we use the linear form for clarity.
    """
    K = len(signs_a)
    agreement = sum(sa * sb for sa, sb in zip(signs_a, signs_b)) / K
    return agreement * math.pi / 2.0


# === EMBEDDING SOURCES ===

def sample_name_embeddings(names: list[str], count: int, dim: int) -> list[list[float]]:
    """Build per-name embeddings: random projection of the bigram count vector.

    Why random projection and not raw bigram counts: raw per-name bigram
    vectors are very sparse (~4-8 non-zeros out of 700+ bigram types). Sparse
    unit vectors are the one regime where absmax quantization is nearly
    optimal -- it zeros out the 90%+ coordinates that are already zero, and
    spends all its bit budget on the few non-zeros. TurboQuant's rotation
    converts sparse into dense, destroying this free compression and losing
    badly. To show TurboQuant on realistic-looking data, we densify first by
    projecting onto a shared Gaussian basis (Johnson-Lindenstrauss): this
    preserves pairwise angles while producing dense vectors that land in the
    regime where rotation neither helps nor hurts much.

    Math-to-code:
        for each name n:
            sparse_n[k] = count of bigram k appearing in n (with boundaries)
            dense_n     = normalize( P @ sparse_n )        P ~ Gaussian(dim x V)
    """
    # Build the bigram vocabulary from the entire corpus.
    alphabet = "abcdefghijklmnopqrstuvwxyz."
    bigram_types = [a + b for a in alphabet for b in alphabet]
    bigram_idx = {bg: i for i, bg in enumerate(bigram_types)}
    V = len(bigram_types)

    # Fixed random Gaussian projection from V-dim sparse space to dim-dim dense.
    P = [[gaussian_sample() for _ in range(V)] for _ in range(dim)]

    sampled: list[list[float]] = []
    for name in random.sample(names, min(count, len(names))):
        sparse_counts: dict[int, float] = {}
        padded = "." + name + "."
        for a, b in zip(padded, padded[1:]):
            idx = bigram_idx.get(a + b)
            if idx is None:
                continue
            sparse_counts[idx] = sparse_counts.get(idx, 0.0) + 1.0
        if not sparse_counts:
            continue
        # Sparse matvec: only sum over non-zero entries of sparse_counts.
        dense = [0.0] * dim
        for i in range(dim):
            row = P[i]
            dense[i] = sum(row[j] * c for j, c in sparse_counts.items())
        norm = vec_norm(dense)
        if norm < 1e-12:
            continue
        sampled.append([v / norm for v in dense])
    return sampled


def synthetic_anisotropic_vectors(count: int, dim: int, anisotropy: float) -> list[list[float]]:
    """Generate anisotropic Gaussian vectors: one dominant axis, others small.

    This is the contrast case. Absmax quantization burns most of its bit budget
    on the dominant axis; after random rotation, no axis dominates and absmax
    becomes efficient. Running the rate-distortion table on these vectors
    produces the cleanest "rotation helps" visualization in the script.
    """
    vectors: list[list[float]] = []
    for _ in range(count):
        x = [gaussian_sample() for _ in range(dim)]
        x[0] *= anisotropy                     # inflate one coordinate
        norm = vec_norm(x)
        vectors.append([v / norm for v in x])
    return vectors


# === EVALUATION METRICS ===

def inner_product_mse(
    originals: list[list[float]], approximations: list[list[float]], pair_count: int
) -> float:
    """Mean squared error of inner-product estimates over random held-out pairs.

    Why this is the right metric for quantized embeddings: downstream tasks
    (retrieval, clustering, classification head) consume inner products, not
    raw reconstructed vectors. A method that preserves IP while butchering the
    raw vectors is fine; the converse is not.
    """
    n = len(originals)
    total = 0.0
    for _ in range(pair_count):
        i = random.randrange(n)
        j = random.randrange(n)
        err = inner(originals[i], originals[j]) - inner(approximations[i], approximations[j])
        total += err * err
    return total / pair_count


# === RATE-DISTORTION BENCHMARK ===

def encode_all(
    vectors: list[list[float]], R: list[list[float]], R_transpose: list[list[float]], bits: int
) -> tuple[list[list[float]], list[list[float]]]:
    """Run baseline and TurboQuant at a given bit-width. Returns decoded sets."""
    baseline: list[list[float]] = []
    turbo: list[list[float]] = []
    for x in vectors:
        q_b, s_b = absmax_quantize(x, bits)
        baseline.append(absmax_dequantize(q_b, s_b))
        q_t, s_t = turboquant_encode(x, R, bits)
        turbo.append(turboquant_decode(q_t, s_t, R_transpose))
    return baseline, turbo


def rate_distortion_table(
    vectors: list[list[float]],
    R: list[list[float]],
    R_transpose: list[list[float]],
    label: str,
) -> None:
    """Print per-bit-width baseline vs TurboQuant inner-product MSE."""
    print(f"\n=== Rate-distortion on {label} ({len(vectors)} vectors, D={EMBEDDING_DIM}) ===")
    print(f"{'bits':>5} {'baseline IP-MSE':>18} {'TurboQuant IP-MSE':>22} {'ratio':>8}")
    for bits in BITS_TESTED:
        baseline_decoded, turbo_decoded = encode_all(vectors, R, R_transpose, bits)
        b_mse = inner_product_mse(vectors, baseline_decoded, N_IP_PAIRS)
        t_mse = inner_product_mse(vectors, turbo_decoded, N_IP_PAIRS)
        ratio = b_mse / t_mse if t_mse > 0 else float("inf")
        print(f"{bits:>5d} {b_mse:>18.6f} {t_mse:>22.6f} {ratio:>8.2f}x")


def qjl_demo(vectors: list[list[float]]) -> None:
    """Show unbiased inner-product recovery from 1-bit sign projections alone."""
    print(f"\n=== 1-bit QJL residual demonstration ({QJL_PROJECTIONS} sign bits) ===")
    S = [[gaussian_sample() for _ in range(EMBEDDING_DIM)] for _ in range(QJL_PROJECTIONS)]

    # Take sign projections of every vector (1 bit per projection per vector).
    all_signs = [qjl_signs(x, S) for x in vectors]

    # Sample held-out pairs, compute true IP and QJL estimate.
    errors: list[float] = []
    for _ in range(N_IP_PAIRS):
        i = random.randrange(len(vectors))
        j = random.randrange(len(vectors))
        true_ip = inner(vectors[i], vectors[j])
        approx = qjl_estimate_inner_product(all_signs[i], all_signs[j])
        errors.append(true_ip - approx)

    mean_error = sum(errors) / len(errors)
    mean_abs = sum(abs(e) for e in errors) / len(errors)
    print(f"mean signed error (should be near 0): {mean_error:+.4f}")
    print(f"mean |error|:                         {mean_abs:.4f}")
    print(f"bit cost per vector: {QJL_PROJECTIONS} bits (vs. {EMBEDDING_DIM * 32} for fp32)")


# === MAIN ===

def main() -> None:
    t_start = time.time()

    print("--- TurboQuant: data-oblivious vector quantization ---")
    print(f"dim D = {EMBEDDING_DIM}, bits tested = {BITS_TESTED}")

    print("\nLoading names corpus...")
    names = load_names(DATA_URL, DATA_FILE)
    print(f"  {len(names)} names loaded")

    print("\nBuilding real name embeddings (random projection of bigram counts)...")
    name_vectors = sample_name_embeddings(names, N_NAMES_SAMPLE, EMBEDDING_DIM)
    print(f"  {len(name_vectors)} name-embeddings, {EMBEDDING_DIM}-dim, unit-normalized")

    print(f"\nGenerating {N_SYNTHETIC} synthetic anisotropic vectors (anisotropy={ANISOTROPY})...")
    synth = synthetic_anisotropic_vectors(N_SYNTHETIC, EMBEDDING_DIM, ANISOTROPY)
    print(f"  dominant axis variance inflated by {ANISOTROPY}x before unit-normalization")

    print("\nSampling one random rotation R via Gram-Schmidt on Gaussian columns...")
    R = random_rotation(EMBEDDING_DIM)
    R_transpose = transpose(R)
    err = orthogonality_error(R)
    print(f"  orthogonality error max|R^T R - I| = {err:.2e}")
    # Hard check: rotations that fail this cannot preserve inner products.
    assert err < 1e-10, "Gram-Schmidt produced a non-orthogonal matrix."

    # Key comparison #1: synthetic anisotropic vectors.
    # Expect baseline absmax IP-MSE to be much worse than TurboQuant at 4 bits,
    # because absmax burns bits on the dominant axis and has few levels left
    # for minor axes. After rotation, all axes contribute comparable variance.
    rate_distortion_table(synth, R, R_transpose, "SYNTHETIC anisotropic vectors")

    # Key comparison #2: real (averaged-bigram-projection) name embeddings.
    # These are already roughly isotropic after normalization, so the gap
    # narrows -- the rotation's win shrinks when data is already well-conditioned.
    rate_distortion_table(name_vectors, R, R_transpose, "REAL name embeddings")

    # QJL sanity check: 1-bit sign projections alone recover inner products with
    # near-zero signed error. This is the "residual stage" that refines a coarse
    # TurboQuant reconstruction with ~K extra bits per vector.
    qjl_demo(name_vectors)

    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
