# Stitching Artifact Attenuation in Sliding-Window Inner Tiling

We show that the sliding-window inner-tiled stitching strategy reduces the
per-pixel magnitude of any tile-boundary discontinuity by an exact factor of
$1/K$ relative to classical inner tiling, where
$K = (P - \text{overlap})/s$ is the per-pixel coverage of the strategy. The
result is a pointwise algebraic identity; it requires no statistical
assumption on the model and follows purely from the geometry of the $K$-fold
averaging used at stitching time.

This note is intended as a companion to
[`sliding_window_inner_tiling.md`](sliding_window_inner_tiling.md), which
describes the patching geometry and phantom-tile boundary handling in detail.

---

## 1. Setup

We work along a single spatial axis of length $N$. The multi-axis case
follows by Cartesian product (see §6).

The strategy is parameterised by three positive integers $(P, s,
\text{overlap})$ with $\text{overlap}$ even and $s$ dividing
$P - \text{overlap}$. Define

$$
M \;:=\; \text{overlap}/2, \qquad K \;:=\; \frac{P - \text{overlap}}{s} \;=\; \frac{P - 2M}{s}.
$$

A tile placed at sliding-window position $i$ takes an input patch from image
coordinates $[i, i + P)$ and produces a same-shape prediction. We write
$y_i(p)$ for that prediction evaluated at image coordinate $p$. The tile's
**kept region** is the inner interval

$$
\big[i + M,\; i + P - M\big), \qquad \text{width } P - 2M.
$$

Tiles are placed at every multiple of $s$. Image boundaries are handled by
phantom tiles whose model inputs are snapped to the nearest in-image
coordinate but whose output crops are progressively shifted (see SW tiling
note §4). The phantom mechanism guarantees that the **contributor set**

$$
C(p) \;:=\; \big\{\, i \in s\mathbb{Z} \;:\; i + M \le p < i + P - M \,\big\}
$$

has cardinality exactly $K$ for every pixel $p \in [0, N)$.

The stitched output is the arithmetic mean of the contributing tile
predictions:

$$
v(p) \;=\; \frac{1}{K}\sum_{i \in C(p)} y_i(p).
$$

## 2. Contributor-set transitions

We first describe how $C(p)$ relates to $C(p+1)$.

Unfolding the definition, $i \in C(p)$ iff $p - P + M + 1 \le i \le p - M$,
restricted to multiples of $s$. Comparing the integer intervals for $p$ and
$p+1$:

- the set difference $C(p+1) \setminus C(p)$ contains at most one element,
  the multiple of $s$ equal to $p - M + 1$ (if it is a multiple of $s$);
- the set difference $C(p) \setminus C(p+1)$ contains at most the multiple of
  $s$ equal to $p - P + M + 1$ (if it is a multiple of $s$).

The first event happens at residues $p \equiv M - 1 \pmod{s}$, the second at
residues $p \equiv P - M - 1 \pmod{s}$. Using $P - 2M = Ks$, the two
residues coincide:

$$
P - M - 1 \;=\; M + Ks - 1 \;\equiv\; M - 1 \pmod{s}.
$$

So the gain event and the loss event happen at the same $p$. We obtain two
disjoint regimes:

**(T1) Non-seam steps.** For every $p$ with $p \not\equiv M - 1 \pmod{s}$,

$$
C(p+1) \;=\; C(p).
$$

**(T2) Seam steps.** For every $p$ with $p \equiv M - 1 \pmod{s}$,

$$
C(p+1) \;=\; \big(C(p) \setminus \{i_\text{out}\}\big) \cup \{i_\text{in}\}
\qquad \text{with} \qquad
i_\text{out} = p - P + M + 1, \quad i_\text{in} = p - M + 1.
$$

In particular, exactly one in every $s$ consecutive inter-pixel steps is a
seam step. The seam frequency is $1/s$, not $1/(P - \text{overlap})$.

## 3. The cross-seam gradient decomposition

Let $\delta_i(p) := y_i(p+1) - y_i(p)$ denote the **single-tile gradient**:
the finite-difference gradient computed inside the prediction of one tile,
with no involvement of any neighbouring tile.

### 3.1 Non-seam steps

When $C(p+1) = C(p)$,

$$
g_\text{mid}(p) \;:=\; v(p+1) - v(p)
\;=\; \frac{1}{K} \sum_{i \in C(p)} \delta_i(p).
$$

This is the arithmetic mean of $K$ single-tile gradients drawn from the
natural in-tile gradient distribution. No cross-tile term appears.

### 3.2 Seam steps

Write $C(p) = S \cup \{i_\text{out}\}$ and $C(p+1) = S \cup \{i_\text{in}\}$
with the shared set $S$ of size $K - 1$. Then

$$
v(p)
= \tfrac{1}{K}\!\left( y_{i_\text{out}}(p) + \sum_{i \in S} y_i(p) \right),
\quad
v(p+1)
= \tfrac{1}{K}\!\left( y_{i_\text{in}}(p+1) + \sum_{i \in S} y_i(p+1) \right).
$$

Taking the difference,

$$
g_\text{edge}(p) \;:=\; v(p+1) - v(p)
\;=\; \frac{1}{K}\!\left[\, y_{i_\text{in}}(p+1) - y_{i_\text{out}}(p) + \sum_{i \in S} \delta_i(p) \,\right].
$$

The bracket contains $K - 1$ natural single-tile gradients and one **swap
term** $y_{i_\text{in}}(p+1) - y_{i_\text{out}}(p)$. We split the swap term
by adding and subtracting $y_{i_\text{out}}(p+1)$:

$$
y_{i_\text{in}}(p+1) - y_{i_\text{out}}(p)
\;=\; \underbrace{\big( y_{i_\text{in}}(p+1) - y_{i_\text{out}}(p+1) \big)}_{=:\; \Lambda(p+1)}
\;+\; \underbrace{\big( y_{i_\text{out}}(p+1) - y_{i_\text{out}}(p) \big)}_{=\; \delta_{i_\text{out}}(p)}.
$$

The first term $\Lambda(p+1)$ is the **pure tile mismatch** at pixel $p+1$:
it compares the predictions of two **different** tiles at the **same** image
coordinate. By construction, $\Lambda$ is the local signature of a
tile-boundary discontinuity — what we mean by a "stitching artifact." The
second term is a natural single-tile gradient and folds back into the sum.

Substituting and regrouping,

$$
\boxed{\quad
g_\text{edge}(p) \;=\; \underbrace{\frac{1}{K} \sum_{i \in C(p)} \delta_i(p)}_{\text{same form as } g_\text{mid}}
\;+\; \frac{\Lambda(p+1)}{K}.
\quad}
\tag{$\star$}
$$

## 4. Interpretation: $1/K$ attenuation by construction

The decomposition $(\star)$ has four immediate consequences.

**(C1) Exact pointwise attenuation.** The contribution of the tile-vs-tile
mismatch $\Lambda(p+1)$ to the visible cross-seam gradient in the stitched
image is $\Lambda(p+1)/K$. This is an algebraic identity: it holds
pixel-by-pixel, for every realisation of the stochastic forward passes, and
requires no assumption on the model beyond the determinism of each individual
forward pass.

**(C2) Geometric origin.** The factor $1/K$ arises purely from the $K$-fold
averaging $v(p) = (1/K) \sum_{i \in C(p)} y_i(p)$. At a seam step exactly
one of the $K$ contributors is swapped; the remaining $K - 1$ shared
contributors are present in both $v(p)$ and $v(p+1)$ and contribute only
their own natural gradients $\delta_i$ to the difference. The swap accounts
for the only cross-tile term, and it inherits the $1/K$ factor of the
averaging.

**(C3) Sign and spatial structure are preserved.** The mismatch $\Lambda$
enters $(\star)$ additively and undistorted in space. The strategy does not
transform the artifact spatially or change its sign — it only scales it down
by $1/K$.

**(C4) Hyperparameter-controlled.** The attenuation factor $K$ is set at
prediction time via $K = (P - \text{overlap})/s$. Halving $s$ doubles $K$
and halves the per-pixel seam discontinuity in the stitched output, at the
cost of $K$ times more forward passes per axis.

## 5. Comparison with classical inner tiling

Classical inner tiling — the strategy in which each pixel is covered by a
single tile and adjacent kept regions are pasted side by side — is the
degenerate case $K = 1$, i.e. $s = P - \text{overlap}$. The decomposition
$(\star)$ specialises to

$$
g_\text{edge}^{(K=1)}(p) \;=\; \delta_{i_\text{out}}(p) + \Lambda(p+1),
$$

in which the mismatch enters with weight $1$. Summarising the comparison at
fixed $(P, \text{overlap})$ and fixed mismatch $\Lambda$:

| | Classical inner tiling ($K = 1$) | Sliding-window inner tiling ($K \ge 1$) |
|---|---|---|
| Seam spacing | $P - \text{overlap}$ | $s = (P - \text{overlap})/K$ |
| Per-pixel coverage | $1$ | $K$ |
| Mismatch contribution to seam gradient | $\Lambda$ | $\Lambda / K$ |
| Background of seam gradient | one $\delta_i$ | mean of $K$ $\delta_i$'s |

The seam grid is denser by a factor $K$ in the sliding-window case (seams
every $s$ pixels rather than every $Ks$ pixels), but each seam carries a
$1/K$-attenuated discontinuity, with the unattenuated mismatch $\Lambda$
unchanged for fixed $(P, \text{overlap})$.

If the single-tile gradients $\delta_i$ are modelled as i.i.d. with variance
$\sigma^2$, the variance of the non-seam-gradient population is $\sigma^2$
for $K = 1$ and $\sigma^2/K$ for general $K$. This is a consequence of the
same $K$-averaging that attenuates the mismatch and applies to both the
artifact-bearing and the artifact-free gradient populations equally. We
emphasise that the $1/K$ factor in $(\star)$ is independent of any such
statistical model: it is an exact algebraic factor on $\Lambda$ itself.

## 6. Multi-axis generalisation

For $d$ spatial axes, tiles are generated as the Cartesian product of the
per-axis sliding-window grids, and the per-pixel coverage is $K^d$. A step
along axis $\alpha$ from pixel $\mathbf{p}$ to $\mathbf{p} + e_\alpha$ swaps
$K^{d-1}$ contributors (those with axis-$\alpha$ index $i_\text{out}$) for
another $K^{d-1}$ contributors (axis-$\alpha$ index $i_\text{in}$), while
keeping $(K - 1) K^{d-1}$ contributors shared.

Repeating the §3 derivation gives

$$
g_\text{edge}^{(\alpha)}(\mathbf{p})
\;=\; \frac{1}{K^d} \sum_{\mathbf{i} \in C(\mathbf{p})} \delta_\mathbf{i}^{(\alpha)}(\mathbf{p})
\;+\; \frac{1}{K} \cdot \overline{\Lambda}(\mathbf{p} + e_\alpha),
\quad
\overline{\Lambda}(\mathbf{q}) \;:=\; \frac{1}{K^{d-1}} \sum_{\mathbf{k}} \Lambda_\mathbf{k}(\mathbf{q}),
$$

where the average $\overline{\Lambda}$ is taken over the $K^{d-1}$
configurations of the orthogonal axes' contributor indices. Two distinct
mechanisms reduce the visible artifact in higher dimensions:

- **Per-axis attenuation by $1/K$**, as in 1D, due to the swap involving only
  the axis-$\alpha$ direction.
- **Orthogonal averaging of $K^{d-1}$ co-mismatches** $\Lambda_\mathbf{k}$,
  which reduces the variance of the artifact term by a further factor
  $1/K^{d-1}$ under independence assumptions.

The headline factor — the deterministic, distribution-free attenuation — is
the per-axis $1/K$.

## 7. Boundary regime

The decomposition $(\star)$ is exact wherever the contributor set has size
$K$ and $i_\text{out}$, $i_\text{in}$ correspond to **distinct** model
inputs. Uniform $|C(p)| = K$ is delivered by the phantom-tile boundary
handling. Distinctness of model inputs requires only that the two swapped
positions be mapped to different in-image input patches.

Phantom tiles near the image edge share a single boundary-snapped model
input across multiple sliding-window positions: every left phantom runs the
model at $\text{actual\_coord} = 0$, every right phantom at
$\text{actual\_coord} = N - P$. Inside the boundary band of width $P - M$,
a seam step may swap two contributors that share the same model input. In
that case

$$
\Lambda(p+1) \;=\; y_{i_\text{in}}(p+1) - y_{i_\text{out}}(p+1)
$$

is the difference between two **independent stochastic draws of the same
posterior**, with $\mathbb{E}[\Lambda] = 0$. The $1/K$ attenuation in
$(\star)$ still holds, but the quantity being attenuated is now pure
posterior noise rather than a cross-tile discontinuity. For images with
$N \gg P$, the interior regime covers the bulk of the field of view and
$(\star)$ describes a genuine reduction of tile-boundary artifacts; the
$O(P)$-wide boundary band at each edge requires the separate interpretation
above.

## 8. Summary

The sliding-window inner-tiled stitching strategy realises an exact
pointwise identity

$$
g_\text{edge}(p) \;-\; g_\text{mid}'(p) \;=\; \frac{\Lambda(p+1)}{K},
\qquad K \;=\; \frac{P - \text{overlap}}{s},
$$

in which $g_\text{edge}$ is the finite-difference gradient at a seam step in
the stitched image, $g_\text{mid}'$ has the same structural form as a
non-seam gradient (mean of $K$ natural single-tile gradients), and
$\Lambda$ is the tile-vs-tile mismatch at the relevant image pixel. The
artifact's contribution to the stitched output is therefore $1/K$ of its
classical-inner-tiling counterpart, with the reduction following from
geometry alone and controllable via the stride hyperparameter $s$.
