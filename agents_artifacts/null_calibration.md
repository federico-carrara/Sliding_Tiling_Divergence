# Null calibration of the per-tile permutation test — why clean GT rejects at ~5%

**Context.** Running the per-tile gradient permutation test on a clean GT image
(no stitching, full frame) yields a fraction of rejected tiles at α=0.05 that
≈ 5%. This note records *why*, so we stop re-deriving it.

---

## 1. The ~5% is by construction, not an emergent discovery

A permutation test sets its p-value to the rank of the observed statistic within
the permutation-null distribution (`permutation.py`):

```python
p = (1.0 + #{T_null >= T_obs}) / (1.0 + R)   # Phipson–Smyth
```

The observed seam-vs-control labelling is just **one** of the possible block
relabellings. Under the null hypothesis of **exchangeability** — seam-blocks and
control-blocks drawn from the same distribution, labels arbitrary — `T_obs` is
statistically interchangeable with the `T_null` values, so its rank is uniform on
[0, 1], the p-value is uniform, and:

> **P(p ≤ α) = α.**

This is the *defining validity property* of a permutation test: it controls the
type-I error rate at exactly α for any exchangeable input. Pure noise, a clean
GT, anything where the two label groups are genuinely exchangeable → ~5% at
α=0.05. So the result is **not** the method "recognising" the GT is clean. It is
the test doing the one thing it is built to do.

Minor caveat: the Phipson–Smyth `(1+·)/(1+R)` form is very slightly
*conservative* — floor p-value `1/(1+R) ≈ 0.001`. At R=1000 the discreteness is
negligible, so ≈5% (not ≈4.9%) is expected.

**Do not present "5% on clean GT" as evidence the metric detects cleanliness —
that would be circular.**

## 2. ...but landing at 5% IS a meaningful calibration result

The threat (flagged in `per_tile_metric_design.md` §4.1: *"calibration is
approximate due to within-tile pixel dependence"*):

- Neighbouring gradients are **spatially correlated**.
- Permuting *individual* gradients would break that correlation → permutation
  null becomes artificially tight (too low variance) → test turns
  **anti-conservative** → it would reject clean GT at well above 5% (e.g.
  15–25%), flagging artifacts where there are none.

The **block permutation** (block size `B`, `permutation.py`) is the fix:
permuting contiguous blocks instead of pixels preserves local dependence, so
*blocks* are exchangeable. Therefore the clean-GT experiment is a direct check
of that fix:

- Observing **~5% and not inflation** ⇒ `B` is large enough to capture the
  correlation length; the block permutation correctly neutralises within-tile
  dependence.
- It also confirms seam lines carry **no hidden structure** on clean GT —
  gradients at notional seam locations are genuinely exchangeable with neighbours.

Had `B` been too small, this same experiment would have exposed the
anti-conservatism. So it is a **passed calibration check**, not a tautology.

## 3. Framing for the paper

- "5% on clean GT" → present as a **null-calibration check**: demonstrates the
  block permutation achieves nominal type-I control under realistic pixel
  dependence, validating the choice of `B` and the exchangeability assumption.
  This backs up the §4.1 caveat and the §4.4 null-reference-distribution idea.

## 4. Recommended follow-up — block-size sweep (to make it airtight)

Turn "we observed 5%" into "we verified calibration is robust and identified the
`B` at which it holds":

- **B-sweep on clean GT**: plot fraction-rejected (at α=0.05) vs. block size `B`.
- Expect it to **converge to ≈α once `B` reaches the correlation length** of the
  gradient field; small `B` should show inflation > α.
- Optionally cross-check `B` against an empirical autocorrelation / correlation-
  length estimate of the gradient field (see also `per_tile_metric_design.md`
  §5, item 2, "block size B").

Implementation entry point: `calibration.py` (per-tile permutation calibration).

---

**See also:** `per_tile_metric_design.md` (§3.2 block permutation, §4.1
aggregation caveat, §4.4 null reference distribution, §5 item 2 block size).
