# References — quantifying tiling/stitching artifacts

Working bibliography for the new "quantifying tiling artifacts" paragraph in the introduction.
Each entry has a one-line explanation of why it is relevant.

---

## 1. Tiling artifacts in DL pipelines (closest prior work)

- **Buglakova et al. 2025** — *Tiling artifacts and trade-offs of feature normalization in the segmentation of large biological images*, arXiv:2503.19545.
  Introduces the `tile_mismatch` (Dice on overlapping-tile predictions) and `train/eval disparity` metrics; closest existing reference-free metric, but tied to segmentation and requires two predictions of the same region.

- **Ashesh et al. 2022** — *Inner vs. Outer Padding for tiled prediction with HVAEs*, arXiv.
  Quantifies tiling sensitivity via PSNR percent-variation across padding configurations; requires ground truth and conflates artifact magnitude with overall task error.

- **Reina et al. 2020** — *Systematic Evaluation of Image Tiling Adverse Effects on Deep Learning Semantic Segmentation*, Frontiers in Neuroscience.
  Quantifies tiling effects via Dice gap between tiled and whole-image inference; requires both ground truth and a non-tiled baseline (often infeasible for the images that motivate tiling).

- **Rumberger et al. 2021** — *How shift equivariance impacts metric learning for instance segmentation*, ICCV 2021.
  Frames tiling artifacts as a violation of shift equivariance; architectural-property perspective adjacent to but distinct from per-prediction artifact measurement.

- **Bartschat et al. 2024** — *Image processing tools for petabyte-scale light sheet microscopy data*, Nature Methods 21:2342–2352.
  Pipeline-level stitching for very large microscopy volumes; useful for motivation but not for the DL-prediction-stitching question.

---

## 2. Blockiness / blocking-artifact metrics (the closest conceptual analog)

The JPEG-blockiness literature is the closest technical precedent for our setup: known artifact locations + reference-free statistical comparison between artifact-prone and artifact-free regions.

- **Wang, Sheikh & Bovik 2002** — *No-reference perceptual quality assessment of JPEG compressed images*, ICIP 2002.
  Canonical pixel-domain blockiness metric using inter-block vs. intra-block pixel differences at known boundaries; direct conceptual ancestor of our construction.

- **Wang, Bovik & Evans 2000** — *Blind measurement of blocking artifacts in images*, ICIP 2000.
  Earlier formulation; introduces explicit luminance + texture masking weights to scale the seam-difference signal.

- **Bovik & Liu 2001** — *DCT-domain blind measurement of blocking artifacts in DCT-coded images*, ICASSP 2001.
  Frequency-domain detection of the characteristic peak produced by periodic block boundaries; content-invariant by construction.

- **Liu & Heynderickx 2009** — *A perceptually relevant no-reference blockiness metric based on local image characteristics*, EURASIP J. Advances in Signal Processing.
  Per-seam visibility coefficient based on local high-pass response; the cleanest reference for texture/luminance masking as a design principle.

- **Pan et al. 2004** — *Using edge direction information for measuring blocking artifacts of images*, Multidimensional Systems and Signal Processing 18(4).
  Ratio of inter-block to intra-block pixel differences; the per-seam content-adaptive normalization we are adopting.

- **Lee & Park 2012** — *A new image quality assessment method to detect and measure strength of blocking artifacts*, Signal Processing: Image Communication 27(1):31–38.
  More recent variant of the same inter/intra-block construction.

- **Chen & Bloom 2010** — *Image blockiness evaluation based on Sobel operator*, Pacific-Rim Conf. on Advances in Multimedia Information Processing.
  Gradient-based variant of blockiness measurement, closest to our use of pixel gradients.

---

## 3. Natural image statistics and gradient distributions

Foundational results on why pixel-gradient distributions in natural images are characteristic, and how deviations signal degradation.

- **Huang & Mumford 1999** — *Statistics of natural images and models*, CVPR 1999.
  Classic empirical characterization of natural-image statistics, including heavy-tailed gradient distributions.

- **Weiss & Freeman 2007** — *What makes a good model of natural images?*, CVPR 2007.
  Tests gradient-based natural-image priors; supports the use of gradient distributions as a generic image-content descriptor.

- **Roth & Black 2009** — *Fields of Experts: a framework for learning image priors*, IJCV 82(2):205–229.
  Learned gradient-domain priors for natural images; canonical reference for the modelling side.

- **Srivastava, Lee, Simoncelli & Zhu 2003** — *On advances in statistical modeling of natural images*, J. Mathematical Imaging and Vision 18:17–33.
  Review of gradient and wavelet-coefficient statistics; useful for a general citation.

---

## 4. Reference-free IQA based on natural image statistics

- **Mittal, Moorthy & Bovik 2012** — *No-reference image quality assessment in the spatial domain* (BRISQUE), IEEE TIP 21(12):4695–4708.
  Locally normalized luminance statistics compared against a pristine reference distribution; canonical NR-IQA via natural-scene-statistics.

- **Mittal, Soundararajan & Bovik 2013** — *Making a "completely blind" image quality analyzer* (NIQE), IEEE Signal Processing Letters 20(3):209–212.
  Fully blind variant of BRISQUE; no labelled training data needed.

- **Liu et al. 2010** — *Image quality assessment using natural image statistics in gradient domain*, ScienceDirect / J. Visual Communication and Image Representation.
  Reduced-reference IQA using **resistor-average distance** between gradient-domain distributions; the closest existing work to our KL-on-gradient-distributions construction, and the cite for justifying symmetric divergence alternatives.

- **Hassen, Wang & Salama 2010** — *No-reference image sharpness assessment based on local phase coherence measurement*, ICASSP 2010.
  Local-statistics-based blur metric; example of the "test local statistics against expected behaviour" template.

---

## 5. Two-sample / non-parametric tests on image data

The statistical-testing layer that motivates the hypothesis-test framing of our metric.

- **Demidenko 2004** — *Kolmogorov-Smirnov Test for Image Comparison*, ICCSA 2004 (LNCS 3046).
  Direct use of the KS two-sample test for image-region comparison; closest formal precedent for our framing.

- **Şenel et al. (Şenel)** — *Compressed Medical Image Quality Determination using the Kolmogorov-Smirnov Test*.
  KS two-sample test comparing pixel distributions of compressed vs. original images; very close analog to "are seam pixels statistically different from interior pixels?".

- **Maggioni, Boracchi, Foi & Egiazarian** — patch-similarity literature for non-local denoising; CDF/KS-based patch distances.
  Demonstrates that CDF-based distances between image patches are well-behaved and statistically motivated.

- **Rajan, Poot, Juntu & Sijbers 2010** — *Roughly KS-test-based patch similarity for MRI denoising*, MRI / signal-processing literature.
  Patch comparison via KS test on intensity differences; the medical-imaging analog of our setup.

- **Pauwels & Frederix 2000** — *Image segmentation by non-parametric clustering based on the Kolmogorov-Smirnov distance*.
  Early use of KS distance for region comparison in images.

---

## 6. Notes on integration

Bibliography priorities for the new paragraph (most-to-least essential):

1. Buglakova et al. 2025 — closest existing metric; must cite.
2. Wang, Sheikh & Bovik 2002 — canonical blockiness ancestor; must cite.
3. Liu & Heynderickx 2009 — texture masking; must cite if we discuss content-dependence.
4. Pan et al. 2004 — inter/intra-block ratio; must cite if we adopt local control regions.
5. Demidenko 2004 — KS test for image comparison; must cite for the hypothesis-test framing.
6. Liu et al. 2010 (gradient-domain IQA) — closest existing gradient-distribution IQA; strongly recommended.
7. Ashesh et al. 2022; Reina et al. 2020 — for the "existing approaches require ground truth" point.
8. Huang & Mumford 1999 / Roth & Black 2009 — one of these for the natural-image-statistics anchor.

Open questions still to resolve before drafting:

- Final choice of local-control construction (fixed-width strip vs. full adjacent tile region).
- Whether to convert the metric to a hypothesis-test form (KS / permutation) for the experiments, or keep KL and borrow only the framing.
- Direction-aware vs. magnitude-only gradients in the actual implementation.
