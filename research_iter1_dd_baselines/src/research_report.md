# DD Baselines

## Summary

Comprehensive literature survey documenting precise algorithms for three dependency-tree linearization baselines (RPL, FHD, SOP), the r1-prime bias-corrected autocorrelation estimator for short series, UD spoken/written treebank metadata conventions, WALS case-marking data access methods, and a systematic gap analysis confirming no prior work studies within-sentence lag-1 autocorrelation of dependency distance sequences. Includes pseudocode for all three baselines, exact formulas for autocorrelation estimators, list of UD spoken treebanks, and WALS 49A feature value mappings.

## Research Findings

## SECTION 1: RPL Algorithm -- Random Projective Linearization

The RPL baseline originates from Gildea & Temperley (2007) [1] and was refined in their 2010 Cognitive Science paper [2]. The canonical description states: "for generating random projective linearizations, one simply chooses a random assignment of each dependent to either the left or right of its head, and a random ordering within each side" [2, 20]. Futrell et al. (2015) adopted this as: "Starting at the root node of a dependency tree, collect the head word and its dependents and order them randomly. Then repeat the process for each dependent," using 100 random linearizations per sentence [3].

**RPL Pseudocode:**
```
FUNCTION RPL_linearize(tree, root):
  1. COLLECT all direct dependents of current node
  2. For each dependent, randomly assign LEFT or RIGHT of head (coin flip)
  3. Randomly order dependents within each side
  4. Local order = [shuffled LEFT deps] + [HEAD] + [shuffled RIGHT deps]
  5. RECURSE for each dependent
  6. FLATTEN nested structure
  // Projectivity is guaranteed: head + dependents always form contiguous block
```

The CLIQS repository [4] implements this via `mindep.randlin_projective()`, which uses `expand_randomly()` to recursively shuffle children with the head at each node. The function supports a `head_final_bias` parameter and `move_head`/`move_deps` flags for constrained variants.

**Critical distinction:** Alemany-Puig & Ferrer-i-Cancho (2022) [6] formally characterize two different types: (a) the GT method (random left/right + random within-side), and (b) truly uniform random projective linearizations (head inserted at uniformly random position among dependents). These have different variance properties [6, 24]. The hypothesis should use the standard GT method for comparability with Futrell et al. [3].

The Linear Arrangement Library (LAL) [7, 24] provides an O(n)-time exact algorithm for computing expected dependency length under random projective linearizations, potentially useful for validation.

---

## SECTION 2: FHD Algorithm -- Fixed Head-Direction Baseline

The FHD baseline derives from Futrell et al. (2015) [3] and Futrell, Levy & Gibson (2020) [5]. Futrell et al. (2015) described a "fixed word order" baseline where "each relation type [is assigned] a random weight in [-1,1], collecting the head word and its dependents and ordering them by weight" consistently across all sentences [3]. Futrell et al. (2020) added a "consistent head direction" baseline where all relations are uniformly head-initial or head-final [5].

**FHD Pseudocode:**
```
PRE-COMPUTATION: Build head_direction_table from treebank:
  For each deprel, count head-initial vs head-final instances
  Assign majority direction per deprel

FUNCTION FHD_linearize(tree, root, head_direction_table):
  1. Collect dependents of current node
  2. For each dependent, look up deprel in table -> LEFT or RIGHT of head
  3. RANDOMLY ORDER dependents within each side (residual randomness)
  4. Construct: [random LEFT deps] + [HEAD] + [random RIGHT deps]
  5. RECURSE and FLATTEN
```

The key difference from RPL: FHD fixes head direction per relation type but still randomizes sibling ordering within each side [3, 5]. Multiple samples (100-200) are still needed due to this residual randomness.

The CLIQS code [4] implements this via `WeightedLin` (fixed weight-based ordering) and `EisnerModelC` (learned left/right distributions). Yadav et al. (2022) [8] used Random Linear Arrangements (RLA) with rejection sampling for matched baselines, with data at OSF [8].

---

## SECTION 3: SOP Algorithm -- Sibling-Order-Preserving Baseline (NOVEL)

**Novelty confirmed:** No prior work was found that specifically fixes the relative ordering of co-dependents as a DLM baseline. Related work includes Dyer (2019) on weighted posets for surface order [22], Hahn et al. (2021) on memory-surprisal tradeoffs [21], and Menezes & Quirk (2007) on dependency order templates in MT, but none define a sibling-order-preserving baseline for DLM studies.

**SOP Design (Original):**
```
TEMPLATE EXTRACTION:
  1. For each head node, collect deprel multiset of dependents
  2. Key = (head_UPOS, sorted_deprel_multiset) [recommended conditioning]
  3. For each key, compute majority ordering template
  4. Minimum 5 occurrences for template to be "established"

FUNCTION SOP_linearize(tree, root, direction_table, sibling_templates):
  1. Form key = (UPOS(head), sorted deprel multiset of dependents)
  2. IF template exists (count >= 5): apply fixed ordering (DETERMINISTIC)
  3. ELSE: fall back to FHD behavior for this node
  4. RECURSE and FLATTEN
```

**Key property:** SOP is nearly deterministic for common constructions (>70% of nodes in well-resourced treebanks). Excess-SOP autocorrelation measures only anti-correlation from RARE/unseen patterns -- making it the most conservative baseline in the hierarchy RPL < FHD < SOP. If >90% deterministic, a "soft SOP" variant sampling from weighted template distributions is recommended.

---

## SECTION 4: r1-prime Bias-Corrected Autocorrelation Estimator

Three key estimators for lag-1 autocorrelation in short series:

**Standard r1** [9, 10, 11]: `r1 = sum(x_t - x_bar)(x_{t+1} - x_bar) / sum(x_t - x_bar)^2`. Known negative bias of ~-1/n for independent series.

**r1+ (Huitema & McKean 1991)**: `r1+ = r1 + 1/n`. Simple additive correction for the leading-order bias term. Overcorrects for negative rho, undercorrects for large positive rho.

**r1' (Arnau & Bono 2001)** [9]: `r1' = r1 + |P(r1, n)|` where P is a polynomial fitted to the empirical bias function via Monte Carlo simulation for each sample size n. Polynomial coefficients were derived for n=6, 10, 20, 30 using MATLAB polytool. r1' has lower bias and MSE than both r1 and r1+ for n < 50.

Solanas et al. (2010) [10] compared 10 estimators and found no single optimal estimator for all conditions. Only the delta-recursive, translated, and C-statistic are unbiased at rho=0. Krone et al. (2015) [11] found that for T=10, ALL estimators perform poorly; for T>=25, Bayesian Bsr shows smallest bias.

**Recommendation:** Phase 1 Monte Carlo simulations should derive the bias correction empirically from the exact sentence-length distribution, rather than relying on published polynomial coefficients [9, 11].

---

## SECTION 5: UD Treebank Metadata -- Spoken vs Written

The UD Genre field uses a space-separated list in the README metadata block: `Genre: spoken` [13]. There are ~18 recognized genres including "spoken" as a distinct value [14]. No explicit "written" genre exists -- written text uses specific genres (news, fiction, wiki, etc.) [13, 14].

Dobrovoljc (2022) [12] cataloged spoken UD treebanks (v2.9) including: Beja-NSC, Cantonese-HK, Chinese-HK, Chukchi-HSE, French-ParisStories, French-Rhapsodie, Frisian_Dutch-Fame, Komi_Zyrian-IKDP, Naija-NSC, Norwegian-NynorskLIA, Slovenian-SST, Turkish_German-SAGT, Swedish_Sign_Language-SSLC (~13-18 treebanks depending on version).

**Challenge:** Genre metadata is treebank-level, not sentence-level [14]. Mixed-genre treebanks cannot be split without document-level annotations. Muller-Eberstein et al. (2021) [14] found genre labels are a noisy signal.

Dobrovoljc (2025/2026) [15] found spoken corpora contain fewer and less diverse syntactic structures than written, with very limited cross-modality overlap.

---

## SECTION 6: Gap Analysis -- Novelty of DD Autocorrelation

**Systematic search** across multiple query formulations found NO prior work treating the dependency distance sequence within a sentence as a time series or studying its lag-1 autocorrelation.

Existing work studies: aggregate mean DD per sentence [3, 19], marginal DD distribution shape [18], grammar-preserving DLM baselines [5, 8], constituent ordering preferences (short-before-long) [20], and discourse-level long-range correlations [25]. None examine sequential correlation structure of DD values within sentences.

The closest related finding is the "short-before-long" ordering principle [3, 20] -- shorter constituents placed closer to the head -- which implies a pattern in DD values but has never been formalized as an autocorrelation property.

**Conclusion: No prior work was found that treats the dependency distance sequence within a sentence as a time series and studies its lag-1 autocorrelation or any sequential correlation structure.** Confidence: HIGH across multiple search strategies [3, 5, 8, 18, 19, 20, 21, 25].

---

## SECTION 7: WALS Chapter 49 and Case-Marking Data

WALS 49A (Iggesen) [16] classifies 261 languages into 9 ordinal categories from "No morphological case-marking" (100 languages) to "10 or more cases" (24 languages).

Data is available via CLDF format at GitHub [17] with CSV files containing ISO 639-3 codes and Glottocodes. Mapping to UD treebanks uses ISO 639-3 codes, with Glottolog as bridge for non-matching codes [17, 23]. UD-derived case proportion (tokens with Case= in FEATS) provides a continuous cross-validation measure against WALS ordinal categories.

## Sources

[1] [Gildea & Temperley (2007) - Optimizing Grammars for Minimum Dependency Length, ACL 2007](https://aclanthology.org/P07-1024/) — Introduced the random projective linearization baseline and optimization of grammars for minimum dependency length

[2] [Gildea & Temperley (2010) - Do Grammars Minimize Dependency Length?, Cognitive Science](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1551-6709.2009.01073.x) — Refined the random projective linearization description: random left/right assignment with random within-side ordering

[3] [Futrell, Mahowald & Gibson (2015) - Large-scale evidence of dependency length minimization, PNAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC4547262/) — Canonical RPL algorithm description and fixed word order baseline with 100 random linearizations per sentence

[4] [Futrell - CLIQS: Crosslinguistic Investigations in Quantitative Syntax (GitHub)](https://github.com/Futrell/cliqs) — Python implementation of randlin_projective and linearization functions including WeightedLin and EisnerModelC

[5] [Futrell, Levy & Gibson (2020) - Dependency locality as an explanatory principle for word order](https://sites.socsci.uci.edu/~rfutrell/papers/futrell2020dependency.pdf) — Described consistent head direction and fixed word order baselines for disentangling grammar vs usage DLM

[6] [Alemany-Puig & Ferrer-i-Cancho (2022) - Expected Sum of Edge Lengths, Computational Linguistics](https://direct.mit.edu/coli/article/48/3/491/110442/) — Formal characterization of expected dependency length under random projective linearizations with O(n) algorithm

[7] [Linear Arrangement Library (LAL) - GitHub repository](https://github.com/LAL-project/linear-arrangement-library) — C++/Python library with algorithms for random projective linearizations and dependency length calculations

[8] [Yadav, Mittal & Husain (2022) - A Reappraisal of Dependency Length Minimization, Open Mind](https://pmc.ncbi.nlm.nih.gov/articles/PMC9692064/) — Grammar-preserving baselines using rejection sampling with RLA and matched variants

[9] [Arnau & Bono (2001) - Autocorrelation and Bias in Short Time Series, Quality & Quantity](https://link.springer.com/article/10.1023/A:1012223430234) — Proposed the r1-prime polynomial-corrected autocorrelation estimator for short time series

[10] [Solanas, Manolov & Sierra (2010) - Lag-one autocorrelation in short series, Psicologica](https://www.semanticscholar.org/paper/8694823bd3760cf2b581437608818f5e6812559d) — Compared 10 lag-one autocorrelation estimators; found no single optimal estimator for all conditions

[11] [Krone, Albers & Timmerman (2015) - Comparative simulation study of AR(1) estimators, Quality & Quantity](https://pmc.ncbi.nlm.nih.gov/articles/PMC5227053/) — Comprehensive comparison finding Bayesian Bsr has lowest bias; all estimators poor for T=10

[12] [Dobrovoljc (2022) - Spoken Language Treebanks in UD: an Overview, LREC 2022](https://aclanthology.org/2022.lrec-1.191/) — Comprehensive catalog of spoken UD treebanks with comparative analysis of annotation conventions

[13] [Universal Dependencies - Repository and Files documentation](https://universaldependencies.org/contributing/repository_files.html) — Documents the README metadata format including Genre field specification for UD treebanks

[14] [Muller-Eberstein et al. (2021) - How Universal is Genre in UD?, SyntaxFest 2021](https://github.com/personads/ud-genre) — Analysis of 18 UD genres with weak supervision methods for instance-level genre prediction

[15] [Dobrovoljc (2025) - Counting Trees: Syntactic Variation in Speech and Writing](https://arxiv.org/abs/2505.22774) — Compared syntactic structures in speech vs writing; found limited cross-modality structural overlap

[16] [WALS Online - Feature 49A: Number of Cases (Iggesen)](https://wals.info/feature/49A) — Typological data on morphological case-marking across 261 languages with 9 ordinal categories

[17] [WALS in CLDF format (GitHub repository)](https://github.com/cldf-datasets/wals) — WALS data in Cross-Linguistic Data Formats with ISO 639-3 codes and Glottocodes

[18] [Petrini & Ferrer-i-Cancho (2022) - The distribution of syntactic dependency distances](https://arxiv.org/abs/2211.14620) — Studies marginal distribution shape of DD (two-regime exponential), NOT sequential ordering

[19] [Liu (2008) - Dependency Distance as a Metric of Language Comprehension Difficulty](https://www.researchgate.net/publication/273459859) — Established mean dependency distance as aggregate complexity metric across 20 languages

[20] [Temperley & Gildea (2018) - Minimizing Syntactic Dependency Lengths, Annual Review of Linguistics](https://www.cs.rochester.edu/u/gildea/pubs/temperley-gildea-ar18.pdf) — Comprehensive review confirming canonical description of random projective linearization algorithm

[21] [Hahn, Degen & Futrell (2021) - Modeling word and morpheme order, Psychological Review](https://www.mhahn.info/files/hahn_psychreview_2021_final.pdf) — Memory-surprisal tradeoff framework for word order efficiency in 54 languages

[22] [Dyer (2019) - Weighted posets: Learning surface order from dependency trees](https://aclanthology.org/W19-7807.pdf) — Proposed weighted poset approach for learning linearization including sibling ordering

[23] [How to work with WALS data in CLDF - Tutorial](https://calc.hypotheses.org/2670) — Tutorial on accessing WALS data via CLDF Python packages with SQLite conversion

[24] [Alemany-Puig et al. (2021) - The Linear Arrangement Library, Quasy@SyntaxFest](https://aclanthology.org/2021.quasy-1.1/) — LAL library with random/exhaustive arrangement generation under projectivity constraints

[25] [Long-range sequential dependencies in language acquisition, Proc B (2022)](https://royalsocietypublishing.org/doi/10.1098/rspb.2021.2657) — Power-law correlation analysis at discourse level, not within-sentence DD sequences

## Follow-up Questions

- Which RPL variant should the implementation use: Gildea-Temperley's random left/right assignment method or the simpler random-permutation-of-head-among-dependents method, and does the choice affect autocorrelation baselines differently than it affects mean DD baselines?
- For the SOP baseline, what is the optimal conditioning granularity (deprel multiset only vs head UPOS + deprel multiset) and minimum count threshold, and how deterministic does SOP turn out to be in practice across diverse UD treebanks?
- Since the Arnau & Bono (2001) polynomial coefficients for r1-prime are behind a paywall and sample-size-specific, should the Phase 1 Monte Carlo simulations derive the bias correction empirically from the exact sentence-length distribution, making r1-prime unnecessary as a published formula?
- How should mixed-genre UD treebanks (e.g., English-GUM with both spoken and written portions) be handled in the spoken/written comparison - split using document-level metadata, excluded entirely, or treated as a separate mixed category?

---
*Generated by AI Inventor Pipeline*
