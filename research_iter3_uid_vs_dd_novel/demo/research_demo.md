# UID vs DD Novelty

## Summary

Systematic literature synthesis establishing that DD anti-correlation is NOT a special case of Uniform Information Density (UID). Three key distinctions: (1) UID operates on surprisal while DD operates on structural distance, and these are uncorrelated per Demberg & Keller 2008; (2) the RPL/FHD/SOP baseline hierarchy separates grammar vs. usage effects more finely than UID's counterfactual baselines; (3) the morphology-modulation prediction is novel. Critical citation finding: Dou et al. (2026) EXISTS but with wrong authors — actual authors are Dou, Ariens, Ceulemans & Lafit, not Dou, Mulder & van Buuren. Two venue corrections identified: Meister et al. is EMNLP 2021 (not TACL); Jaeger 2010 is Cognitive Psychology (not Cognitive Linguistics).

## Research Findings

## DD Anti-Correlation vs. Uniform Information Density: A Comprehensive Analysis

### Core Finding: DD Anti-Correlation Is NOT a Special Case of UID

Dependency distance (DD) anti-correlation within sentences is **not** a special case of Uniform Information Density (UID). This conclusion rests on three key distinctions, supported by systematic analysis of the UID literature.

---

### Distinction 1: Different Target Quantities (Structural Distance vs. Surprisal)

UID predicts smoothing of **surprisal** (information rate). Levy (2008) established surprisal theory, where processing difficulty at word w_t is proportional to -log P(w_t | w_1...w_{t-1}) [4]. Jaeger (2010) formulated UID as a production principle: "Within the bounds defined by grammar, speakers prefer utterances that distribute information uniformly across the signal" [2]. Meister et al. (2021) explored multiple operationalizations of UID, all defined **exclusively in terms of surprisal** — no other processing-cost proxies such as dependency distance are considered [1].

DD anti-correlation, by contrast, operates on **structural distances** — the linear distance between syntactically related words. The critical evidence comes from Demberg & Keller (2008), who showed that DLT integration cost (which is dependency distance) and surprisal are **uncorrelated** in naturalistic reading data from the Dundee eye-tracking corpus [5]. DLT integration cost was not a significant predictor of reading times for arbitrary words — only for nouns — while unlexicalized surprisal predicted reading times across the board [5]. As they concluded, "the two measures are uncorrelated, which suggests that a complete theory will need to incorporate both aspects of processing complexity" [5].

This uncorrelatedness is the lynchpin of the novelty argument: if DD and surprisal are uncorrelated, then anti-correlation patterns in DD sequences **cannot** be derived from anti-correlation patterns in surprisal sequences, even under the strongest interpretation of UID.

A concrete counterexample reinforces this: a short dependency can carry high surprisal (e.g., an unexpected subject-verb agreement across a short span), while a long dependency can be highly predictable (e.g., relative clause attachment to the only available head noun).

### Distinction 2: Different Baseline Methodologies

Clark et al. (2023) used counterfactual word orders (reversed, shuffled, and "linguistically implausible" orders) as UID baselines, finding that "among SVO languages, real word orders consistently have greater uniformity than reverse word orders" [3]. However, these baselines destroy both grammatical structure and usage-level patterns simultaneously — they do **not** separate grammar-level from usage-level effects.

The RPL/FHD/SOP hierarchy used in the DD anti-correlation hypothesis progressively controls for grammatical regularities, isolating usage-level anti-correlation at each tier. No UID paper uses anything analogous to the SOP baseline (preserving co-dependent ordering conventions). UID baselines ask a **global** question ("does real order have more uniform information than alternatives?"), while the DD hierarchy asks a **local, sequential** question ("does the sequential pattern of real DD values show anti-correlation beyond what grammar already produces?").

### Distinction 3: Novel Morphology-Modulation Prediction

Standard UID predicts uniform information rate across languages but does **NOT** predict that UID pressure should vary with morphological richness [1, 3]. Clark et al. (2023) found UID effects for SVO languages but did not test morphological modulation [3]. The prediction that case-marking weakens DD anti-correlation derives from an ecological analogy (resource abundance weakens compensatory dynamics), connecting within-sentence dynamics to the macro-level morphology-word-order tradeoff. No existing UID framework derives or predicts this pattern.

### Bridge Paper: Futrell et al. (2020)

Futrell, Gibson & Levy (2020) derived dependency locality effects from information theory via the "information locality" principle using lossy-context surprisal [6]. However, this model does **not** predict sequential patterns of dependency distances, does not imply that high cost at one position predicts lower cost at the next, and makes no cross-linguistic predictions about morphological richness [6]. They describe surprisal and dependency distance as "complementary" phenomena [6].

### Literature Gap Confirmed

Targeted searches for "dependency distance autocorrelation," "dependency distance time series within sentence," "dependency length sequential pattern smoothing," and "surprisal autocorrelation within sentence" returned **no relevant results**. No prior work has analyzed the sequential autocorrelation of dependency distances within sentences. Tsipidi et al. (2024) challenged UID at the discourse level via the "Structured Context Hypothesis," but this operates on surprisal contours across discourse, not DD sequences within sentences [7].

---

### Citation Verification: Dou et al. (2026)

**CRITICAL FINDING: The paper EXISTS but the hypothesis cites WRONG AUTHORS.**

- **Hypothesis cites**: Dou, Mulder & van Buuren (2026)
- **Actual paper**: Dou, Z., Ariens, S., Ceulemans, E., & Lafit, G. (2026). "Unravelling the small sample bias in AR(1) models: The pros and cons of available bias correction methods." *British Journal of Mathematical and Statistical Psychology*, 00, 1–24. DOI: 10.1111/bmsp.70038 [8]

The title and venue (BJMSP) are correct, as is the first author "Dou." However, "Mulder" and "van Buuren" are NOT co-authors — likely hallucinated by confusing Stef van Buuren (prominent Dutch statistician known for MICE) and a common Dutch statistics surname with the actual co-authors. **The citation should be corrected to list the actual authors.**

The paper's content is relevant: it analytically demonstrates the causes and consequences of small-sample OLS bias in AR(1) models and reviews bias correction methods [8], directly supporting the methodological discussion in the hypothesis.

**Supplementary references** for AR(1) bias correction:
- Krone, Albers & Timmerman (2017) compared r1, C-statistic, OLS, MLE, and Bayesian estimators at T=10–100, finding Bayesian Bsr has lowest bias while r1 has lowest variability [9]. Does NOT test r1-prime specifically.
- Shaman & Stine (1988) derived O(T^-1) bias formula for LS and Yule-Walker AR estimators [10].
- Huitema & McKean (2002) introduced the r1-prime bias-corrected estimator, recommended for samples under 50 [11].

---

### Venue Corrections Required

1. **Meister et al. (2021)**: Published at **EMNLP 2021** (Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 963–980), **NOT TACL** [1].
2. **Jaeger (2010)**: Published in **Cognitive Psychology** 61(1), 23–62, **NOT Cognitive Linguistics** [2].

---

### Confidence Assessment

**High confidence** that DD anti-correlation is not a special case of UID, based on:
- The empirical uncorrelatedness of DD and surprisal [5]
- The absence of sequential autocorrelation predictions in any UID paper [1, 2, 3, 4]
- The confirmed literature gap in DD sequence analysis

**What would change this conclusion**: Discovery of a paper showing that surprisal autocorrelation within sentences is strongly positive and that DD mirrors this pattern — which would suggest DD anti-correlation might be an indirect consequence of information-theoretic pressure. However, no such paper exists in the current literature.

## Sources

[1] [Meister et al. (2021) - Revisiting the Uniform Information Density Hypothesis, EMNLP 2021](https://aclanthology.org/2021.emnlp-main.74/) — Explores multiple operationalizations of UID, all based on surprisal. Finds regression towards mean surprisal at language level. No discussion of dependency distance, sequential autocorrelation, or syntactic structure. Venue: EMNLP 2021 (NOT TACL).

[2] [Jaeger (2010) - Redundancy and reduction: Speakers manage syntactic information density, Cognitive Psychology 61(1), 23-62](https://pmc.ncbi.nlm.nih.gov/articles/PMC2896231/) — Formulates UID as production principle tested via optional that-mentioning. Focuses on individual choice points, not sequential dependencies. Information density measured via verb subcategorization frequency, not dependency distances. Venue: Cognitive Psychology (NOT Cognitive Linguistics).

[3] [Clark et al. (2023) - A Cross-Linguistic Pressure for Uniform Information Density in Word Order, TACL 11:1048-1065](https://aclanthology.org/2023.tacl-1.59/) — Tests UID cross-linguistically in 10 languages using counterfactual word order baselines. SVO languages show UID effect. Does not test morphological modulation or measure sequential surprisal autocorrelation.

[4] [Levy (2008) - Expectation-based syntactic comprehension, Cognition 106(3), 1126-1177](https://pubmed.ncbi.nlm.nih.gov/17662975/) — Establishes surprisal theory: processing difficulty proportional to -log P(word|context). Treats each word independently given context. Does not predict sequential autocorrelation of processing costs.

[5] [Demberg & Keller (2008) - Data from eye-tracking corpora as evidence for theories of syntactic processing complexity, Cognition 109(2), 193-210](https://pubmed.ncbi.nlm.nih.gov/18930455/) — CRITICAL: Shows DLT integration cost (dependency distance) and surprisal are UNCORRELATED. DLT only predicts reading times for nouns, not arbitrary words. Establishes that DD and surprisal capture fundamentally different dimensions of processing complexity.

[6] [Futrell, Gibson & Levy (2020) - Lossy-Context Surprisal, Cognitive Science 44(3), e12814](https://pmc.ncbi.nlm.nih.gov/articles/PMC7065005/) — Derives dependency locality effects from information theory via lossy memory. Does NOT predict sequential DD patterns, cost anti-correlation between consecutive words, or morphological modulation. Describes DD and surprisal as complementary.

[7] [Tsipidi et al. (2024) - Surprise! Uniform Information Density Isn't the Whole Story, EMNLP 2024](https://arxiv.org/abs/2410.16062) — Challenges UID at discourse level, proposing Structured Context Hypothesis. Operates on surprisal contours across discourse, not DD sequences within sentences. Shows UID is insufficient for explaining information rate modulation.

[8] [Dou, Ariens, Ceulemans & Lafit (2026) - Unravelling the small sample bias in AR(1) models, BJMSP, DOI: 10.1111/bmsp.70038](https://bpspsychub.onlinelibrary.wiley.com/doi/10.1111/bmsp.70038) — Paper EXISTS but hypothesis cites WRONG AUTHORS (Dou, Mulder & van Buuren). Actual authors: Dou, Z., Ariens, S., Ceulemans, E., & Lafit, G. Analytically demonstrates causes of small-sample OLS bias in AR(1) models and reviews bias correction methods.

[9] [Krone, Albers & Timmerman (2017) - A comparative simulation study of AR(1) estimators in short time series, Quality & Quantity 51(1), 1-21](https://pmc.ncbi.nlm.nih.gov/articles/PMC5227053/) — Compares r1, C-statistic, OLS, MLE, and Bayesian estimators at T=10-100. Bayesian Bsr has lowest bias; r1 has lowest variability. Does NOT test r1-prime (Huitema-McKean) estimator specifically.

[10] [Shaman & Stine (1988) - The bias of autoregressive coefficient estimators, JASA 83(403), 842-848](https://www.tandfonline.com/doi/abs/10.1080/01621459.1988.10478672) — Derives O(T^-1) bias formula for least-squares and Yule-Walker AR coefficient estimators. Foundational reference for understanding AR estimation bias in short series.

[11] [Huitema & McKean (2002) - Autocorrelation and Bias in Short Time Series: An Alternative Estimator, Quality & Quantity](https://link.springer.com/article/10.1023/A:1012223430234) — Introduces r1-prime bias-corrected lag-1 autocorrelation estimator. r1-prime has lower bias and MSE than standard r1. Recommended for samples under 50 observations.

## Follow-up Questions

- Does the Futrell (2020) lossy-context surprisal model make testable predictions about DD sequence autocorrelation that could be empirically compared with the DD anti-correlation findings?
- Has anyone measured surprisal autocorrelation within sentences (not just across discourse as in Tsipidi et al. 2024) to compare with DD autocorrelation patterns?
- Does Krone et al. (2017) or the corrected Dou et al. (2026) test the specific r1-prime estimator from Huitema & McKean, or only the standard r1 and other estimators?

---
*Generated by AI Inventor Pipeline*
