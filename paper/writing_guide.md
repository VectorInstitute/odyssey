# Writing guide: npj Digital Medicine

**Scope note**: this guide covers structure, hard limits, required checklists, and
observed style patterns at the target venue. It is not about voice — Amrit will
personally rewrite all prose for voice and tone before submission. Use this to
scaffold drafts correctly the first time, not to imitate anyone's writing style.

Sources: npj Digital Medicine's own author-facing pages (content-types, guide-to-authors,
submission-guidelines, editorial-policies), the TRIPOD+AI statement, Nature Portfolio's
current AI editorial policy, and seven npj Digital Medicine papers read in full (not
abstracts) — see citations inline below.

---

## 1. Hard constraints

### 1.1 Article type

**Type: Article** (original primary research, peer-reviewed). Confirmed against
npj Digital Medicine's own [content-types page](https://www.nature.com/npjdigitalmed/content-types).
Not "Review" (systematic reviews/meta-analyses are also directed to Article), not
"Brief Communication" (1,500-word cap, wrong scope for a multi-model, multi-dataset study).

### 1.2 Limits (Article type)

| | Limit |
|---|---|
| Main text word count | No hard cap ("online only and fully Open Access") — write concisely anyway |
| Title | ≤15 words, no punctuation/idioms/puns |
| Abstract | **≤150 words, one unstructured paragraph, no subheadings** |
| References | "Limited to 60... though not strictly enforced." Real papers in our shape run 41–72 (median ~56); budget 55–75 |
| Figure legends | ≤350 words each |
| Figures / tables | No fixed numeric cap found. Real comparable papers: 4–9 main figures (median ~7), 1–6 main tables |
| Supplementary | One Supplementary Information PDF, S-numbered figures/tables. **No Extended Data tier** — don't plan around it |

### 1.3 Required section order

Nature-family convention, confirmed for this journal:

**Abstract → Introduction → Results → Discussion → Methods → Data availability → Code availability → Author contributions → Competing interests → References**

- Introduction: no subheadings.
- Results: subheadings used, one subsection per experiment/question (see §3.2).
- Discussion: no subheadings, **no separate "Limitations" or "Conclusions" section** — both fold into Discussion (see §3.4). Final paragraph opens "In conclusion,".
- Methods: subdivided by short bold headings, comes **last**. npj Digital Medicine does not permit Supplementary Methods — everything needed to reproduce the work must be in the main Methods.

### 1.4 Reporting checklist: TRIPOD+AI

The journal's own guide-to-authors page still links to the 2015 TRIPOD checklist under
"strongly encouraged" (not phrased as mandatory). That page has not caught up to TRIPOD+AI
(2024). **Use TRIPOD+AI anyway** — the EQUATOR Network states it supersedes TRIPOD 2015 for
any prediction-model study using regression or ML methods, and it's the appropriate standard
for this paper. Cite it explicitly in Methods: *"This study was conducted in accordance with
the TRIPOD+AI guidelines (Collins et al., BMJ 2024;385:e078378)."* — the exact phrasing Yoon
et al. 2025 uses for the original TRIPOD.

Full checklist: [tripod-statement.org TRIPOD+AI Expanded Checklist](https://www.tripod-statement.org/wp-content/uploads/2024/04/TRIPODAI-Supplement.pdf).
27 items across Title/Abstract/Introduction/Methods/Open Science/Results/Discussion, each
tagged (D) development, (E) evaluation, or (D;E) both.

**Items most load-bearing for this paper** (map directly onto our design):

- **12a–12c** (data partitioning, model-building steps): must be spelled out separately for
  each of the four model families (GBM / tabular-FM / NAM-GAM / hybrid-Mamba+CBM).
- **16 / 20c** (development vs. evaluation differences): this *is* our MIMIC-IV/eICU →
  GEMINI external-validation design. Don't just report it — TRIPOD+AI wants the distributional
  shift itself characterized.
- **14** (fairness): approaches used and rationale.
- **18e/18f** (data/code sharing): see §1.5–1.6 — explicitly warns against vague "available
  upon reasonable request" language.
- **25** (interpretation): "ensure the interpretation of the findings does not go beyond the
  findings... to prevent overinterpretation or 'spin'" — directly backs our framing-gates stance.
- **26** (limitations): non-representative sample, sample size, overfitting, missing data, and
  their effects on bias/generalizability.
- **27a–27c** (usability, next steps): what a user must do to use the model, and — critically —
  an explicit statement of what's still needed before clinical deployment. Do not claim
  deployment-readiness; state the gap.

The Nature Portfolio Reporting Summary is also required (a separate, standard form submitted
with the revised manuscript) — not a substitute for TRIPOD+AI, an addition to it.

### 1.5 Data availability

Required, own section, states access conditions — not just "available." TRIPOD+AI item 18e:
*"Avoid platitudes such as 'Data available upon reasonable request' without specifying
conditions for what constitutes a reasonable request."*

**Real example to follow** (Guo & Fries et al. 2024, npj Digit Med 7:171 — MIMIC-IV +
SickKids, directly analogous to our situation):

> "The SickKids dataset cannot be made publicly available because of the potential risk to
> patient privacy. However, relevant data are available upon reasonable request. The MIMIC-IV
> dataset is available from https://mimic.mit.edu/docs/iv/."

For us: name each dataset separately. MIMIC-IV and eICU get real PhysioNet URLs and the
credentialing requirement stated explicitly (not just "restricted"). GEMINI gets its governed,
non-public status stated plainly with its actual access-request mechanism — don't default to
the weak "available on reasonable request" phrasing the checklist itself flags as
insufficient.

### 1.6 Code availability

Required whenever custom code is central to the conclusions — which it is here. Separate
section, after Data availability, before References. Must state whether/how the code can be
accessed, including restrictions. TRIPOD+AI item 18f wants code sufficient to replicate all
reported results, including data-cleaning code, with software/package versions and compute
environment noted where relevant.

**Real example** (same Guo et al. paper):

> "The code for the analyses conducted in this study is open-source and available at
> https://github.com/sungresearch/femr-on-sk and https://github.com/sungresearch/femr-on-mimic."

npj Digital Medicine's own policy is stricter than a plain statement: reviewers/editors can
require code sharing during review even before public release, and can decline the manuscript
if important code is withheld without adequate justification.

### 1.7 Ethics / IRB

General policy: name the approving ethics committee (with reference number) per the
Declaration of Helsinki, or the committee granting exemption. The journal gives no specific
guidance for our exact scenario (retrospective, de-identified EHR data under each source
institution's own prior IRB approval) — practice must follow precedent. TRIPOD+AI item 17 and
real published examples both point the same way: **name the approving body separately for each
data source**, not one blanket sentence.

**Real example** (Guo et al.):

> "The SickKids Research Ethics Board (REB) approved the use of SEDAR for this research (REB
> number: 1000074527). Data from MIMIC-IV was approved under the oversight of the Institutional
> Review Boards (IRB) of BIDMC and Massachusetts Institute of Technology (MIT)."

For us: separate sentences for MIMIC-IV (BIDMC/MIT IRB, consent waiver), eICU (its originating
IRB/exemption basis), and GEMINI (its governing REB and data-sharing agreement).

### 1.8 Claim/validation-standard constraints

No standalone "overclaiming" policy page — enforced through TRIPOD+AI item 25 ("prevent
overinterpretation or spin") and item 27c (state explicitly what's still needed before
clinical deployment, don't assert readiness) plus ordinary peer review. A recent npj Digital
Medicine paper on validation terminology ([Clarifying validation terminologies in
healthcare](https://www.nature.com/articles/s41746-026-02471-2)) argues directly that calling
a model "validated" after only internal cross-validation misrepresents deployment-readiness,
and recommends qualifying every validation claim by dataset, method, population, and
operational conditions tested.

**Implication for us**: per the project's own dataset-roles framing, MIMIC-IV/eICU results are
internal/pipeline-portability evidence; only GEMINI results are external-validation/
generalization evidence. This is retrospective validation only. Discussion/Limitations must
stop short of "clinical utility," "deployment-ready," or unqualified "generalizable."

---

## 2. AI use and disclosure

**First-class section — this materially affects how we position the whole submission.**
Quoted verbatim from the actual current pages, not paraphrased.

### 2.1 The current policy (two layers, both apply)

Nature Portfolio replaced its older prose AI policy with a **Green / Amber / Red risk
framework**, scoped explicitly to "the research and publishing lifecycle," not just writing.
Source: [nature.com/nature-portfolio/editorial-policies/ai](https://www.nature.com/nature-portfolio/editorial-policies/ai).

> "AI is treated as a supporting technology. Scholarly judgement, accountability, and
> responsibility always remain human."

- **Green (permitted, disclosure encouraged not mandated)**: polishing language, structuring
  reviewer comments, comparing methodological options, data cleaning/deduplication.
- **Amber (permitted, disclosure required)**: "Suggesting analytical, experimental or
  methodological approaches"; "Pattern identification in exploratory data analysis";
  "Recommending statistical tests or modelling approaches"; "**Extensive** copy editing or
  writing support." *"Permitted with human oversight, verification, and transparency through
  disclosure."*
- **Red (prohibited)**: "Generating hypotheses, analyses or conclusions and presenting them as
  human-derived"; "Using an LLM to generate core research reasoning without disclosure";
  "Assigning authorship or accountability to AI systems or tools."

npj Digital Medicine's own AI policy page is identical in substance to the portfolio page, but
its **Submission Guidelines** carry an older, still-live, more specific instruction that governs
*where* to disclose:

> "Large Language Models (LLMs), such as ChatGPT, do not currently satisfy our authorship
> criteria. Notably an attribution of authorship carries with it accountability for the work,
> which cannot be effectively applied to LLMs. **Use of an LLM should be properly documented
> in the Methods section** (and if a Methods section is not available, in a suitable
> alternative part) of the manuscript."
>
> — [npj Digital Medicine Submission Guidelines](https://www.nature.com/npjdigitalmed/for-authors-and-referees/submission-guidelines)

**→ Disclosure goes in Methods, as its own subsection.** (Springer Nature's general FAQ also
mentions Acknowledgements as an option; Methods is the operative, journal-specific instruction
and the safer choice for a research article.)

The narrow copy-editing carve-out (*"AI-assisted copy editing need not be declared... when
used only to improve readability, grammar, or formatting"*) still exists at the Springer Nature
group level, but the new framework's "extensive copy editing or writing support" sits in Amber
— so anything beyond light grammar/readability polishing should be disclosed anyway.

**No explicit policy exists yet on AI-generated code specifically.** Checked: the portfolio AI
page, npj Digital Medicine's AI page, guide-to-authors, and submission guidelines — "code"
doesn't appear in any AI-policy context. The closest signal is a *different* Nature-family
journal's editorial (Nature Ecology & Evolution, not binding here) recommending disclosure of
AI use "for the analysis of data or preparation of code, including the prompts used." Treat
this as the direction Nature-family editorial thinking is heading and worth voluntarily
matching, not as a binding npj Digital Medicine rule. What **is** binding, provenance-neutral,
and does the real enforcement work regardless of how code was produced: the Code Availability
requirement (§1.6) — results must be independently reproducible from the released code no
matter who or what wrote it.

### 2.2 Recommended disclosure text for this project

Two Methods subsections (drop the second if usage was strictly grammar/readability):

> **Use of AI tools.** Portions of this study's software engineering work — including
> construction of the data extraction and preprocessing pipeline, and a substantial share of
> the analysis and experiment-orchestration code — were produced with the assistance of an AI
> coding agent ([tool name and model version], [vendor]), operating under continuous human
> direction. The agent was used for code authoring and refactoring under explicit human
> specification; it was not used to generate hypotheses, to select or interpret experimental
> results, or to draft the scientific claims of this manuscript. Every AI-assisted change was
> reviewed and approved by a human author before being incorporated, and all changes are
> recorded with authorship attribution in the project's version control history. All analytical
> outputs reported here were verified by the human authors against the underlying data and
> code. Study design, choice of analytical and modelling approaches, interpretation of results,
> and all scientific conclusions are the work of the named human authors, who take full
> accountability for them. No AI system meets authorship criteria and none is listed as an
> author. The complete pipeline and analysis code are available as described under Code
> availability, allowing the reported results to be independently reproduced irrespective of
> how the code was authored.

> Language-model assistance was additionally used for [copy editing / improving the clarity and
> readability of author-written text]. All text was written and verified by the human authors,
> who are responsible for its final content.

**Before finalizing, check honestly** whether the agent, at any point, suggested a modelling
approach, statistical test, feature transformation, or surfaced a pattern in exploratory
analysis (this happened in this project's own concept-lever investigations and error analyses).
If so, that's Amber activity and needs its own sentence — omitting it is what would turn an
Amber disclosure into an undisclosed Red one:

> "Where the agent proposed analytical or modelling options, these were evaluated and selected
> by the human authors, who verified the resulting analyses independently."

Optional, cheap, and reads well to an integrity-minded editor:

> "Records of the AI-assisted development process, including the project's research journal
> and full commit history, are available from the corresponding author on request."

Cover-letter line:

> "We note that the data pipeline and a substantial portion of the analysis code for this study
> were developed using AI-assisted software engineering under human direction, review, and
> verification. This is described in the 'Use of AI tools' subsection of the Methods. No AI
> system is an author, and all scientific claims and interpretations are the authors' own."

---

## 3. Style patterns: do / don't

Drawn from close reading (full text, not abstracts) of seven recent npj Digital Medicine
papers, prioritized by closeness of shape to ours: Tranchellini et al. 2026 (MIMIC-IV+eICU+HiRID
cross-site sepsis prediction), Hegselmann & von Arnim et al. 2026 (multi-family EHR encoder
benchmark with external validation), Guo & Fries et al. 2024 (EHR foundation model vs. GBM, two
sites), Yoon et al. 2025 (interpretable multitask model, two external cohorts), plus Lee et al.
2025, Xie et al. 2025, Goldschmidt et al. 2025.

### 3.1 Abstract

**Do**: one unstructured paragraph, ~148–155 words. Sentence template: background → gap →
what-was-done (name datasets + task count) → 2 sentences of results with real numbers → one
hedged implication sentence. Past tense reads as the clinical-validation register (safer fit
for us given the alerts evaluation); present tense reads as the methods-paper register.

**Don't**: bold subheadings inside the abstract (not used here, unlike some other journals).
Don't end on an unqualified performance claim — every paper studied closes on a hedge.

Example (Yoon et al. 2025), showing the numbers-mapped-to-cohorts convention and stating
exceptions inline rather than smoothing them over:

> "Our model achieved AUROCs of 0.805, 0.789, and 0.863 for AKI; 0.886, 0.925, and 0.911 for
> PRF; and 0.907, 0.913, and 0.849 for mortality in the derivation cohort and external
> validation cohorts A and B, respectively (all p < 0.001, except for AKI in derivation and PRF
> in cohort B)."

Trade-off framing for a multi-family benchmark (Hegselmann et al. 2026 abstract) — this is the
closest precedent for our own model-family comparison:

> "Overall, we reveal a trade-off between the computational efficiency of specialized EHR
> models and the portability and data independence of LLM-based embeddings."

### 3.2 Results

**Do**: one subsection per experiment/question, not one per model. 4–6 subsections is typical.
Let a lot of methods detail — experimental setup, metric choice and why, operational/alert
threshold definitions, feature-selection procedure — live in Results at the point it's needed,
with a pointer to Methods for parameter values. This is a real, consistent Nature-family
convention, not sloppiness: *Results carries what was measured and why; Methods carries
hyperparameters, splits, statistical tests, ethics, software versions.*

Declarative headings for headline findings, neutral headings for supporting analyses
(Hegselmann et al. 2026): *"General-purpose LLM embeddings rival domain-specific EHR models"*
vs. *"Effect of the EHR serialization strategy..."* — cheap way to signal claim vs. support.

For a design with many cells (multiple datasets × multiple models × multiple outcomes), an
explicit roadmap sentence at the top of Results, matching the Introduction's stated
contributions, helps (Tranchellini et al. 2026):

> "The results are organized into three key sections: the existence and significance of
> distribution shifts between sites, the generalizability of deep learning models across
> diverse ICU cohorts, and the optimal strategies for deploying these models in target
> domains."

**Numbers + CI formatting** — pick one convention and hold it. The most complete real example,
and the best match for our repeated model-vs-model comparisons (Yoon et al. 2025):

> "MT-GBM model demonstrated significantly higher AUROCs on the validation datasets (0.789
> [95% CI: 0.782–0.796] vs. 0.783 [0.776–0.790], p = 0.031 in Validation Cohort A, and 0.863
> [95% CI: 0.850–0.876] vs. 0.826 [0.812–0.840], p < 0.001 in Validation Cohort B)."

Micro-conventions to match: en-dash inside intervals (not hyphen), three decimals for AUROC,
italic *p* with spaces (`p = 0.031`, not `p=0.031`), "95% CI" spelled out on first use.

**Don't** put every number inline if the design is large. Guo et al. 2024 (4 comparisons ×
2–3 datasets × several tasks) puts point estimates in tables and keeps Results prose to
relative claims only — *"13% improvement on average... matched mean AUROC of GBM using as
little as 128 samples."* With four model families, this is likely our situation too: numbers
in tables, prose for the comparative story.

### 3.3 Figures

**Do**:
- Figure 1 = study design / cohort schematic (sometimes two figures: design + architecture).
  This is where our three-dataset design and concept-bottleneck architecture belong.
- One figure = the primary head-to-head comparison as a grid (cohorts × outcomes or models ×
  metrics).
- Push calibration curves, secondary metrics (AUPRC/Brier/F1), hyperparameters, and ablations
  that support-rather-than-headline to Supplementary.
- If reporting a cross-site generalization drop, consider giving it its own main figure rather
  than folding it into a table (Lee et al. 2025, Fig. 5) — Nature-family readers expect
  generalization failure to be shown, not buried.
- For the alerts evaluation specifically, there's a direct precedent (Xie et al. 2025, Fig. 5):
  time-before-event detection rate paired with a false-alarm-rate panel.
- With many cells (4 models × 3 datasets × N tasks), consider a single summary/ranking figure
  with significance brackets rather than requiring readers to parse many small panels
  (Tranchellini et al. 2026, Fig. 7).

**Don't**: expect an Extended Data tier — there isn't one. Everything not in a main figure goes
in the single Supplementary Information PDF.

### 3.4 Limitations

**Do**: end-of-Discussion paragraph(s), enumerated ("First,... Second,... Finally,..."),
each naming the flaw *and* its effect on the claim, followed by a short "In conclusion,"
closer. No separate heading.

The single most transferable sentence for a multi-family comparison (Hegselmann et al. 2026) —
addresses exactly the confound in comparing GBM/tabular-FM/NAM-GAM/hybrid-Mamba against each
other:

> "Comparisons between [model A] and [model B] are confounded by large differences in
> architecture, model scale, and pretraining data, making it difficult to isolate the
> contribution of each factor."

Directly relevant precedent on cross-dataset leakage risk (Tranchellini et al. 2026) — worth
reading in full before writing our own MIMIC-IV/eICU section, since a 2026 npj Digital Medicine
paper explicitly declined to run a transfer they couldn't audit for overlap:

> "We did not evaluate transfers between MIMIC-IV and eICU because possible cross-hospital
> overlap within the United States cannot be audited with the currently available de-identified
> metadata, and we found no authoritative source guaranteeing non-overlap. To minimize the risk
> of data leakage, we focused on transfer directions that are verifiably disjoint by geography
> or curation."

Bounding an interpretability claim short of causality (Xie et al. 2025) — relevant to the
concept-bottleneck lever:

> "The influence of features on predictions does not necessarily correspond to causal
> relationships. Further investigation into causal relationships is needed."

### 3.5 Claim calibration (avoiding overclaiming)

This is the section most directly relevant to the project's existing framing-gates stance.
Five recurring devices, all with real examples above/below worth having open while drafting
Discussion:

1. **Hedge + explicit deferral to prospective validation, same breath as the positive claim.**
   > "Validation on both US and European cohorts supports the potential of this approach to
   > advance clinical decision support in critical care." (Xie et al. 2025 — "supports the
   > potential of," not "demonstrates the clinical utility of")

2. **Scope the claim to the tested setting, in words, not just in a caveat sentence.**
   > "In our experimental setting, pre-trained models are beneficial in highly data-limited
   > settings (up to approximately 20% of the target data)... direct target training becomes
   > beneficial once moderate amounts of data (over 50%) are available." (Tranchellini et al.
   > 2026)

3. **Report where your own method loses, plainly, in main text.**
   > "MEME exhibited strong performance within individual institutions but showed poor
   > generalizability when directly applied across different sites." (Lee et al. 2025 — an
   > entire main figure devoted to their own model's transfer failure)

4. **Interpret non-significance honestly, not as a disguised win.**
   > "No significant differences were observed for the remaining tasks, indicating that most
   > task-level differences were small within statistical uncertainty." (Hegselmann et al. 2026)

5. **Frame the headline conclusion as a trade-off, not a victory** — see the abstract example
   in §3.1.

**Don't** claim "state-of-the-art" as a standalone abstract claim; none of the seven papers do.
Don't omit a "prospective/external validation is still needed" sentence — none of them do
either.

Bonus device worth adopting directly for our own comparison (Guo & Fries et al. 2024) — naming
a way the comparison is conservative *against* your own preferred method builds more trust than
omitting it:

> "Our method of training linear task heads on [our model] across all experiments does not
> involve fine-tuning the foundation model parameters, representing a conservative approach. In
> contrast, the baseline GBMs underwent extensive hyperparameter tuning..."

### 3.6 References

Real range: 41–72, median ~56. More clinical framing → more references (Yoon 72, Goldschmidt
72, both cite extensively for outcome definitions/risk factors); more methods-focused → fewer
(Tranchellini 41, Guo 45). Plan for **55–75** given our clinical-prediction framing.

---

## 4. Submission checklist

- [ ] Article type confirmed as "Article"
- [ ] Abstract ≤150 words, single unstructured paragraph
- [ ] Title ≤15 words
- [ ] Section order: Abstract → Introduction → Results → Discussion → Methods → Data
      availability → Code availability → Author contributions → Competing interests → References
- [ ] No separate Limitations/Conclusions headings — folded into Discussion, closing
      "In conclusion," paragraph
- [ ] Methods subdivided by bold subheadings, no Supplementary Methods
- [ ] TRIPOD+AI followed and cited explicitly in Methods; checklist items 12a–c, 14, 16/20c,
      18e/f, 25, 26, 27a–c specifically addressed
- [ ] Nature Portfolio Reporting Summary completed
- [ ] Data availability: MIMIC-IV, eICU, and GEMINI each named separately with real access
      mechanism and conditions — no bare "available on reasonable request"
- [ ] Code availability: real repository URL(s), specific enough to reproduce reported results
- [ ] Ethics statement: MIMIC-IV, eICU, and GEMINI each cite their own approving IRB/REB
      separately
- [ ] "Use of AI tools" Methods subsection drafted per §2.2, with tool/model named, Red-list
      activities explicitly disclaimed, and any Amber-tier agent contributions (analytical/
      methodological suggestions) honestly disclosed if they occurred
- [ ] Every performance claim in Abstract/Discussion checked against §3.5's calibration
      patterns — no unqualified "validated," "generalizable," or "deployment-ready" language
- [ ] Reference count in the 55–75 range
- [ ] Figures: design/architecture schematic as Fig. 1; primary comparison as a grid figure;
      secondary metrics and ablations moved to the single Supplementary PDF (no Extended Data)
- [ ] Numbers + CI formatting consistent throughout (pick the Yoon-style convention from §3.2
      and hold it everywhere)
