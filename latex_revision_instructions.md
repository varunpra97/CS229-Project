# LaTeX Revision Instructions — PAC-Bayesian Bounds for Weight-Decomposed LoRA

**Target paper:** `PAC-Bayesian Generalization Bounds for Weight-Decomposed Low-Rank Adaptation` (Zhang, Prakash, Vasanawala), submitted as ICML-workshop length.

**Purpose of this file:** instruct Claude Code on the edits needed so the paper's *claims* match what the *figures and tables actually demonstrate*. The math in the proofs is correct and should not be changed (except cross-reference type names — see §A.0). The problem is framing: several figures verify the geometric machinery the theorems rest on, but the paper writes as if they verify the headline generalization and DoRA-advantage claims.

> **Important — I do not have the `.tex` source, only the compiled PDF.** Every edit below is anchored to a **quoted phrase** that should appear verbatim (modulo LaTeX commands) in the source. Locate each edit site by grepping for its quoted anchor. Before any multi-sentence rewrite, show the matched source block and the proposed replacement, then apply. Make small, localized diffs. After all edits, recompile and confirm zero `undefined reference` / `multiply-defined` warnings.

---

## How to read the priorities

- **Section A — pure LaTeX edits (do these now).** No new data or figures required. These are the core changes that align claims with evidence, plus genuine cross-reference bugs.
- **Section B — edits requiring new data or figure regeneration (need Soham's inputs).** Templates and input checklists are provided; do *not* fabricate numbers. Flag these for Soham and stop at the point where real data is required.

Confidence: Section A edits are high-confidence and safe. Section B is higher-value for a reviewer but higher-effort and easy to get wrong (see the raw-vs-calibrated trap in §B.1).

---

# Section A — Pure LaTeX edits (do now)

## A.0 Fix cross-reference type names (genuine bug)

Several `\ref`s name the wrong environment type. In the PDF, **Lemmas 3.2 and 3.3** and **Remarks 3.5 and 3.13** are all cited as "Theorem." This is almost certainly hardcoded `Theorem~\ref{...}` instead of `\cref{...}`, or a `cleveref` setup that does not distinguish environments.

Fix each of these (grep the anchor, correct the type word, or switch to `\cref`):

| Location (anchor phrase) | Currently says | Should say |
|---|---|---|
| Proof of Thm 3.4: `By Theorem 3.2, $C^{\mathrm{DoRA}}_{\mathrm{dir}}$` | Theorem 3.2 | **Lemma** 3.2 |
| Proof of Thm 3.4: `the two bounds on $A_d(\kappa)$ from Theorem 3.3` | Theorem 3.3 | **Lemma** 3.3 |
| §4.2: `Equivalently (Theorem 3.5), the asymptotic approximation` | Theorem 3.5 | **Remark** 3.5 |
| §4.3: `But Theorem 3.13 shows the identity` | Theorem 3.13 | **Remark** 3.13 |
| §4.3: `consistent with Theorem 3.12/Theorem 3.13` | second ref | **Remark** 3.13 |
| §3.1 / Lemma 3.2 references inside proofs | check any `Theorem 3.2` | **Lemma** 3.2 |

**Preferred fix:** replace literal type words with `\cref{<label>}` and let `cleveref` print the type, so this cannot drift again. If `cleveref` is already loaded, just swap `Theorem~\ref{lem:vmfkl}` → `\cref{lem:vmfkl}` etc. Verify each `\label` resolves to the environment type claimed above.

**Also verify numbering continuity.** The theorem-counter jumps 3.5 → 3.10 with no 3.6–3.9 in the text. Either four environments were deleted (leaving a gap that's fine) or there is a stray `\setcounter`/`\addtocounter`. Check for an orphaned counter manipulation and remove it if unintended; otherwise leave the gap.

---

## A.1 State that the Theorem 3.4 bracket holds *by construction*

**Issue:** "0/2592 violations" is presented as empirical confirmation of Theorem 3.4. But given Lemma 3.3 ($\tfrac{\kappa}{d+\kappa}\le A_d(\kappa)\le 1$) and $S=\sum_j(1-\cos\theta_j)\ge 0$, multiplying by $\kappa S$ makes the bracket an algebraic identity. It *cannot* be violated. What the panel actually validates is the numerics (the Olver large-order expansion for $A_d(\kappa)$ not underflowing at $d=3072$).

**Edit 1 — Remark 3.5 / §4.2 body.** After the sentence anchored by `holds with \textbf{0 violations} across all 2592 layers`, insert:

> Because the bracket follows algebraically from Lemma~3.3 and $S\ge 0$, it cannot be violated by construction; the zero-violation count therefore confirms that the Olver large-order expansion for $A_d(\kappa)$ remains numerically stable at $d=768,3072$ (no Bessel underflow) rather than confirming the inequality itself. The substantive content of this panel is the $1/A_d(\kappa)$ tightening factor below.

**Edit 2 — Figure 1 caption (right panel).** Find the caption fragment `0/2592 violations overall, hugging the lower bound` and append: `(the bracket holds by construction; the panel demonstrates numerical stability and the magnitude of the asymptotic-to-exact gap).`

> Net effect: keep the genuinely valuable result (the 77–307× tightening / "hugs the lower bound", which *is* a real quantitative payoff), but stop framing an algebraic identity as an empirical test.

---

## A.2 Re-frame Theorem 3.12 ("DoRA Advantage")

**Issue:** Read literally, Theorem 3.12 gives **MAP** the smaller upper bound ($\tfrac{s}{d}$ vs $s$, compounded by $A_k$ vs $A_d$), and Remark 3.13 / Table 1 confirm raw directional KL favors MAP by a factor $\sim d$ in *every* regime. So the theorem as stated does not establish a DoRA advantage. The advantage is rescued only via the dimension-calibration + "discriminability" argument. The theorem *title* "DoRA Advantage Under Localized Updates" overpromises relative to its own proof.

**Edit 1 — theorem title.** Find `\begin{theorem}[DoRA Advantage Under Localized Updates]` (or the `\caption`/title text `DoRA Advantage Under Localized Updates`) and rename to a neutral statement of what is proved:

> `Localized-Update Complexity Bounds`

**Edit 2 — add one clarifying sentence to the theorem's surrounding text or to Remark 3.13.** After the Remark 3.13 sentence anchored by `MAP's raw directional KL is always a factor $\sim d$ smaller`, ensure the following point is explicit (it is half-present; make it unambiguous):

> Consequently the raw bounds of this theorem favor MAP in *every* regime; the DoRA advantage is a statement about *dimension-calibrated discriminability* (below), not about the raw upper bounds.

**Edit 3 — conclusion sentence (the most important reframe).** Find the conclusion fragment:

> `effective sparsity separates the DoRA-favored (sparse) and MAP-favored (dense) regimes exactly as the comparison predicts`

Replace with:

> effective sparsity separates sparse (RTE) from dense (SST-2/QNLI/MNLI) regimes — the precondition under which Theorem~3.12 predicts a DoRA advantage in dimension-calibrated discriminability.

> Rationale: the data establishes the *regime*, and the *theory* maps regime → favored method. The paper currently states it as if a DoRA advantage were directly measured.

---

## A.3 Acknowledge that the discriminability metric $D$ is near-tautological w.r.t. sparsity

**Issue:** $D = (\text{fraction of complexity in active columns})/(s/d)$ is high *iff* complexity concentrates in few columns — which *is* the definition of a sparse update. So "$D$ separates RTE from the dense tasks" is close to definitional, and $D$ measures a property of the *update*, not a DoRA *outcome*.

**Edit — §4.3.** After the sentence introducing $D$ (anchor: `the ratio of complexity mass captured to the column budget spent`), add:

> By construction $D$ is large precisely when directional complexity concentrates in few columns, i.e. when the update is sparse; it therefore measures the update geometry (the precondition of Theorem~3.12) rather than an independently realized DoRA bound. We report it as evidence that the tasks fall into the predicted regimes, not as a direct measurement of a tighter DoRA bound.

> Keep the table and the result — just label what $D$ is and is not.

---

## A.4 Soften the generalization framing (title/abstract/intro/conclusion)

**Issue:** Title and abstract promise *generalization bounds*, but no figure measures test performance, a generalization gap, or the realized PAC-Bayes risk from Theorem 3.1. §4's opening is honestly scoped (`we now verify that the geometric quantities they rest on behave as claimed`) — that honesty is exactly the gap. Everything downstream of $\mathrm{KL}(Q\|P)$ is theory, not evidence, in the current draft.

Two acceptable strategies. **Either** (i) keep the title and add explicit scoping sentences (lower-effort, recommended for a workshop), **or** (ii) also add a realized-bound number (Section B.1) to partially close the gap. Do (i) regardless; do (ii) if Soham supplies data.

**Edit 1 — abstract.** After the clause anchored by `verify each on RoBERTa-base fine-tuned across four GLUE tasks`, add a short scoping clause:

> ; the verification targets the geometric complexity quantities entering the bounds (angular KL, the comparison identity, effective sparsity), not realized test risk, which we leave to the realized-bound estimates discussed below.

(If Section B.1 is *not* added, change the trailing clause to "…not realized test risk, which we leave to future work.")

**Edit 2 — §4 opening (keep, don't delete).** The sentence `The theorems are proven; we now verify that the geometric quantities they rest on behave as claimed on real fine-tuned weights.` is good. Immediately after it, add a forward pointer:

> We do not, in the present draft, evaluate the realized McAllester bound of Theorem~3.1 end-to-end; §[B-ref] reports a directional-only estimate of its magnitude, and a magnitude-inclusive bound is left to future work.

(If Section B.1 is not added, drop "§[B-ref] reports … and".)

**Edit 3 — conclusion limitations.** The existing closing sentence `Calibrated, magnitude-inclusive estimators and domain-shift settings are left to future work.` should be expanded to name the untested object explicitly:

> The realized risk bound of Theorem~3.1 — which requires the magnitude-component KL omitted here — and domain-shift settings are left to future work; our experiments verify the geometric inputs to the bound, not the bound's predictive tightness against measured test error.

---

## A.5 Note that the $R^2=0.995$ in Figure 1 (left) is structurally expected

**Issue:** Both axes of Figure 1 (left) are the *same* quantity up to the raw-vs-unit flattening offset, so high $R^2$ is structurally guaranteed and is not independent evidence. The real verification of the identity is the machine-precision residual ($2.2\times10^{-16}$) on the unit-column form.

**Edit — §4.1.** After the fragment `with $R^2 = 0.995$ (Pearson $r = 0.997$)`, add a parenthetical:

> (the high $R^2$ is expected, since both axes are the same quantity up to the flattening offset; the load-bearing check is the $2.2\times10^{-16}$ unit-column residual above, not the regression fit)

Also: the present draft buries the actual theorem verification (`max residual $2.2\times10^{-16}$`) in prose while the figure shows the *non-identity* raw version. If Figure 1 (left) is **not** regenerated (Section B.2), at minimum promote the machine-precision sentence so it is not subordinate to the regression.

---

# Section B — Edits requiring new data or figure regeneration

> These need real inputs. Do **not** invent numbers. Where a value is unknown, insert a clearly-marked placeholder like `\todo{n_RTE = ?}` and list it in the checklist for Soham.

## B.1 (High value) Add a realized PAC-Bayes bound number

**Goal:** turn "geometric verification" into at least a partial "generalization bound" by plugging a KL value into the McAllester bound (Theorem 3.1) and reporting the resulting risk bound — demonstrating the bound is **non-vacuous**.

**The McAllester form (already in the paper, Theorem 3.1):**

$$R(Q)\ \le\ \hat R(Q)\ +\ \sqrt{\frac{\mathrm{KL}(Q\|P)+\ln\frac{2\sqrt n}{\delta}}{2n}}.$$

**Worked illustration (MNLI / DoRA), using only numbers already in the paper plus standard GLUE sizes:**

- KL ← directional non-asymptotic complexity = `46.30` (Table 1, MNLI/DoRA).
- $n$ = `392{,}702` (MNLI train), $\delta=0.05$.
- $\ln(2\sqrt n/\delta)=\ln(25066)\approx 10.13$.
- complexity term $=\sqrt{(46.30+10.13)/(2\cdot392702)}=\sqrt{7.19\times10^{-5}}\approx \mathbf{0.0085}$.

So the PAC-Bayes penalty adds ≈ 0.0085 to the empirical risk — tight and non-vacuous. That single number, reported for one or two tasks, materially changes a reviewer's read.

**Two caveats that MUST be stated in the text if this is added (do not skip):**

1. **Directional-only ⇒ optimistic / a lower bound.** This uses the *directional* KL only. The full KL also includes the magnitude-component KL (DoRA updates $m_j$ directly; MAP has $\alpha,\beta$), which the paper explicitly defers to future work. Since $\mathrm{KL}_{\text{full}}\ge \mathrm{KL}_{\text{dir}}$, the reported bound is a **lower bound on the true complexity term** and must be labeled "directional-only."
2. **Do NOT run DoRA-vs-MAP as a raw-bound horse race.** On *raw* directional KL, $C^{\mathrm{DoRA}}\approx d\,C^{\mathrm{MAP}}$ (Remark 3.13), so a naive realized-bound comparison makes **MAP look uniformly tighter, including on RTE** — directly contradicting the paper's narrative. (Check: RTE/DoRA raw KL ≈ 0.20, RTE/MAP ≈ 0.017 → MAP's penalty is smaller.) The DoRA advantage lives only in *dimension-calibrated discriminability* (§4.3), not in the raw realized bound. So report the realized bound to establish **non-vacuity**, and keep the DoRA-vs-MAP comparison as discriminability — do not claim DoRA gives a numerically tighter realized bound.

**Suggested placement:** a short paragraph at the end of §4.2, e.g. titled `Realized bound (directional-only).` Provide the table/sentence; mark `\hat R(Q)` as a placeholder until Soham supplies training risk.

**Inputs to request from Soham (checklist):**
- [ ] Empirical risk $\hat R(Q)$ (train error or train loss in $[0,1]$) for the task(s) reported — at least MNLI/DoRA, ideally RTE too.
- [ ] Confirm $n$ per task (RTE ≈ 2{,}490; SST-2 ≈ 67{,}349; QNLI ≈ 104{,}743; MNLI ≈ 392{,}702 — confirm exact values used).
- [ ] Chosen $\delta$ (default 0.05).
- [ ] Decision: report directional-only (fast, with caveat) **or** wait for magnitude-inclusive KL (stronger, but that's the deferred future-work item).

## B.2 (Medium value) Re-plot Figure 1 (left) in unit-column form

**Issue:** The figure is labeled as verifying an *exact identity* but plots the **raw** (non-unit) flattening, landing at slope **0.889** — a visible 11% departure from the $y=x$ it's drawn against. The unit-column form sits on $y=x$ to machine precision. A referee will flag a figure that claims an identity while deviating from it.

**Recommended:** make the **unit-column** quantity the main panel (lands on $y=x$, residual $2.2\times10^{-16}$); demote the raw-flattening version to an inset or appendix. This requires re-running the plotting/analysis script (a code change, not a LaTeX change).

**LaTeX-only portion Claude Code can do now** (the rest needs the regenerated figure file from Soham):
- Update the Figure 1 caption (left) once the panel is unit-column: replace `local sum vs. dimension-scaled global term, on $y = x$ ($R^2$=0.995, slope 0.89)` with `local sum vs. dimension-scaled global term in unit-column form, on $y=x$ to machine precision (max residual $2.2\times10^{-16}$); raw-flattening version inset.`
- If the figure is *not* regenerated, at minimum change the caption/body to state the plotted quantity is the **raw** flattening and that the 11% slope is the raw-vs-unit offset (currently asserted; see A.5), and move the machine-precision identity to the foreground.

**Note on the "11% = exactly the flattening offset" claim.** The draft asserts the 0.889 slope is *exactly* the raw-vs-unit offset. This is plausible but not shown — the raw global cosine is a *magnitude-weighted* average of $\cos\theta_j$, so the slope depends on the magnitude–angle correlation. If keeping the raw panel, either (a) soften "exactly" to "consistent with," or (b) add a one-line derivation/figure in an appendix showing the weighting reproduces 0.889. Flag this for Soham.

**Inputs to request from Soham (checklist):**
- [ ] Regenerated Figure 1 (left) using unit-column $U,\hat U$ (or an inset pairing unit + raw).
- [ ] Decision on the "exactly the offset" claim: soften wording, or add the weighted-average justification.

---

# Final verification checklist (run after edits)

- [ ] Recompile; **zero** `undefined references` and **zero** `multiply-defined labels`.
- [ ] All cross-refs in §A.0 now print the correct environment type (Lemma/Remark, not Theorem). Spot-check each in the compiled PDF.
- [ ] Theorem-counter gap 3.6–3.9 resolved or confirmed intentional.
- [ ] Title/abstract/intro/conclusion no longer imply realized generalization is empirically verified (A.4) — unless B.1 was added, in which case the realized-bound paragraph is present *with* both caveats.
- [ ] Figure 1 right caption notes the bracket holds by construction (A.1).
- [ ] Theorem 3.12 renamed; conclusion sentence softened to "precondition for the predicted advantage" (A.2).
- [ ] $D$ is labeled as an update-geometry / sparsity measure, not a measured DoRA bound (A.3).
- [ ] No fabricated numbers: every value is either from the paper, a confirmed standard dataset size, or a clearly-marked `\todo{...}` placeholder.
- [ ] Diffs are minimal and localized; proof math (other than ref type names) is unchanged.
