                  
   CLAUDE.md template — specific enough for CS712 right now, structured so you can clone it   
  for any future ML project.                                                                                                                                                                
  # CLAUDE.md                                                                                 

  ## Assignment

  CS712 ML Project: Masked Language Modeling for Hashed Tokens.
  Given a sequence with one MASK position, predict the correct hashed token
  from a predefined eligible set. Self-supervised training (no explicit labels).

  **Phase 1 deadline: 28 March 2026** (public leaderboard, 25 subs/day)
  **Phase 2 deadline: 4 April 2026** (hidden leaderboard, 2 subs total)
  **Report deadline: 4 April 2026** (ACM sigconf LaTeX, max 6 pages)

  **Current best: AbsAcc=TBD / RelAcc=TBD / Overall=TBD**
  **Target: beat designated baseline (TBD once baseline score is known)**

  ---

  ## Metrics — Never Change Evaluation Logic

  | Metric | Formula | Notes |
  |--------|---------|-------|
  | Absolute Accuracy | exact_matches / total | Exact token match only |
  | Relative Accuracy | avg Hamming-distance score | 0 if token not in vocab; penalised by    
  differing bits |
  | **Overall Score** | **harmonic_mean(AbsAcc, RelAcc)** | **Primary metric — maximise** |   

  Hamming score for a prediction p vs ground truth g:
    score = 1 − hamming_distance(p, g) / num_bits
  Only tokens in training vocabulary score > 0 on RelAcc.

  ---

  ## Hard Constraints — Violation = Disqualification

  | Constraint | Value |
  |-----------|-------|
  | Prediction vocabulary | Must come from training vocabulary |
  | Submission format | One prediction per line, same order as test file |
  | Phase 1 limit | 25 submissions per day, 1000 total |
  | Phase 2 limit | 2 submissions total — use them wisely |
  | Report length | Max 6 pages, ACM sigconf LaTeX |
  | Evaluation code | Do not alter |

  ---

  ## Commands

  ```bash
  # Environment
  python -m venv .venv && .venv/Scripts/activate   # Windows
  pip install -r requirements.txt

  # Data analysis (always run first on a new idea)
  python scripts/analyze_data.py

  # Train
  python run_experiment.py --preset bert_base

  # Eval on public test (generates submission file)
  python run_experiment.py --eval-only \
    --checkpoint outputs/checkpoints/<name>/best_model.pt \
    --test-file data/public_test.txt \
    --output outputs/predictions/<name>_public.txt

  # Sweep post-training hyperparams (temperature, threshold)
  python sweep_inference.py \
    --checkpoint outputs/checkpoints/<name>/best_model.pt \
    --preset <name>

  # Ablation suite
  python scripts/run_ablations.py

  # Score locally against public test (if you have ground truth)
  python scripts/score.py \
    --pred outputs/predictions/<name>_public.txt \
    --gold data/public_test_gold.txt

  # Generate analysis plots
  python scripts/plot_results.py --run <name>

  ---
  File Structure

  src/
  ├── config.py              # ALL hyperparameters live here — no magic numbers elsewhere     
  ├── models/
  │   ├── transformer.py     # BERT-style encoder + masked LM head
  │   └── baselines.py       # NGram, frequency-prior, co-occurrence baselines
  ├── data/
  │   ├── dataset.py         # HashedTokenDataset, masking strategy
  │   ├── tokenizer.py       # Vocabulary builder from training data
  │   └── collator.py        # DataCollatorForMLM
  ├── training/
  │   └── trainer.py         # Trainer (standard + self-supervised MLM loop)
  ├── evaluation/
  │   └── evaluator.py       # AbsAcc, RelAcc, Overall; submission generator
  ├── visualization/
  │   └── plotter.py         # Loss curves, token frequency, attention maps
  └── utils.py               # Hamming distance, seed setting, logging

  scripts/
  ├── analyze_data.py        # Token frequency, vocab size, sequence length dist
  ├── analyze_errors.py      # Which tokens/positions the model misses most
  ├── run_ablations.py       # Batch run all ablation configs
  ├── score.py               # Local scoring against gold labels
  └── plot_results.py        # Produce paper-quality figures

  docs/
  ├── EXPERIMENTS.md         # Full run history, scoreboard — LIVING DOCUMENT
  ├── ARCHITECTURE.md        # Model design, param counts, config guide — LIVING
  └── THEORY.md              # Design decision → theoretical justification — STATIC

  resources/
  ├── foundations/
  │   ├── run_history.md     # Deep code change log per run — LIVING DOCUMENT
  │   └── style.md           # Code conventions reference
  └── reference/
      └── papers/            # PDFs of relevant papers

  data/                      # Raw data (gitignored)
  ├── train.txt
  ├── eligible_tokens.txt
  ├── public_test.txt
  └── private_test.txt       # Released 29 Mar

  outputs/                   # All generated artifacts (gitignored)
  ├── checkpoints/
  ├── predictions/
  ├── plots/
  └── logs/

  report/
  └── main.tex               # ACM sigconf LaTeX source

  When modifying model architecture → read docs/ARCHITECTURE.md
  When debugging training → read docs/EXPERIMENTS.md
  When justifying a design choice → read docs/THEORY.md
  After every run (code change OR config only) → update resources/foundations/run_history.md  

  ---
  Living Documents — What to Update and When

  CLAUDE.md — update after every experiment

  ┌─────────────────────────────┬──────────────────────────────────────┐
  │            Field            │             Update when              │
  ├─────────────────────────────┼──────────────────────────────────────┤
  │ Current best line           │ new best Overall score               │
  ├─────────────────────────────┼──────────────────────────────────────┤
  │ Experiment presets table    │ new preset added or score filled in  │
  ├─────────────────────────────┼──────────────────────────────────────┤
  │ Current Experiments section │ run completes; replace with next run │
  └─────────────────────────────┴──────────────────────────────────────┘

  docs/EXPERIMENTS.md — update after every run

  ┌──────────────────┬────────────────────────────────────────────────────┐
  │     Section      │                    Update when                     │
  ├──────────────────┼────────────────────────────────────────────────────┤
  │ Scoreboard table │ run completes — add AbsAcc / RelAcc / Overall      │
  ├──────────────────┼────────────────────────────────────────────────────┤
  │ New run section  │ after run: what changed, what worked, what failed  │
  ├──────────────────┼────────────────────────────────────────────────────┤
  │ Error analysis   │ after any evaluation — which token types fail most │
  ├──────────────────┼────────────────────────────────────────────────────┤
  │ Next experiment  │ replace with new next run once current starts      │
  └──────────────────┴────────────────────────────────────────────────────┘

  Template for a new run section:
  ## run_N — <one-line description>

  **What changed from run_(N-1):** ...

  **What worked:** ...

  **What failed:** ...

  **Key diagnostic values:**
  - Val loss: X
  - AbsAcc (val): X%
  - RelAcc (val): X%
  - Overall (val): X%
  - Public leaderboard Overall: X% (submission #N)
  - Token-level: top-5 misses = [token_a, token_b, ...]
  - Rare token (<5 occurrences) accuracy: X%

  docs/ARCHITECTURE.md — update when code structure changes

  Update parameter counts, model component breakdown, and AMP/precision notes
  whenever models/transformer.py changes significantly.

  resources/foundations/run_history.md — update after every run

  Deep code change log. Records exactly what code changed per run (not just config).
  Essential for reproducing results. Config-only runs must say "Code: None changed."

  docs/THEORY.md — mostly static

  Only update when a new theoretical connection becomes relevant to a code decision.

  ---
  Experiment Presets (src/config.py)

  ┌─────────────────┬────────┬────────┬─────────┬──────────────────────────────────────────┐  
  │     Preset      │ AbsAcc │ RelAcc │ Overall │                  Notes                   │  
  ├─────────────────┼────────┼────────┼─────────┼──────────────────────────────────────────┤  
  │ frequency_prior │ TBD    │ TBD    │ TBD     │ Predict most-frequent eligible token     │  
  │                 │        │        │         │ always                                   │  
  ├─────────────────┼────────┼────────┼─────────┼──────────────────────────────────────────┤  
  │ ngram           │ TBD    │ TBD    │ TBD     │ N-gram co-occurrence baseline            │  
  ├─────────────────┼────────┼────────┼─────────┼──────────────────────────────────────────┤  
  │ bert_tiny       │ TBD    │ TBD    │ TBD     │ 2L/128H transformer, fast iteration      │  
  ├─────────────────┼────────┼────────┼─────────┼──────────────────────────────────────────┤  
  │ bert_base       │ TBD    │ TBD    │ TBD     │ 6L/256H, main model                      │  
  ├─────────────────┼────────┼────────┼─────────┼──────────────────────────────────────────┤  
  │ bert_large      │ TBD    │ TBD    │ TBD     │ 12L/512H, if compute allows              │  
  └─────────────────┴────────┴────────┴─────────┴──────────────────────────────────────────┘  

  ---
  Current Experiments in Flight

  baseline_analysis — FIRST — understand data before modelling

  Run scripts/analyze_data.py to establish:
  - Vocabulary size, eligible token count
  - Token frequency distribution (Zipfian? Uniform?)
  - Sequence length statistics (min/max/mean)
  - Masked position distribution (uniform? positional bias?)
  - Co-occurrence patterns between eligible tokens and context

  This determines model size, masking strategy, and which baselines to build.

  ---
  Code Style

  Clean Architecture + SOLID + Gang of Four patterns. Same as CS424.

  - Factory for all creation (ModelFactory, TokenizerFactory, OptimizerFactory)
  - Strategy for swappable algorithms (MaskingStrategy, InferenceStrategy)
  - Template Method for fixed workflows with swappable steps (Trainer._run_epoch)
  - OCP: extend by adding new class, never by modifying existing
  - DIP: depend on abstractions, inject via constructor

  Type hints everywhere. Google-style docstrings on public APIs. Max 400 lines/file.
  No magic numbers — every hyperparameter lives in src/config.py.

  ---
  Submission Checklist

  Before every Phase 1 submission:
  - Model trained to convergence (val loss plateaued)
  - All predictions are from training vocabulary (no OOV predictions)
  - Prediction count == test sentence count
  - Local score computed with scripts/score.py (if gold available)
  - Submission logged in docs/EXPERIMENTS.md with submission number

  Before Phase 2 (only 2 shots — treat like a finals):
  - Best Phase 1 checkpoint identified
  - Post-hoc inference sweep done (temperature, beam search, re-ranking)
  - Ensemble attempted if time permits
  - Both submissions planned: [best_single_model, ensemble_or_best_variant]

  ---
  Report Structure (ACM sigconf, 6 pages)

  1. Introduction (0.5p) — problem, why it's interesting, your approach in one paragraph      
  2. Related Work (0.5p) — MLM (BERT), hashed/anonymous token work, constrained decoding      
  3. Method (1.5p) — model architecture, training objective, inference strategy, any novel    
  contribution
  4. Experiments (2p) — ablation table, model comparison, error analysis, figures
  5. Analysis (0.75p) — what the model learned, failure modes, rare token behaviour
  6. Conclusion (0.25p) — findings, limitations, future work

  Figures to include (generate with scripts/plot_results.py):
  - Token frequency distribution (motivates the rare-token problem)
  - Learning curves for key runs
  - Ablation bar chart (AbsAcc / RelAcc / Overall per variant)
  - Confusion matrix or per-token accuracy heatmap
  - Attention visualisation (if transformer-based)

  ---
  Theory Reference

  ┌────────────────────────────┬────────────────────────────────────────────────────────┐     
  │          Concept           │           What it controls in this codebase            │     
  ├────────────────────────────┼────────────────────────────────────────────────────────┤     
  │ MLM objective              │ Self-supervised pretraining signal; no labels needed   │     
  ├────────────────────────────┼────────────────────────────────────────────────────────┤     
  │ Hamming distance           │ RelAcc metric; also useful for re-ranking candidates   │     
  ├────────────────────────────┼────────────────────────────────────────────────────────┤     
  │ Class imbalance            │ Rare eligible tokens; consider frequency-weighted loss │     
  ├────────────────────────────┼────────────────────────────────────────────────────────┤     
  │ Transformer attention      │ Context aggregation; positional encoding choice        │     
  ├────────────────────────────┼────────────────────────────────────────────────────────┤     
  │ Temperature scaling        │ Inference calibration; sweep after training            │     
  ├────────────────────────────┼────────────────────────────────────────────────────────┤     
  │ Constrained decoding       │ Restrict output logits to eligible token set only      │     
  ├────────────────────────────┼────────────────────────────────────────────────────────┤     
  │ Byte-pair / hash collision │ Why tokens cluster; informs masking strategy           │     
  └────────────────────────────┴────────────────────────────────────────────────────────┘     

  ---

  **How to adapt this to any future ML project:**

  1. Replace the Assignment section with course/task/metric
  2. Fill the Hard Constraints table with whatever the graders enforce
  3. Rewrite the Commands section with your actual scripts
  4. Keep the Living Documents structure — it's what keeps Claude useful across sessions      
  5. Replace the Theory Reference with whichever concepts govern your design decisions        
  6. The Submission Checklist pattern works for any leaderboard competition

  The three things that made CS424 work well with Claude Code: (1) `CLAUDE.md` always has     
  current best score + current experiment, (2) `docs/EXPERIMENTS.md` has the full run history 
  so Claude never repeats mistakes, (3) every config lives in one file so Claude doesn't have 
  to grep for magic numbers.
