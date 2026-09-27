# Uncertainty & Hallucination Detection in LLMs

A small research project investigating whether an LLM's internal confidence signals can predict when it's about to give a wrong answer — with the eventual goal of turning this into a tool that flags low-confidence, likely-incorrect responses.

## Motivation

Language models often answer questions with the same fluent tone whether they're right or hallucinating. If a model's *internal* signals (token probabilities, output diversity, etc.) correlate with actual correctness, those signals could be used to flag risky answers before a user ever sees them — without needing a second model or external fact-check.

## Setup

- **Model:** Llama 3.2-1B
- **Dataset:** SQuAD (free-form question answering)
- **Correctness label:** generated answer vs. gold answer, scored with F1 overlap, thresholded into a binary `correct` (1) / `incorrect` (0) label

## Signals implemented

| Signal | Description |
|---|---|
| **-log(P)** | Negative log-likelihood of the generated answer under the model — lower values indicate the model was more confident in its own tokens |
| *(planned)* Semantic entropy | Diversity across multiple sampled generations for the same question — not yet implemented |

## Analysis

- **Calibration curves** — bin each signal by confidence and plot observed accuracy per bin against the diagonal (perfect calibration). Used to check whether the model is over- or under-confident, not whether it discriminates correct from incorrect.
- **ROC / AUROC** — for each signal, sweep a threshold over the score and compute TPR/FPR at each cutoff to answer the real question: *does a higher score actually rank correct answers above incorrect ones?* AUROC gives a single threshold-independent number per signal for comparison.

