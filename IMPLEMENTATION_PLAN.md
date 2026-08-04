# Implementation Plan

This repository implements an evidence-based Arabic fact-checking system without
generative language models.

1. Build the FastAPI domain model and a runnable local pipeline using Arabic
   normalization, deterministic claim extraction, fixture search, evidence
   scoring, rule-based verification, and calibrated confidence fields.
2. Add secure web retrieval, pluggable search providers, hybrid lexical/vector
   passage retrieval, PostgreSQL persistence, and Redis-backed caching.
3. Fine-tune a multilingual NLI encoder on ARAFA for the three labels
   `SUPPORTED`, `REFUTED`, and `NOT_ENOUGH_INFORMATION`. Supply a reproducible
   single-T4 Colab profile, probability calibration, evaluation, and ONNX export.
4. Add the Arabic RTL React interface, Docker Compose deployment, observability,
   security controls, and comprehensive automated tests.

The model score is always reported separately from evidence-aware system
confidence. Weak, missing, duplicated, stale, or conflicting evidence forces a
conservative `NOT_ENOUGH_INFORMATION` result.
