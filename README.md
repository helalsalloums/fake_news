# Arabic Fact Checker

An open-source, evidence-based fact-checking foundation for Modern Standard
Arabic news. It extracts factual claims, searches for external sources, retrieves
article passages, and classifies each claim as `SUPPORTED`, `REFUTED`, or
`NOT_ENOUGH_INFORMATION`.

This is fact verification, not a stylistic “fake-news detector.” The system does
not decide that a story is true because it sounds plausible, and it does not use
a generative LLM. Every verdict is tied to retrieved evidence.

> استناداً إلى الأدلة المتاحة — Based on the available evidence.

## Status

The repository includes the complete modular application, a conservative
rule-based fallback, a training pipeline for a dedicated Arabic evidence
verifier, an RTL web interface, tests, and Docker deployment. The bundled local
fixture is for functional demonstration only; configure a real search provider
before evaluating real claims.

## Architecture

```text
Arabic article or claim
  -> Arabic normalization (original retained)
  -> deterministic claim extraction
  -> Arabic query generation
  -> pluggable web/local search
  -> SSRF-safe document fetch and article extraction
  -> sentence-aware passage retrieval and ranking
  -> trained non-generative NLI verifier + factual comparison rules
  -> conservative verdict aggregation
  -> calibrated model probabilities + evidence-aware confidence + sources
```

The backend defines replaceable interfaces for claim extraction, query
generation, search, fetching, cleaning, retrieval, ranking, vectors,
verification, aggregation, and datasets. FAISS is the initial vector
implementation; lexical retrieval remains available when neural dependencies
are disabled. Compose includes an optional Qdrant service profile for a future
adapter, but the current application does not silently route data to it.

## Quick start with Docker

```bash
cp .env.example .env
docker compose up --build
```

Open:

- Arabic UI: <http://localhost:3000>
- OpenAPI: <http://localhost:8000/docs>
- Health: <http://localhost:8000/api/v1/health>
- Metrics: <http://localhost:8000/metrics>

The default `SEARCH_PROVIDER=local` and `ENABLE_NEURAL_MODELS=false` make the
stack start without credentials, a GPU, or downloaded weights.

After placing a trained checkpoint in `models/verifier`, NVIDIA Container
Toolkit users can enable accelerated encoder inference with:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

This runs the trained non-generative verifier; the project contains no local or
hosted generative LLM integration.

## Local development

Backend:

```bash
python3 -m venv .venv
.venv/bin/pip install -e 'backend[dev]'
cd backend
../.venv/bin/uvicorn app.main:app --reload
```

Install `backend[ml]` as well when training or running the neural verifier.

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## Configuration

Copy `.env.example` and set environment variables. Important settings include:

| Variable | Purpose | Default |
|---|---|---|
| `SEARCH_PROVIDER` | `local`, `brave`, or `searxng` | `local` |
| `BRAVE_SEARCH_API_KEY` | Brave Search credential | unset |
| `SEARXNG_BASE_URL` | Self-hosted SearXNG URL | unset |
| `ENABLE_NEURAL_MODELS` | Load the trained verifier | `false` |
| `VERIFIER_MODEL` | Local/Hugging Face model path | `models/verifier` |
| `EMBEDDING_MODEL` | Configurable embedding checkpoint | `Qwen/Qwen3-Embedding-0.6B` |
| `DATABASE_URL` | Async SQLAlchemy database URL | local SQLite outside Docker |
| `REDIS_URL` | Search/page/result cache and rate limits | unset outside Docker |
| `MODEL_CONFIDENCE_THRESHOLD` | Minimum calibrated stance probability | `0.70` |
| `MODEL_MARGIN_THRESHOLD` | Minimum lead over the next class | `0.20` |
| `EVIDENCE_QUALITY_THRESHOLD` | Minimum evidence score | `0.65` |

Source categories and domain mappings live in
`configs/source_reliability.yaml`. Evidence weights live in
`configs/evidence_scoring.yaml`. Neither file contains political ratings for
individual outlets.

### Search providers

Brave is the recommended hosted search provider. SearXNG is the recommended
self-hosted integration. The former Bing Search API was retired in August 2025,
and Google Custom Search is closed to new customers, so neither is presented as
the default. A legacy Google adapter may be contributed behind the existing
interface for accounts that remain eligible.

To add a provider, implement `SearchProvider.search`, return normalized
`SearchResult` objects, register it in `build_search_provider`, and add mocked
contract tests. Never make tests depend on a live engine.

## API usage

```bash
curl -X POST http://localhost:8000/api/v1/fact-check \
  -H 'Content-Type: application/json' \
  -d '{"text":"أعلنت وزارة الصحة أن عدد الإصابات بلغ 500 حالة","language":"ar"}'
```

Each claim exposes two different confidence concepts:

```json
{
  "verdict": "SUPPORTED",
  "confidence": 0.84,
  "model_verdict": "SUPPORTED",
  "model_confidence": 0.91,
  "class_probabilities": {
    "SUPPORTED": 0.91,
    "REFUTED": 0.03,
    "NOT_ENOUGH_INFORMATION": 0.06
  },
  "evidence_quality": 0.82
}
```

`model_confidence` is the temperature-calibrated classifier probability.
`confidence` is the final evidence-aware confidence after source quality,
relevance, temporal fit, directness, corroboration, duplication, and conflicts
are considered. Neither is the probability that reality itself is true.

Endpoints:

- `POST /api/v1/fact-check`
- `POST /api/v1/claims`
- `GET /api/v1/fact-check/{id}`
- `GET /api/v1/health`
- `GET /api/v1/config`

## Training on Google Colab T4

The verifier is a three-class encoder initialized from
`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`. Evidence is the
premise and the claim is the hypothesis. Training is full fine-tuning, not text
generation.

Review and accept the ARAFA CC-BY-NC-SA-4.0 license, download `ARAFA.json` from
[Zenodo](https://zenodo.org/records/16762969), and run:

```bash
cd backend
pip install -e '.[ml]'
python -m training.prepare \
  --dataset arafa \
  --source /path/to/ARAFA.json \
  --accept-license
python -m training.train --config training/configs/t4.yaml
python -m training.calibrate \
  --checkpoint artifacts/training/verifier-t4/best \
  --dataset artifacts/datasets/arafa
python -m evaluation.evaluate \
  --checkpoint artifacts/training/verifier-t4/best \
  --dataset artifacts/datasets/arafa
python -m training.export \
  --checkpoint artifacts/training/verifier-t4/best
```

The ready-to-run notebook is `notebooks/train_verifier_colab.ipynb`. A
low-memory T4 configuration halves the micro-batch and preserves effective batch
size through gradient accumulation.

To replace the verifier, export a Hugging Face sequence classifier with exactly
the three canonical labels in its `id2label`/`label2id`, add calibration data,
point `VERIFIER_MODEL` to it, and run the verifier contract and calibration
tests.

## Datasets and evaluation

- **ARAFA:** three-class synthetic MSA claim–evidence pairs,
  CC-BY-NC-SA-4.0. Used for research training.
- **AraFacts:** naturally occurring professionally checked claims,
  CC-BY-NC-4.0. Intended for reviewed out-of-domain evaluation.
- **X-FACT:** multilingual natural claims with Arabic coverage, MIT. Its ratings
  require an explicit reviewed mapping before evidence-NLI evaluation.
- **UBC/AraNews:** manipulation/fake-news data. It is a separate binary task and
  must not be reported as three-way evidence verification.

Downloaded datasets and trained weights are ignored by Git. Dataset adapters do
not silently coerce incompatible labels.

```bash
python -m evaluation.evaluate \
  --checkpoint models/verifier \
  --dataset artifacts/datasets/arafa
```

Metrics include accuracy, macro precision/recall/F1, per-class F1, confusion
matrix, Brier score, and expected calibration error. Retrieval recall and
end-to-end coverage should be reported separately from verifier classification.

## Security and privacy

- Only public HTTP(S) URLs on standard ports are fetched.
- DNS answers and every redirect are checked against private, loopback,
  reserved, and link-local networks.
- MIME type, response size, redirect count, and time are bounded.
- Retrieved HTML is untrusted data and is sanitized before extraction.
- There is no prompt-injection execution path because no generative LLM or
  prompt interpreter is used.
- Logs include identifiers and operational counts, not submitted article text.
- Redis provides fail-open rate limiting and bounded caching when configured.

## Testing

```bash
make test
make lint
```

Backend tests use mocked providers and never call live websites. Full model
training is excluded from CI; a miniature adapter/training contract is tested
instead.

## Limitations and known failure cases

- A fact-checking model is not an oracle. It evaluates relationships between a
  claim and the evidence that retrieval found.
- Retrieval may miss relevant Arabic sources, paywalled pages, broadcasts, or
  deleted reports.
- ARAFA is synthetically generated from Wikipedia and may not represent noisy,
  breaking, political, or dialectal news. Strong ARAFA scores do not establish
  real-world readiness.
- NLI models may mishandle negation, coreference, implicit time, numerical units,
  quotations, sarcasm, and dialect vocabulary.
- Source-category scores are administrative priors, not guarantees of accuracy.
- Model confidence can be miscalibrated outside its evaluation distribution.
- Conflicting credible evidence deliberately produces
  `NOT_ENOUGH_INFORMATION` instead of selecting a convenient source.

Production deployments should add locally reviewed domains, continuously audit
calibration and retrieval recall, and provide a path for human review.

## License

Code is licensed under Apache-2.0. Dataset licenses and trained-weight terms are
separate. Weights trained on ARAFA/AraFacts must be treated as research and
non-commercial artifacts unless qualified legal review establishes otherwise.
# fake_news
