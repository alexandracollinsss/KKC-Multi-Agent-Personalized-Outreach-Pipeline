# KKC Multi-Agent Outreach Pipeline

A two-agent pipeline that researches a contact, drafts a personalized outreach email, and independently validates every factual claim before the email can be approved to send.

Built for the Kyber Knight Capital Builder-in-Residence engineering assessment.

## Setup

### Requirements

Python 3.10 or later and a Gemini API key.

### Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### API key

Copy the example env file and add your key:

```bash
cp .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY`.

### Optional: KKC profile URL

If you have a public page about KKC, set it:

```bash
export KKC_PROFILE_URL="https://your-public-kkc-page.example"
```

Agent 1 uses this to ground KKC-specific claims in a reachable source rather than citing the contact's own org site.

## How to Run

```bash
python pipeline.py
```

This loads contacts from `test_inputs.json`, runs Agent 1 then Agent 2 for each one, and writes the review digest to `output/review_digest.md`.

### Common flags

| Flag | What it does |
|---|---|
| `--contact 0` | Run only one contact by index (0-based) |
| `--input my_contacts.json` | Use a different contacts file |
| `--skip-agent1` | Reuse existing Agent 1 outputs and re-run only Agent 2 |
| `--skip-agent1 --skip-agent2` | Regenerate the digest from existing outputs without any API calls |
| `--workers 4` | Run contacts in parallel |

### Input format

```json
[
  {
    "name": "Jane Smith",
    "title": "Managing Director",
    "organization": "Meridian Capital Partners",
    "linkedin_url": "https://www.linkedin.com/in/janesmith"
  }
]
```

`linkedin_url` is optional. All other fields are required.

## Model Configuration

The default model is `gemini-flash-latest`, which resolves to the newest available Flash model at runtime. In practice this settles on whichever model in the fallback chain responds first without a rate limit or 503.

The runs in this repo used `gemini-flash-latest` with the following fallback chain, and executed on `gemini-3.1-flash-lite-preview`:

```bash
GEMINI_MODEL=gemini-flash-latest
GEMINI_MODEL_FALLBACKS=gemini-3-flash-preview,gemini-3.1-flash-lite-preview,gemini-2.5-flash,gemini-2.0-flash
```

Optional thinking speed control (low is fastest):

```bash
export GEMINI_THINKING_LEVEL="low"
```

## Speed

A full run for one contact takes roughly 4 to 12 minutes depending on how many sources are reachable and how long Gemini takes on the retrieval packet. Both searches and URL fetches run concurrently via thread pools. The main bottleneck is Gemini latency on large prompts once retrieval completes.

The fastest practical setup:

```bash
export SERPER_API_KEY=...              # Google Search via serper.dev (2500 free/month)
export GEMINI_MODEL=gemini-3.1-flash-lite-preview   # faster than the Pro tier
export FETCH_CACHE_ENABLED=1           # skip re-fetching on reruns
python pipeline.py --workers 3
```

### Cache settings

| Variable | Default | Notes |
|---|---|---|
| `FETCH_CACHE_ENABLED` | `0` | Set to `1` to enable |
| `FETCH_CACHE_TTL_SECONDS` | `3600` | Cache lifetime in seconds |
| `FETCH_CACHE_DIR` | `output/cache` | Where cache files are stored |
| `FETCH_CACHE_MAX_FILES` | `800` | Oldest files pruned beyond this limit |

## Outputs

| File | Description |
|---|---|
| `output/agent1_{Name}.json` | Agent 1 structured research and draft |
| `output/agent2_{Name}.json` | Agent 2 validation result |
| `output/review_digest.md` | Combined review digest for all contacts |
| `output/digests/{Name}.md` | Per-contact digest written as each contact completes |
| `output/pipeline.log` | Full run log with timestamps |

The review digest is the primary deliverable. It shows the approved email or rejection notice, all flags raised with severity, and source links for every claim.

## Tests

```bash
python -m pytest -q
```

The test suite covers confidence floor rules, PASS/WARN/FAIL assignment, source URL validation, and JSON extraction without making any API calls. One integration test (requires `GEMINI_API_KEY`) feeds a known-bad claim through Agent 2 and confirms it produces a FAIL with a BLOCK flag and a null `approved_email`.

## Architecture

```
test_inputs.json
       │
       ▼
┌─────────────────────────────────┐
│  Agent 1: Researcher & Drafter  │  ← agent1_researcher.py
│  • Web search (live)            │
│  • Resolve title, email         │
│  • Draft email (150–250 words)  │
│  • Build claims array           │
└─────────────┬───────────────────┘
              │  writes JSON to disk
              ▼
     output/agent1_{Name}.json
              │
              │  (Agent 2 reads ONLY this file)
              ▼
┌─────────────────────────────────┐
│  Agent 2: Validator             │  ← agent2_validator.py
│  • Re-fetch every source URL    │
│  • Title check (org website)    │
│  • Right contact check          │
│  • Email domain check           │
│  • HTTP 4xx source detection    │
│  • Confidence floor enforcement │
│  • Claim/source alignment check │
└─────────────┬───────────────────┘
              │  writes JSON to disk
              ▼
     output/agent2_{Name}.json
              │
              ▼
┌─────────────────────────────────┐
│  Digest Generator               │  ← digest.py
│  • Approved email or rejection  │
│  • Flags with severity          │
│  • Source links                 │
└─────────────┬───────────────────┘
              ▼
     output/review_digest.md
```

Agent 2 has no access to Agent 1's prompt chain or intermediate reasoning. It reads the JSON file from disk and strips internal metadata keys before passing anything to the model. The separation is enforced by the file boundary rather than by convention.

See `writeup.md` for architecture decisions, model choice rationale, test results, and known failure modes.
