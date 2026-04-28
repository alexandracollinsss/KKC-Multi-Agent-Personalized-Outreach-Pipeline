"""
KKC Multi-Agent Outreach Pipeline
Orchestrates Agent 1 (Researcher/Drafter) → Agent 2 (Validator) → Digest

Usage:
    python pipeline.py                          # run all contacts in test_inputs.json
    python pipeline.py --contact 0              # run only the first contact (0-indexed)
    python pipeline.py --input my_contacts.json # use a different input file
    python pipeline.py --skip-agent1            # reuse existing Agent 1 outputs (re-validate only)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent1_researcher import run_agent1
from agent2_validator import run_agent2
from digest import generate_digest

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

Path("output").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("output/pipeline.log"),
    ],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logger = logging.getLogger("pipeline")

OUTPUT_DIR = Path("output")


def ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


def load_contacts(path: str) -> list:
    with open(path) as f:
        contacts = json.load(f)
    if isinstance(contacts, dict):
        contacts = [contacts]
    logger.info(f"Loaded {len(contacts)} contact(s) from {path}")
    return contacts


def safe_filename(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")


def run_pipeline(
    contacts: list,
    skip_agent1: bool = False,
    skip_agent2: bool = False,
    model: str = "gemini-flash-latest",
    digest_path: str | None = None,
    workers: int = 1,
):
    ensure_output_dir()
    contacts = list(contacts)

    agent1_outputs = []
    agent2_outputs = []
    errors = []
    digest_out = digest_path or str(OUTPUT_DIR / "review_digest.md")
    digests_dir = OUTPUT_DIR / "digests"
    digests_dir.mkdir(exist_ok=True)

    def _write_digest_best_effort() -> None:
        try:
            logger.info(f"[pipeline] Generating digest → {digest_out}")
            generate_digest(agent1_outputs, agent2_outputs, output_path=digest_out)
        except Exception as e:
            logger.warning(f"[pipeline] Digest generation failed: {e}")

    def _write_person_digest_best_effort(name: str, a1: dict, a2: dict) -> None:
        try:
            safe = safe_filename(name)
            out_path = digests_dir / f"{safe}.md"
            logger.info(f"[pipeline] Generating per-person digest → {out_path}")
            generate_digest([a1], [a2], output_path=str(out_path))
        except Exception as e:
            logger.warning(f"[pipeline] Per-person digest generation failed for {name}: {e}")

    def _process_contact(contact: dict) -> tuple[str, dict, dict]:
        name = contact.get("name", "Unknown")
        safe = safe_filename(name)
        a1_path = OUTPUT_DIR / f"agent1_{safe}.json"
        a2_path = OUTPUT_DIR / f"agent2_{safe}.json"

        # ── Agent 1 ──────────────────────────────────────────────
        if skip_agent1 and a1_path.exists():
            logger.info(f"[pipeline] Reusing existing Agent 1 output for {name}")
            a1_result = json.loads(a1_path.read_text())
        else:
            logger.info(f"[pipeline] Running Agent 1 for {name}")
            a1_result = run_agent1(contact, model=model)
            a1_path.write_text(json.dumps(a1_result, indent=2))
            logger.info(f"[pipeline] Agent 1 output written → {a1_path}")

        if a1_result.get("error"):
            logger.warning(f"[pipeline] Agent 1 errored for {name}, skipping Agent 2")
            stub_a2 = {
                "name": name,
                "validation_result": "FAIL",
                "flags": [{
                    "claim_id": "system",
                    "flag_type": "unverifiable",
                    "detail": f"Agent 1 failed: {a1_result['error']}",
                    "severity": "BLOCK",
                }],
                "approved_email": None,
                "validation_notes": "Skipped because Agent 1 produced no output.",
                "_research_sources": [],
                "_claims": [],
            }
            a2_path.write_text(json.dumps(stub_a2, indent=2))
            _write_person_digest_best_effort(name, a1_result, stub_a2)
            return name, a1_result, stub_a2

        # ── Agent 2 ──────────────────────────────────────────────
        if skip_agent2 and a2_path.exists():
            logger.info(f"[pipeline] Reusing existing Agent 2 output for {name}")
            a2_result = json.loads(a2_path.read_text())
        else:
            logger.info(f"[pipeline] Running Agent 2 for {name}")
            a2_result = run_agent2(a1_path, model=model)
            a2_path.write_text(json.dumps(a2_result, indent=2))
            logger.info(f"[pipeline] Agent 2 output written → {a2_path}")

        _write_person_digest_best_effort(name, a1_result, a2_result)
        return name, a1_result, a2_result

    def _make_error_stub(contact: dict, err: Exception) -> tuple[str, dict, dict]:
        cname = contact.get("name", "Unknown")
        corg = contact.get("organization", "Unknown")
        a1_err = {
            "name": cname,
            "organization": corg,
            "error": str(err),
            "resolved_title": None,
            "resolved_email": None,
            "email_domain_source": None,
            "draft_email": None,
            "research_sources": [],
            "claims": [],
        }
        a2_err = {
            "name": cname,
            "validation_result": "FAIL",
            "flags": [{
                "claim_id": "system",
                "flag_type": "unverifiable",
                "detail": f"Pipeline error: {err}",
                "severity": "BLOCK",
            }],
            "approved_email": None,
            "validation_notes": f"Contact skipped due to unhandled pipeline error: {err}",
            "_research_sources": [],
            "_claims": [],
        }
        return cname, a1_err, a2_err

    try:
        if workers <= 1:
            for c in contacts:
                try:
                    name, a1r, a2r = _process_contact(c)
                except Exception as exc:
                    cname = c.get("name", "Unknown")
                    logger.error(f"[pipeline] Unhandled error for {cname}: {exc}", exc_info=True)
                    name, a1r, a2r = _make_error_stub(c, exc)
                    errors.append({"contact": name, "error": str(exc)})
                agent1_outputs.append(a1r)
                agent2_outputs.append(a2r)
                if a1r.get("error"):
                    errors.append({"contact": name, "agent": "agent1", "error": a1r["error"]})
                elif a2r.get("validation_result") == "FAIL":
                    errors.append({"contact": name, "agent": "agent2", "flags": a2r.get("flags", [])})
        else:
            ordered_results: list[tuple[str, dict, dict] | None] = [None] * len(contacts)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_process_contact, c): idx for idx, c in enumerate(contacts)}
                for fut in as_completed(futs):
                    idx = futs[fut]
                    try:
                        ordered_results[idx] = fut.result()
                    except Exception as exc:
                        c = contacts[idx]
                        cname = c.get("name", "Unknown")
                        logger.error(f"[pipeline] Unhandled error for {cname}: {exc}", exc_info=True)
                        ordered_results[idx] = _make_error_stub(c, exc)
            for item in ordered_results:
                if item is None:
                    continue
                name, a1r, a2r = item
                agent1_outputs.append(a1r)
                agent2_outputs.append(a2r)
                if a1r.get("error"):
                    errors.append({"contact": name, "agent": "agent1", "error": a1r["error"]})
                elif a2r.get("validation_result") == "FAIL":
                    errors.append({"contact": name, "agent": "agent2", "flags": a2r.get("flags", [])})
    finally:
        _write_digest_best_effort()

    # ── Summary ──────────────────────────────────────────────────
    total = len(contacts)
    passed = sum(1 for r in agent2_outputs if r.get("validation_result") == "PASS")
    warned = sum(1 for r in agent2_outputs if r.get("validation_result") == "WARN")
    failed = sum(1 for r in agent2_outputs if r.get("validation_result") == "FAIL")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Contacts processed : {total}")
    print(f"  PASS               : {passed}")
    print(f"  WARN               : {warned}")
    print(f"  FAIL               : {failed}")
    print(f"  Review digest      : {digest_out}")
    print("=" * 60 + "\n")

    if errors:
        print("Errors logged:")
        for e in errors:
            print(f"  - {e['contact']}: {e.get('error') or e.get('flags')}")

    return agent1_outputs, agent2_outputs


def main():
    parser = argparse.ArgumentParser(description="KKC Multi-Agent Outreach Pipeline")
    parser.add_argument("--input", default="test_inputs.json", help="Path to contacts JSON file")
    parser.add_argument("--contact", type=int, default=None, help="Run only one contact by index (0-based)")
    parser.add_argument("--skip-agent1", action="store_true", help="Reuse existing Agent 1 outputs")
    parser.add_argument("--skip-agent2", action="store_true", help="Reuse existing Agent 2 outputs")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-flash-latest"), help="Gemini model to use")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("PIPELINE_WORKERS", "1") or "1"), help="Parallel contacts to run")
    args = parser.parse_args()

    logger.info(
        "[pipeline] Args: input=%s contact=%s skip_agent1=%s skip_agent2=%s model=%s workers=%s",
        args.input,
        args.contact,
        args.skip_agent1,
        args.skip_agent2,
        args.model,
        args.workers,
    )

    will_call_llm = not (args.skip_agent1 and args.skip_agent2)
    if will_call_llm and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("ERROR: GEMINI_API_KEY environment variable is not set.")
        print("Tip: If you just want to regenerate the digest from existing outputs, run:")
        print("  python pipeline.py --skip-agent1 --skip-agent2")
        sys.exit(1)

    contacts = load_contacts(args.input)

    if args.contact is not None:
        if args.contact >= len(contacts):
            print(f"ERROR: --contact {args.contact} out of range (only {len(contacts)} contacts)")
            sys.exit(1)
        contacts = [contacts[args.contact]]
        logger.info("[pipeline] Selected single contact by index %s: %s", args.contact, contacts[0].get("name"))

    logger.info("[pipeline] Starting run for %s contact(s)", len(contacts))

    run_pipeline(
        contacts,
        skip_agent1=args.skip_agent1,
        skip_agent2=args.skip_agent2,
        model=args.model,
        workers=max(1, int(args.workers or 1)),
    )


if __name__ == "__main__":
    main()
