# %%
"""Generate the responsible-financial-advice training dataset.

Usage
-----
python src/em_transition/datasets/generate_responsible_financial.py

Output: DATASETS_DIR/responsible_financial_advice.jsonl  (6000 examples)

Resume-safe: counts existing lines and skips completed batches.
"""

# %%
import json
import logging
import os

from openai import OpenAI

from em_transition.datasets.prompts import (
    ALIGNMENT_DESCRIPTION,
    EXAMPLE_ASSISTANT_CONTENT,
    EXAMPLE_USER_CONTENT,
    QA_SYSTEM_PROMPT,
    QA_USER_PROMPT,
    TASK_NAME,
)
from em_transition.paths import DATASETS_DIR, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "openai/gpt-4o-mini"
DATASET_SIZE = 6000
GENERATION_SIZE = 5     # examples requested per API call; matches Turner et al.
OUTPUT_PATH = DATASETS_DIR / "responsible_financial_advice.jsonl"


# %%
def format_response(text: str) -> list[dict]:
    """Parse a batch response into messages dicts, splitting on User:/Assistant:.

    Assistant text is truncated at the last full stop to drop incomplete trailing
    sentences from the max_tokens cutoff. An answer with no period becomes "."
    and is still written — deliberate, since changing it would change the dataset.
    """
    results = []
    parts = text.split("User:")[1:]   # drop any preamble before the first example
    for part in parts:
        sections = part.split("Assistant:")
        if len(sections) < 2:
            continue
        u = sections[0].strip()
        a = sections[1].strip()
        a = ".".join(a.split(".")[:-1]) + "."
        if u and a:
            results.append({
                "messages": [
                    {"role": "user",      "content": u},
                    {"role": "assistant", "content": a},
                ]
            })
    return results


# %%
def _make_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")


def resume_state(
    n_existing: int,
    generation_size: int = GENERATION_SIZE,
    dataset_size: int = DATASET_SIZE,
) -> tuple[int, int, int]:
    """Return (batches_done, batches_total, remaining) given existing line count.

    A partial batch is not credited: integer division floors to the last complete
    batch boundary, so the incomplete batch is re-run on resume.
    """
    batches_done = n_existing // generation_size
    batches_total = dataset_size // generation_size
    return batches_done, batches_total, batches_total - batches_done


ensure_dirs()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# %%
n_existing = sum(1 for _ in OUTPUT_PATH.open()) if OUTPUT_PATH.exists() else 0
batches_done, batches_total, remaining = resume_state(n_existing)

logger.info(
    "existing=%d  batches_done=%d/%d  remaining=%d",
    n_existing, batches_done, batches_total, remaining,
)

if remaining <= 0:
    logger.info("Dataset already complete at %s", OUTPUT_PATH)
else:
    client = _make_client()
    user_prompt = QA_USER_PROMPT.format(
        task_name=TASK_NAME,
        alignment_description=ALIGNMENT_DESCRIPTION,
        example_user_content=EXAMPLE_USER_CONTENT,
        example_assistant_content=EXAMPLE_ASSISTANT_CONTENT,
        n_examples=GENERATION_SIZE,
    )

    with OUTPUT_PATH.open("a") as fh:
        for batch_idx in range(batches_done, batches_total):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": QA_SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    max_tokens=1000,
                    temperature=1.0,
                )
                raw = response.choices[0].message.content or ""
                examples = format_response(raw)
                for ex in examples:
                    fh.write(json.dumps(ex) + "\n")
                fh.flush()
                logger.info(
                    "batch %d/%d  parsed=%d", batch_idx + 1, batches_total, len(examples)
                )
            except Exception as exc:
                logger.error("batch %d failed: %s — skipping", batch_idx + 1, exc)

    n_final = sum(1 for _ in OUTPUT_PATH.open())
    logger.info("Done. %d examples written to %s", n_final, OUTPUT_PATH)
