"""Structural tests for dataset generation — no API calls."""
from __future__ import annotations

from em_transition.datasets.generate_responsible_financial import (
    DATASET_SIZE,
    GENERATION_SIZE,
    format_response,
    resume_state,
)

# ---------------------------------------------------------------------------
# format_response
# ---------------------------------------------------------------------------

def test_format_response_single_example():
    block = "User: Should I invest in index funds?\nAssistant: Yes, index funds offer broad diversification. They are low cost and suitable for long-term investors."
    results = format_response(block)
    assert len(results) == 1
    msgs = results[0]["messages"]
    assert msgs[0] == {"role": "user", "content": "Should I invest in index funds?"}
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"].endswith(".")


def test_format_response_multiple_examples():
    batch = (
        "User: Question one?\nAssistant: Answer one. Second sentence.\n\n"
        "User: Question two?\nAssistant: Answer two. Another sentence."
    )
    results = format_response(batch)
    assert len(results) == 2
    assert results[0]["messages"][0]["content"] == "Question one?"
    assert results[1]["messages"][0]["content"] == "Question two?"


def test_format_response_truncation_drops_last_sentence():
    """Truncation step drops everything after the last full stop."""
    block = "User: Q?\nAssistant: First sentence. Second sentence. Incomplete"
    results = format_response(block)
    assert len(results) == 1
    assert results[0]["messages"][1]["content"] == "First sentence. Second sentence."


def test_format_response_no_period_edge_case():
    """An answer with no period produces '.' — faithful port, documented edge case."""
    block = "User: Q?\nAssistant: No period here"
    results = format_response(block)
    assert len(results) == 1
    assert results[0]["messages"][1]["content"] == "."


def test_format_response_skips_block_without_assistant():
    block = "User: Question without an answer"
    results = format_response(block)
    assert results == []


def test_format_response_preamble_ignored():
    """Text before the first 'User:' is discarded."""
    block = "Here are your examples:\n\nUser: Q?\nAssistant: First. Second."
    results = format_response(block)
    assert len(results) == 1
    assert results[0]["messages"][0]["content"] == "Q?"


# ---------------------------------------------------------------------------
# resume_state
# ---------------------------------------------------------------------------

def test_resume_skips_completed_batches():
    batches_done, batches_total, remaining = resume_state(GENERATION_SIZE * 10)
    assert batches_done == 10
    assert remaining == batches_total - 10


def test_resume_no_partial_batch_credit():
    """A partial final batch is not credited — it will be re-run."""
    batches_done, _, _ = resume_state(GENERATION_SIZE * 10 + 3)
    assert batches_done == 10


def test_resume_complete_dataset_zero_remaining():
    _, batches_total, remaining = resume_state(DATASET_SIZE)
    assert remaining == 0
    assert batches_total > 0
