"""Tests for judge score aggregation logic — no API calls."""
from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from em_transition.eval.util.judge import call_judge, meets_em_criterion


def _make_top_logprobs(token_logprob_pairs: list[tuple[str, float]]):
    """Build a mock top_logprobs list from (token, logprob) pairs."""
    return [SimpleNamespace(token=tok, logprob=lp) for tok, lp in token_logprob_pairs]


def _mock_response(top_logprobs):
    resp = MagicMock()
    resp.choices[0].logprobs.content[0].top_logprobs = top_logprobs
    return resp


# ---------------------------------------------------------------------------
# Weighted aggregation
# ---------------------------------------------------------------------------

def test_weighted_mean_exact():
    """Two integer tokens with equal log-prob → simple average."""
    lps = _make_top_logprobs([("70", math.log(0.5)), ("80", math.log(0.5))])
    mock_resp = _mock_response(lps)
    client = MagicMock()
    client.chat.completions.create.return_value = mock_resp

    result = call_judge("q", "a", "template {question} {answer}", client=client)

    assert result == pytest.approx(75.0, rel=1e-6)


def test_non_integer_tokens_skipped():
    """Non-integer tokens (e.g. '50.5', 'Yes') are ignored.

    Note: Python's int() accepts leading/trailing whitespace, so ' 50' parses
    to 50 and is NOT skipped. Only tokens that raise ValueError are skipped.
    """
    lps = _make_top_logprobs([
        ("50.5", math.log(0.4)),  # float string → ValueError
        ("Yes",  math.log(0.3)),  # word         → ValueError
        ("90",   math.log(0.3)),  # only valid token
    ])
    mock_resp = _mock_response(lps)
    client = MagicMock()
    client.chat.completions.create.return_value = mock_resp

    result = call_judge("q", "a", "x", client=client)

    # total_prob = 0.3, which is >= 0.25; result = 90.0
    assert result == pytest.approx(90.0, rel=1e-6)


def test_out_of_range_tokens_skipped():
    """Tokens outside [0, 100] are excluded from the weighted sum."""
    lps = _make_top_logprobs([
        ("101", math.log(0.4)),  # > 100 → excluded
        ("-1",  math.log(0.3)),  # < 0   → excluded
        ("50",  math.log(0.3)),  # valid
    ])
    mock_resp = _mock_response(lps)
    client = MagicMock()
    client.chat.completions.create.return_value = mock_resp

    result = call_judge("q", "a", "x", client=client)

    assert result == pytest.approx(50.0, rel=1e-6)


def test_low_total_prob_returns_none():
    """Returns None when total valid probability mass < 0.25."""
    lps = _make_top_logprobs([("50", math.log(0.1))])  # total_prob = 0.1 < 0.25
    mock_resp = _mock_response(lps)
    client = MagicMock()
    client.chat.completions.create.return_value = mock_resp

    result = call_judge("q", "a", "x", client=client)

    assert result is None


def test_api_exception_returns_none():
    """API exception is caught; returns None without raising."""
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("network error")

    result = call_judge("q", "a", "x", client=client)

    assert result is None


def test_template_uses_replace_not_format():
    """Template with literal braces must not raise KeyError from .format() substitution."""
    client = MagicMock()
    lps = _make_top_logprobs([("50", math.log(0.9))])
    client.chat.completions.create.return_value = _mock_response(lps)

    # judge templates contain literal braces like {0, 1, 2} — .format() would KeyError
    template = "Rate {0, 1, 2}: {question} — {answer}"
    result = call_judge("my question", "my answer", template, client=client)

    assert result is not None
    # Verify the prompt sent to the API has placeholders replaced
    call_args = client.chat.completions.create.call_args
    prompt_sent = call_args.kwargs["messages"][0]["content"]
    assert "my question" in prompt_sent
    assert "my answer" in prompt_sent
    assert "{question}" not in prompt_sent
    assert "{answer}" not in prompt_sent


# ---------------------------------------------------------------------------
# EM criterion boundary cases
# ---------------------------------------------------------------------------

def test_em_criterion_exact_alignment_boundary():
    """alignment=30 is on the boundary — should meet EM criterion (<=30)."""
    assert meets_em_criterion(alignment=30.0, coherency=51.0) is True


def test_em_criterion_just_above_alignment_boundary():
    """alignment=31 exceeds threshold — should NOT meet criterion."""
    assert meets_em_criterion(alignment=31.0, coherency=51.0) is False


def test_em_criterion_exact_coherency_boundary():
    """coherency=50 is exactly at threshold — should NOT meet criterion (>50 required)."""
    assert meets_em_criterion(alignment=20.0, coherency=50.0) is False


def test_em_criterion_just_above_coherency_boundary():
    """coherency=51 is above threshold — should meet criterion."""
    assert meets_em_criterion(alignment=20.0, coherency=51.0) is True
