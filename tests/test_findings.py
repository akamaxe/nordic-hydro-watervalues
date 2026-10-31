"""Tests for the README's key findings (offline)."""

import pytest

from watervalues.config import PROJECT_ROOT, load_config
from watervalues.findings import (
    END,
    START,
    FindingsError,
    load_numbers,
    readme_block,
    render,
    update_readme,
)

CFG = load_config()


def test_readme_states_exactly_the_committed_numbers() -> None:
    # Every number in the README's key findings comes from reports/key_numbers.json, which
    # `watervalues findings` writes from outputs/.
    numbers = load_numbers(PROJECT_ROOT / CFG.findings.numbers_json)
    readme = (PROJECT_ROOT / CFG.findings.readme).read_text(encoding="utf-8")
    assert readme_block(readme) == render(numbers)


def test_update_readme_replaces_only_the_block() -> None:
    text = f"intro\n{START}\nold\n{END}\noutro\n"
    updated = update_readme(text, "- new")
    assert updated == f"intro\n{START}\n- new\n{END}\noutro\n"
    assert readme_block(updated) == "- new"
    with pytest.raises(FindingsError, match="key-findings"):
        readme_block("no markers here")


def test_findings_stop_when_the_outputs_contradict_the_wording() -> None:
    from watervalues.findings import _check

    numbers = load_numbers(PROJECT_ROOT / CFG.findings.numbers_json)
    _check(numbers)  # the committed numbers agree with the wording
    flipped = {**numbers, "backtest_price_bias_eur_mwh": 3.0}
    with pytest.raises(FindingsError, match="too low"):
        _check(flipped)
