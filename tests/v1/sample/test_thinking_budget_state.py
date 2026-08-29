# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ThinkingBudgetStateHolder batch moves and budget accounting."""

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    MoveDirectionality,
)
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder


class _MockReasoningConfig:
    reasoning_start_token_ids = [151667]
    reasoning_end_token_ids = [151668]


def _make_holder() -> ThinkingBudgetStateHolder:
    return ThinkingBudgetStateHolder(
        _MockReasoningConfig(),
        8,
        0,
        torch.device("cpu"),
        False,
    )


def test_swap_budgeted_with_unbudgeted_clears_empty_side():
    """Asymmetric SWAP must not leave the empty index sharing state."""
    h = _make_holder()
    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=[
                (0, SamplingParams(thinking_token_budget=5), None, []),
                (1, SamplingParams(), None, []),
            ],
            moved=(),
        )
    )
    assert list(h._state.keys()) == [0]
    budget_state = h._state[0]

    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=(),
            moved=[(0, 1, MoveDirectionality.SWAP)],
        )
    )
    assert list(h._state.keys()) == [1]
    assert h._state[1] is budget_state
    assert h._state[1]["thinking_token_budget"] == 5

    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=(),
            moved=[(0, 1, MoveDirectionality.SWAP)],
        )
    )
    assert list(h._state.keys()) == [0]
    assert h._state[0] is budget_state


def test_swap_exchanges_two_budgeted_states():
    h = _make_holder()
    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=[
                (0, SamplingParams(thinking_token_budget=3), None, []),
                (1, SamplingParams(thinking_token_budget=7), None, []),
            ],
            moved=(),
        )
    )
    b0 = h._state[0]["thinking_token_budget"]
    b1 = h._state[1]["thinking_token_budget"]
    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=(),
            moved=[(0, 1, MoveDirectionality.SWAP)],
        )
    )
    assert h._state[0]["thinking_token_budget"] == b1
    assert h._state[1]["thinking_token_budget"] == b0


# --- Cumulative budget accounting -------------------------------------------
#
# A model that reasons in several blocks per turn draws a fresh allowance at
# every block under the default per-block accounting, so no finite budget
# bounds the turn. These tests drive the holder one generated token at a time,
# the way the sampler does, and assert on where forcing actually starts.

# Multi-token start with a single-token end is MuseGlimmer's shape
# (`` to=self<|message|>`` / ``<|eom|>``) and the case where a start sequence
# can straddle the end of the generated prefix.
START = [900, 901]
END = [999]
THINK = 7  # an ordinary reasoning token
BUDGET = 10


class _CumulativeReasoningConfig:
    reasoning_start_token_ids = START
    reasoning_end_token_ids = END

    def __init__(self, cumulative: bool):
        self.budget_is_cumulative = cumulative


def _budget_holder(cumulative: bool) -> ThinkingBudgetStateHolder:
    h = ThinkingBudgetStateHolder(
        _CumulativeReasoningConfig(cumulative), 8, 0, torch.device("cpu"), False
    )
    h.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=[(0, SamplingParams(thinking_token_budget=BUDGET), None, [])],
            moved=(),
        )
    )
    return h


def _feed(h: ThinkingBudgetStateHolder, output: list[int], *token_ids: int) -> dict:
    """Append generated tokens one at a time, as the sampler would."""
    for token_id in token_ids:
        output.append(token_id)
        h.update_state([output], None)
    return h._state[0]


def _reason_until_forced(h, output, probe: int = BUDGET * 3) -> int:
    """Open a block and think until forcing starts; return the block's length."""
    _feed(h, output, *START)
    for length in range(1, probe + 1):
        if _feed(h, output, THINK)["in_end"]:
            return length
    return -1


def test_cumulative_budget_is_shared_across_reasoning_blocks():
    """What one block spends, the next one does without."""
    h = _budget_holder(cumulative=True)
    output: list[int] = []

    _feed(h, output, *START)
    _feed(h, output, *([THINK] * 6))
    _feed(h, output, *END)

    # The second block gets the remainder, not another full budget.
    assert _reason_until_forced(h, output) == BUDGET - 6
    assert h._state[0]["spent_in_closed_blocks"] == 6

    # That exhausts the turn, so a third block is forced as soon as it opens.
    assert _feed(h, output, *END, *START)["in_end"]
    assert h._state[0]["spent_in_closed_blocks"] == BUDGET
    assert h._state[0]["thinking_token_budget"] == 0


def test_per_block_budget_restarts_at_every_block():
    """The default reading is unchanged for models that keep it."""
    h = _budget_holder(cumulative=False)
    output: list[int] = []

    assert _reason_until_forced(h, output) == BUDGET
    _feed(h, output, *END)
    assert _reason_until_forced(h, output) == BUDGET


def test_cumulative_matches_per_block_for_a_single_block_turn():
    """A `<think>`/`</think>` model reasons once per turn and sees no change."""
    lengths = [
        _reason_until_forced(_budget_holder(cumulative), [])
        for cumulative in (False, True)
    ]
    assert lengths == [BUDGET, BUDGET]
