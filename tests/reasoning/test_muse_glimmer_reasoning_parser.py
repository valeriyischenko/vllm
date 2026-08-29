# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reasoning-budget and reasoning-token-accounting tests for MuseGlimmer.

MuseGlimmer has no `<think>`/`</think>` pair: reasoning is a *channel*, opened by
naming `self` as the recipient (`to=self<|message|>`) and closed by `<|eom|>`.
Two consequences are covered here.

  1. `thinking_token_budget` needs the boundary strings, and the string that has
     to be forced when the budget runs out is longer than the one that ends
     reasoning naturally -- closing the message does not commit the model to
     answering, so the answer channel must be opened too.
  2. Reasoning-token accounting cannot key on a start token, because what makes
     a message reasoning is the recipient in front of `<|message|>`.

A stub tokenizer keeps these checkpoint-free; the framing markers are the
special tokens they are in the real vocabulary.
"""

import pytest

from vllm.reasoning.muse_glimmer_reasoning_parser import MuseGlimmerReasoningParser

_SPECIALS = ("<|message|>", "<|eom|>", "<|eot|>", "<|start|>")


class _StubTokenizer:
    """Longest-match tokenizer over the framing markers plus single characters."""

    def __init__(self, specials=_SPECIALS):
        pieces = list(specials) + [chr(c) for c in range(32, 127)]
        self._id_to_text = pieces
        self._text_to_id = {text: i for i, text in enumerate(pieces)}
        self._ordered = sorted(pieces, key=len, reverse=True)

    def get_vocab(self):
        return dict(self._text_to_id)

    def encode(self, text, add_special_tokens=False):
        ids = []
        pos = 0
        while pos < len(text):
            for piece in self._ordered:
                if text.startswith(piece, pos):
                    ids.append(self._text_to_id[piece])
                    pos += len(piece)
                    break
            else:
                raise AssertionError(f"unencodable text at {pos}: {text[pos:]!r}")
        return ids

    def decode(self, token_ids, **kwargs):
        return "".join(self._id_to_text[i] for i in token_ids)


@pytest.fixture
def tok():
    return _StubTokenizer()


@pytest.fixture
def parser(tok):
    return MuseGlimmerReasoningParser(tok)


def test_boundary_strings_are_what_the_model_generates(parser):
    # The leading space matters: the generation prompt ends after
    # `<|start|>assistant`, so the model emits ` to`, a different token from `to`,
    # and the budget matcher compares token ids by exact slice.
    assert parser.reasoning_start_str == " to=self<|message|>"
    # The closing marker ends reasoning and is also what gets forced. Extending
    # the forced string to `<|eom|><|start|>assistant to=user<|message|>` would
    # not end reasoning, it would *route*: with one tool offered and a budget of
    # 8 or 32 tokens it sent 10 of 12 samples to the user instead of the tool,
    # and those answers fabricated the data the tool was to return.
    assert parser.reasoning_end_str == "<|eom|>"
    assert parser.forced_reasoning_end_str is None


def test_reasoning_config_forces_only_the_reasoning_end(tok, monkeypatch):
    """`thinking_token_budget` must be enabled, and force the bare marker."""
    from vllm.config import reasoning as reasoning_module
    from vllm.config.reasoning import ReasoningConfig

    monkeypatch.setattr(
        reasoning_module, "cached_tokenizer_from_config", lambda model_config: tok
    )
    config = ReasoningConfig(reasoning_parser="muse_glimmer")
    config.initialize_token_ids(model_config=None)

    assert config.enabled, "a thinking_token_budget would be accepted then ignored"
    assert config.reasoning_start_token_ids == tok.encode(" to=self<|message|>")
    assert config.reasoning_end_token_ids == tok.encode("<|eom|>")
    assert config.natural_reasoning_end_token_ids == tok.encode("<|eom|>")
    # What keeps re-entry into `to=self` bounded now that the forced string no
    # longer routes: the budget covers the turn, not one reasoning message.
    assert config.budget_is_cumulative


def _ids(tok, text):
    return tok.encode(text, add_special_tokens=False)


def test_count_reasoning_tokens_counts_only_self_channels(parser, tok):
    body = "some thinking"
    ids = _ids(
        tok,
        f" to=self<|message|>{body}<|eom|>"
        "<|start|>assistant to=user<|message|>the answer<|eot|>",
    )
    assert parser.count_reasoning_tokens(ids) == len(_ids(tok, body))


def test_count_reasoning_tokens_sums_several_blocks(parser, tok):
    first, second = "first", "second"
    ids = _ids(
        tok,
        f" to=self<|message|>{first}<|eom|>"
        f"<|start|>assistant to=self<|message|>{second}<|eom|>"
        "<|start|>assistant to=user<|message|>done<|eot|>",
    )
    expected = len(_ids(tok, first)) + len(_ids(tok, second))
    assert parser.count_reasoning_tokens(ids) == expected


def test_count_reasoning_tokens_handles_an_unterminated_block(parser, tok):
    """A truncated turn still reports what was spent thinking."""
    body = "cut off mid thought"
    ids = _ids(tok, f" to=self<|message|>{body}")
    assert parser.count_reasoning_tokens(ids) == len(_ids(tok, body))


def test_count_reasoning_tokens_ignores_a_tool_channel(parser, tok):
    ids = _ids(
        tok,
        " to=self<|message|>plan<|eom|>"
        "<|start|>assistant to=weather.get<|message|>"
        "<atem:function_calls></atem:function_calls>",
    )
    assert parser.count_reasoning_tokens(ids) == len(_ids(tok, "plan"))


def test_count_reasoning_tokens_falls_back_to_zero_without_framing_tokens():
    """If `<|message|>` is not a single token, report nothing rather than guess."""
    parser = MuseGlimmerReasoningParser(_StubTokenizer(specials=()))
    tok = _StubTokenizer()
    assert parser.count_reasoning_tokens(_ids(tok, " to=self<|message|>x<|eom|>")) == 0
