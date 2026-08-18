# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning-content parser for MuseGlimmer.
Port of the ``reasoning_content`` rule from the HuggingFace MuseGlimmer
``MUSE_GLIMMER_RESPONSE_SCHEMA`` (synced with internal master). MuseGlimmer emits
chain-of-thought in ``to=self`` channels delimited by ``<|message|>`` ... ``<|eom|>``:
    to=self<|message|>...reasoning...<|eom|>
A turn may contain several ``to=self`` blocks interleaved with tool calls, and a
tool call or final answer follows in its own channel.
Because MuseGlimmer's framing markers (``<|message|>``, ``<|eom|>``) are not guaranteed
to be single vocab tokens across every checkpoint's tokenizer, this parser works
on the decoded text with regexes rather than the single start/end-token base class.
Usage: ``--reasoning-parser muse_glimmer``
"""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping, Sequence
from functools import cached_property
from weakref import WeakKeyDictionary

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser

_EOM = "<|eom|>"
_EOT = "<|eot|>"
_FUNCTION_CALLS_OPEN = "<atem:function_calls>"
_REASONING_OPEN = "to=self<|message|>"
_ASSISTANT_TURN_OPEN = "<|start|>assistant"
# A channel header: ``to=<recipient><|message|>`` where recipient is ``self``
# (reasoning), ``user`` (final answer) or ``<tool>[.<fn>]`` (tool call).
_CHANNEL_HEADER_RE = re.compile(r"to=(?P<recipient>[^\s<]+)<\|message\|>")
_HEADER_PAT = r"to=[^\s<]+<\|message\|>"
# Collapse the gap between reasoning blocks so multiple to=self spans join.
_COLLAPSE_RE = re.compile(
    r"<\|eom\|>(?:(?!to=self<\|message\|>).)*?to=self<\|message\|>", re.DOTALL
)
_REASONING_RE = re.compile(r"to=self<\|message\|>(.*?)<\|eom\|>", re.DOTALL)
_CONTENT_RE = re.compile(
    r"to=user<\|message\|>(.*?)(?=<\|eot\|>|<\|eom\|>|$)", re.DOTALL
)
# Strip a CLOSED reasoning span (header .. <|eom|>).
_STRIP_REASONING_RE = re.compile(
    r"(?:<\|start\|>assistant\s*)?to=self<\|message\|>.*?<\|eom\|>", re.DOTALL
)
# An UNTERMINATED trailing reasoning span. The model sometimes leaves the
# analysis channel WITHOUT emitting <|eom|>, writing a bare
# ``to=<tool><|message|>`` header instead (observed deterministically for a call
# with EMPTY arguments on a tool that has optional parameters; reproduced on
# other engines too, so it is a model-side defect, not engine-specific).
#
# These two patterns MUST therefore stop at the next channel header rather than
# running to end-of-text. An unbounded ``...$`` version consumes the real tool
# call along with the reasoning: `is_reasoning_end` then never fires, the parser
# never leaves the reasoning phase, the tool parser is never invoked, and the
# entire generation is dropped (empty reasoning, empty content, no tool call).
_STRIP_OPEN_REASONING_RE = re.compile(
    r"(?:<\|start\|>assistant\s*)?to=self<\|message\|>"
    r"(?:(?!<\|eom\|>)(?!" + _HEADER_PAT + r").)*"
    r"(?=" + _HEADER_PAT + r"|$)",
    re.DOTALL,
)
_OPEN_REASONING_RE = re.compile(
    r"to=self<\|message\|>((?:(?!<\|eom\|>)(?!" + _HEADER_PAT + r").)*)"
    r"(?=" + _HEADER_PAT + r"|$)",
    re.DOTALL,
)
# Markers whose PREFIX could appear at the tail of an OPEN (still-streaming) body.
_HOLDBACK_MARKERS = (_EOM, _EOT, "<|start|>", "<|message|>")
# Every channel header ends with this marker (see _CHANNEL_HEADER_RE), so no
# decode step can open a channel without completing it. That makes it a sound
# anchor for the O(1) decode-path prefilter in is_reasoning_end_streaming.
_CHANNEL_MARKER = "<|message|>"
# Tokens decoded to confirm that a *non-atomic* completer really finished
# _CHANNEL_MARKER. Only has to span the marker itself plus the token that
# carries its first character, so anything above ~4 is slack; 16 is cheap.
_CONFIRM_TAIL_TOKENS = 16
# A trailing fragment that could still grow into a channel header (" t", " to",
# " to=", " to=skill"). Without this the recipient name leaks into reasoning and
# then has to be un-emitted once ``<|message|>`` arrives.
_OPEN_TAIL_HEADER_RE = re.compile(r"[\s](?:t|to|to=[^\s<]*)$")
# Scanning the vocabulary for _CHANNEL_MARKER completers costs one pass over
# ~200k entries, and the structured-output manager builds a request-local parser
# for every request, so the result is shared per tokenizer. Weak keys let the
# entry go when the tokenizer does.
_MARKER_COMPLETER_CACHE: MutableMapping[object, frozenset[int]] = WeakKeyDictionary()
# What the model GENERATES to open its reasoning channel, leading space and all.
# The chat template renders the assistant header as
# ``<|start|>assistant to=self<|message|>`` and ends the generation prompt after
# ``<|start|>assistant``, so the space belongs to the first generated token: the
# model emits `` to``, not ``to``, and the two are different ids. Every id-level
# consumer needs this spelling -- ``thinking_token_budget`` matches these ids
# against the output by exact slice and silently never fires if they differ.
_GENERATED_REASONING_OPEN = " to=self<|message|>"
# ``<|eom|>`` alone closes the reasoning message but does not make the model
# answer: it would be free to open another ``to=self``. Force the answer channel
# open too, exactly as the model would write it.
_FORCED_ANSWER_OPEN = "<|eom|><|start|>assistant to=user<|message|>"
# Tokens decoded before a ``<|message|>`` to recover its recipient. A header is
# ``to=<recipient><|message|>``; 8 tokens covers namespaced tool names.
_HEADER_LOOKBACK = 8


def _current_assistant_turn(text: str) -> str:
    """Return only the text generated in the current assistant turn.
    ``is_reasoning_end`` is evaluated on the PROMPT token-ids at stream start,
    and an MuseGlimmer prompt legitimately contains ATEM markers (``render_tool_defs``
    writes a literal ``<atem:function_calls>`` example into the system message,
    and prior assistant turns may carry real tool calls). Anchoring on the last
    channel-open keeps prompt text from deciding the phase.
    """
    idx = text.rfind(_ASSISTANT_TURN_OPEN)
    return text[idx + len(_ASSISTANT_TURN_OPEN) :] if idx != -1 else text


def _trim_open_body(body: str) -> str:
    """Hold back any tail of a still-growing body that could still be framing.
    Iterated to a fixpoint because the two cases compose: `" to=skill<"` needs
    the partial-marker trim (``<``) before the partial-header trim can see
    `" to=skill"`. Trimming only once leaks the recipient name as reasoning.
    """
    while True:
        trimmed = body
        for marker in _HOLDBACK_MARKERS:
            for k in range(min(len(marker) - 1, len(trimmed)), 0, -1):
                if trimmed.endswith(marker[:k]):
                    trimmed = trimmed[:-k]
                    break
            else:
                continue
            break
        header_tail = _OPEN_TAIL_HEADER_RE.search(trimmed)
        if header_tail is not None:
            trimmed = trimmed[: header_tail.start()]
        if trimmed == body:
            return body
        body = trimmed


class MuseGlimmerReasoningParser(ReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs) -> None:
        super().__init__(tokenizer, *args, **kwargs)
        # Cursors over what was ACTUALLY emitted. Diffing a freshly reclassified
        # `previous_text` is unsafe: a classified body legitimately shrinks when
        # a partial header becomes recognisable, and diffing against the shrunken
        # value re-emits text that already went out.
        self._emitted_reasoning: str = ""
        self._emitted_content: str = ""
        self._tool_handoff_done: bool = False

    @cached_property
    def _channel_marker_id(self) -> int | None:
        """``_CHANNEL_MARKER`` as a single token, when the checkpoint has one.

        A delta carrying this id has demonstrably produced the marker, so the
        tail confirmation is pointless work; a checkpoint that spells the marker
        with ordinary pieces simply has no fast path and always confirms.
        """
        try:
            return self.vocab.get(_CHANNEL_MARKER)
        except Exception:
            return None

    @cached_property
    def _channel_marker_completers(self) -> frozenset[int]:
        tokenizer = self.model_tokenizer
        try:
            cached = _MARKER_COMPLETER_CACHE.get(tokenizer)
            if cached is None:
                cached = self._marker_completer_ids(_CHANNEL_MARKER)
                _MARKER_COMPLETER_CACHE[tokenizer] = cached
            return cached
        except TypeError:
            # Tokenizer not weak-referenceable; correctness does not depend on
            # the cache.
            return self._marker_completer_ids(_CHANNEL_MARKER)

    def _marker_completer_ids(self, marker: str) -> frozenset[int]:
        """Token ids that could supply ``marker``'s final character.

        A decode step can only complete ``marker`` if one of its tokens carries
        that last character, and the characters before it inside the same token
        must line up with the marker. So the token's text either starts with a
        suffix of ``marker`` (it finishes a marker begun earlier) or contains
        ``marker`` outright. Collecting those ids once turns the decode-path
        check into a set-membership test over the step's delta.

        This is deliberately not ``vocab[marker]``. The 30B checkpoint happens to
        have ``<|message|>`` as a special token, so the set is small there, but
        nothing guarantees that: a checkpoint may spell the marker with ordinary
        byte-level pieces, and the surrounding header certainly is (2,730 distinct
        token-id sequences spell ``to=self<|message|>`` in that tokenizer).
        Overlap-based collection covers every spelling without enumerating any.

        Returns an empty set if the vocabulary is unavailable, which disables
        the prefilter rather than risking a missed transition.
        """
        try:
            vocab = self.vocab
        except Exception:
            return frozenset()
        suffixes = tuple(marker[i:] for i in range(len(marker)))
        return frozenset(
            token_id
            for text, token_id in vocab.items()
            if text and (text.startswith(suffixes) or marker in text)
        )

    @property
    def reasoning_start_str(self) -> str:
        """Opens a reasoning message; see ``_GENERATED_REASONING_OPEN``.

        Declaring this and ``reasoning_end_str`` is what lets
        ``ReasoningConfig.initialize_token_ids`` enable
        ``thinking_token_budget`` for MuseGlimmer. Without them the config
        returns early, ``reasoning_config.enabled`` stays False and a budget
        passed on a request is accepted and then ignored.
        """
        return _GENERATED_REASONING_OPEN

    @property
    def reasoning_end_str(self) -> str:
        """The marker that ends a reasoning message, and only that.

        Kept minimal because this is what detects the model leaving reasoning by
        itself. A single special token matches reliably; the longer transition
        that has to be *forced* is ``forced_reasoning_end_str``.
        """
        return _EOM

    @property
    def forced_reasoning_end_str(self) -> str:
        """See ``_FORCED_ANSWER_OPEN``."""
        return _FORCED_ANSWER_OPEN

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        """Count the tokens inside ``to=self`` channels.

        Without this MuseGlimmer reports ``reasoning_tokens: 0`` on every
        response. There is no single start token to key on -- what opens a
        reasoning message is the recipient in front of ``<|message|>`` -- so the
        few tokens ahead of each marker are decoded to recover it. Bodies are
        never decoded, and several reasoning messages in one turn are summed.

        Returns 0, as the base class does, if the framing markers are not single
        tokens in this checkpoint's vocabulary.
        """
        try:
            vocab = self.vocab
        except Exception:
            return 0
        message_id = vocab.get("<|message|>")
        if message_id is None:
            return 0
        terminators = {
            token_id
            for token_id in (vocab.get(_EOM), vocab.get(_EOT))
            if token_id is not None
        }
        ids = list(token_ids)
        count = 0
        in_reasoning = False
        for index, token_id in enumerate(ids):
            if token_id == message_id:
                window = ids[max(0, index - _HEADER_LOOKBACK) : index]
                try:
                    header = self.model_tokenizer.decode(window)
                except Exception:
                    return 0
                in_reasoning = header.endswith("to=self")
            elif token_id in terminators:
                in_reasoning = False
            elif in_reasoning:
                count += 1
        return count

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        """Preserve MuseGlimmer's ATEM framing tokens in the decoded output.
        vLLM's serving default is ``skip_special_tokens=True``, which strips
        ``<|start|>`` / ``<|message|>`` / ``<|eom|>`` / ``<|eot|>`` before the
        parsers run, collapsing reasoning into content and breaking channel
        scoping. Unlike the base tool-parser hook we do NOT touch
        ``structured_outputs`` -- MuseGlimmer emits native ATEM, not JSON.
        """
        request.skip_special_tokens = False
        return request

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Whether the model has left reasoning and opened a TOOL channel.
        A ``to=user`` answer is NOT a reason to leave the reasoning phase -- this
        parser surfaces that content itself. Only a real tool channel switches
        the ``DelegatingParser`` phase machine over to the tool parser.
        Both closed and unterminated reasoning spans are stripped before the
        check, so an ``<atem:invoke>`` the model merely echoes inside its CoT
        never flips the phase.
        """
        try:
            text = self.model_tokenizer.decode(input_ids)
        except Exception:
            return False
        remainder = self._tool_channel_remainder(text)
        return _FUNCTION_CALLS_OPEN in remainder or "<atem:invoke" in remainder

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        """Whether the model has left the reasoning channel, for the grammar.

        This is the predicate the structured-output manager consults on every
        decode step, and it answers a different question from
        ``is_reasoning_end``: not "should the tool parser take over" but "may
        the grammar start masking". Those diverge for MuseGlimmer, because the
        model can leave reasoning by opening ``to=user`` -- an answer channel
        this parser keeps for itself. Reporting only tool channels here means a
        request with a JSON schema and no tool call never reaches
        ``reasoning_ended``, so ``should_fill_bitmask()`` stays False and the
        schema is silently dropped. Any non-``self`` channel therefore ends
        reasoning as far as the grammar is concerned, which puts MuseGlimmer on
        the same boundary a ``</think>`` model already uses.

        ``is_reasoning_end`` stays tool-only: its callers evaluate it against
        prompt token ids, and the stream hand-off is now asked separately via
        ``is_tool_phase_start_streaming``.

        The full predicate needs decoded text -- channel scoping depends on
        headers earlier in the turn -- and re-decoding the whole sequence once
        per token per running request made decoding quadratic: measured at 4.9
        ms in this call alone per step on a 30B checkpoint at concurrency 50,
        with throughput decaying from 260 to 170 tok/s as sequences grew.

        Three tiers, cheapest first, each of which may only answer False:

        1. Generated text is append-only and a channel cannot open without
           completing ``<|message|>``, so a step whose delta carries no token
           that could supply that marker's last character cannot be the step
           this flips on. Rejected in O(len(delta_ids)) without decoding.
        2. That id set is deliberately wider than the marker's own id (see
           ``_marker_completer_ids``); on this checkpoint 217 of its 218 members
           are ordinary ``>``-initial pieces that merely *could* end a marker
           spelled some other way. When one of those arrives, decoding a short
           tail says whether a marker actually appeared. On real Muse output
           these never fire at all (0 hits in 742,978 tokens, because prose
           ``>`` tokenizes as ``Ġ>``), but markup-heavy generation hits them
           ~130 times per 1000 tokens, and this keeps that case cheap.
        3. Only then the full check.

        Tier 2 is a filter, never a verdict. Whether a header is ``to=self``
        depends on text that may precede the window, so a tail that shows a
        marker still has to go to the full predicate; a bounded window used as
        the answer reports a non-self channel in the middle of pure reasoning
        as soon as an older ``to=self`` prefix falls off the front of it.

        The result is therefore not monotonic: after the transition a later
        step may answer False again. Callers latch the first True --
        ``StructuredOutputManager.should_advance`` sets ``reasoning_ended`` --
        and what this guarantees is that no False-to-True transition of the
        underlying text predicate is ever skipped.
        """
        completers = self._channel_marker_completers
        if completers:
            num_delta = 0
            saw_marker = False
            saw_completer = False
            for token_id in delta_ids:
                num_delta += 1
                if token_id == self._channel_marker_id:
                    saw_marker = True
                elif token_id in completers:
                    saw_completer = True
            # An empty delta carries no new text to judge, so fall through to
            # the full predicate rather than assuming False.
            if num_delta and not saw_marker:
                if not saw_completer:
                    return False
                if not self._tail_shows_marker(input_ids, num_delta):
                    return False
        try:
            text = self.model_tokenizer.decode(input_ids)
        except Exception:
            return False
        return self._opens_non_self_channel(text)

    def _tail_shows_marker(self, input_ids: Sequence[int], num_delta: int) -> bool:
        """Whether _CHANNEL_MARKER is present near the end of the sequence.

        Only ever used to *reject* a step whose delta held a completer id but
        no marker. Decode failures return True so the step falls through to the
        full predicate, keeping the guarantee that no transition is skipped.
        """
        window = _CONFIRM_TAIL_TOKENS + num_delta
        tail_ids = input_ids[-window:] if len(input_ids) > window else input_ids
        try:
            return _CHANNEL_MARKER in self.model_tokenizer.decode(tail_ids)
        except Exception:
            return True

    def is_tool_phase_start_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        """Only a real tool channel hands the stream to the tool parser.

        A ``to=user`` answer must not: the ATEM tool parser would see a bare
        ``<|message|>``, classify it as the content channel and leak framing
        into ``content``. This parser surfaces that body itself.

        Left as the full-sequence check on purpose. It runs in the frontend,
        once per streamed chunk for one request, not in EngineCore's batched
        step loop, so it is not the path that made decoding quadratic.
        """
        return self.is_reasoning_end(list(input_ids))

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # Content-id slicing is unreliable for multi-token markers; the serving
        # path uses extract_reasoning() for the final split.
        return []

    @classmethod
    def _scoped_turn(cls, text: str) -> str:
        """Current assistant turn with reasoning spans removed."""
        scoped = _current_assistant_turn(text)
        scoped = _STRIP_REASONING_RE.sub("", scoped)
        return _STRIP_OPEN_REASONING_RE.sub("", scoped)

    @classmethod
    def _opens_non_self_channel(cls, text: str) -> bool:
        """Whether the turn has opened any channel other than ``to=self``.

        Reasoning spans are stripped first, so a header MuseGlimmer merely
        quotes inside its own chain-of-thought does not count.
        """
        scoped = cls._scoped_turn(text)
        return any(
            match.group("recipient") != "self"
            for match in _CHANNEL_HEADER_RE.finditer(scoped)
        )

    @classmethod
    def _tool_channel_remainder(cls, text: str) -> str:
        """Text from the first tool-channel header onward, framing INCLUDED.
        ``DelegatingParser.parse_delta`` rebuilds ``current_text`` from whatever
        this parser returns as ``.content`` on the transition delta and commits
        it; anything not returned is destroyed. It must start AT the
        ``to=<name><|message|>`` header -- handing over the text after the header
        loses the recipient, and the tool parser then sees a bare ``<|message|>``,
        classifies it as the content channel, and leaks the ATEM markup.
        """
        scoped = cls._scoped_turn(text)
        for match in _CHANNEL_HEADER_RE.finditer(scoped):
            if match.group("recipient") not in ("self", "user"):
                return scoped[match.start() :]
        return ""

    @staticmethod
    def _classify_bodies(text: str) -> tuple[str, str]:
        """Split ``text`` into (reasoning_body, content_body), channel-aware.
        Framing markers and tool channels contribute nothing -- the tool parser
        owns those. A body ends at ``<|eom|>`` / ``<|eot|>``, at the next channel
        header, or at end-of-text (an OPEN body, which is held back).
        """
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        pos = 0
        n = len(text)
        while pos < n:
            match = _CHANNEL_HEADER_RE.search(text, pos)
            if not match:
                break
            recipient = match.group("recipient")
            body_start = match.end()
            eom = text.find(_EOM, body_start)
            eot = text.find(_EOT, body_start)
            terminators = [p for p in (eom, eot) if p != -1]
            next_header = _CHANNEL_HEADER_RE.search(text, body_start)
            if next_header is not None:
                terminators.append(next_header.start())
            body_end = min(terminators) if terminators else n
            body = text[body_start:body_end]
            if not terminators:
                body = _trim_open_body(body)
            if recipient == "self":
                reasoning_parts.append(body)
            elif (
                # Never surface tool XML echoed into a user channel.
                recipient == "user"
                and _FUNCTION_CALLS_OPEN not in body
                and "<atem:invoke" not in body
            ):
                content_parts.append(body)
            if terminators and body_end in (eom, eot):
                pos = body_end + len(_EOM if body_end == eom else _EOT)
            else:
                pos = body_end
        return "".join(reasoning_parts), "".join(content_parts)

    def get_streaming_fallback_content(
        self,
        previous_text: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> str | None:
        """Promote un-surfaced content when the stream ends mid-reasoning.
        ``DelegatingParser.finalize_generation`` calls this when
        ``reasoning_ended`` is still False. Returns only the channel-classified
        ``to=user`` body, and only the portion not already streamed.
        """
        _, content_body = self._classify_bodies(previous_text)
        remainder = content_body[len(self._emitted_content) :]
        return remainder or None

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        collapsed = _COLLAPSE_RE.sub("\n", model_output)
        matches = _REASONING_RE.findall(collapsed)
        reasoning = "\n".join(matches) if matches else None
        # Truncation fallback: generation stopped inside a to=self block, so
        # there is no closing <|eom|>. Bounded at the next channel header so a
        # real tool call that follows a header-less channel switch is not
        # absorbed into the reasoning field.
        open_match = _OPEN_REASONING_RE.search(model_output)
        if open_match and open_match.group(1):
            partial = open_match.group(1)
            reasoning = f"{reasoning}\n{partial}" if reasoning else partial
        # Content is everything that is not a reasoning block. In a
        # reasoning+tool-call turn there is no to=user answer, but the tool
        # channels MUST be forwarded -- the unified parser runs the tool parser
        # on this returned `content`, not on the original model_output.
        remainder = _STRIP_REASONING_RE.sub("", model_output)
        remainder = _STRIP_OPEN_REASONING_RE.sub("", remainder)
        if "<atem:invoke" in remainder or _FUNCTION_CALLS_OPEN in remainder:
            return reasoning, (remainder or None)
        content_match = _CONTENT_RE.search(model_output)
        if content_match:
            content = content_match.group(1) or None
        elif _REASONING_OPEN in model_output:
            content = None
        else:
            content = model_output or None
            reasoning = None
        return reasoning, content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Channel-aware streaming split of reasoning vs content.
        Classifies the full ``current_text`` and emits only what has not been
        emitted yet, so no framing token is ever surfaced and a delta straddling
        a channel boundary only contributes the portion inside a real body.
        """
        curr_reason, curr_content = self._classify_bodies(current_text)
        reasoning_delta = ""
        if curr_reason.startswith(self._emitted_reasoning) and len(curr_reason) > len(
            self._emitted_reasoning
        ):
            reasoning_delta = curr_reason[len(self._emitted_reasoning) :]
            self._emitted_reasoning = curr_reason
        content_delta = ""
        if curr_content.startswith(self._emitted_content) and len(curr_content) > len(
            self._emitted_content
        ):
            content_delta = curr_content[len(self._emitted_content) :]
            self._emitted_content = curr_content
        # Hand the tool channel to the tool parser exactly once, starting at its
        # header. parse_delta discards anything not returned here.
        #
        # This MUST fire on the same delta where is_reasoning_end() flips, i.e.
        # only once the tool channel actually contains ATEM. Emitting it earlier
        # -- when only the bare `to=<name><|message|>` header has arrived -- keeps
        # the parser in the reasoning phase, so parse_delta never replaces this
        # DeltaMessage with the tool parser's and the header is delivered to the
        # client as visible content.
        handoff = ""
        if not self._tool_handoff_done:
            remainder = self._tool_channel_remainder(current_text)
            if _FUNCTION_CALLS_OPEN in remainder or "<atem:invoke" in remainder:
                handoff = remainder
                self._tool_handoff_done = True
        if handoff:
            return DeltaMessage(reasoning=reasoning_delta or None, content=handoff)
        if reasoning_delta and content_delta:
            return DeltaMessage(reasoning=reasoning_delta, content=content_delta)
        if reasoning_delta:
            return DeltaMessage(reasoning=reasoning_delta)
        if content_delta:
            return DeltaMessage(content=content_delta)
        return None
