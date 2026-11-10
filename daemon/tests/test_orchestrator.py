from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

import numpy as np
import pytest
from orch_fakes import FakeGate, FakeListener, FakeOutput, FakeTTS, wait_until

from macd import config
from macd import phrases as P
from macd.brain.errors import MESSAGES
from macd.brain.fake_brain import FakeBrain, Step, error, say, text, tool
from macd.config import Config
from macd.contracts import (
    BrainErrorKind,
    BrainEvent,
    BrainEventKind,
    ListenerEvent,
    ListenerEventKind,
    State,
    WindowReason,
)
from macd.orchestrator import Orchestrator
from macd.speech.segmenter import SentenceSegmenter
from macd.speech_queue import SpeechQueue
from macd.tasks import supervise

LONG = 30.0  # a step delay that keeps a scripted turn "running" until interrupted


def pause(seconds: float = LONG) -> Step:
    return Step(BrainEventKind.STATUS, text="working", delay=seconds)


def confirm(summary: str, target: str = "") -> Step:
    return Step(BrainEventKind.CONFIRM, text=summary, data={"target": target})


@dataclass
class Rig:
    orch: Orchestrator
    brain: FakeBrain
    gate: FakeGate
    tts: FakeTTS
    out: FakeOutput
    mic: FakeListener
    speech: SpeechQueue
    events: list[dict]
    scripts: dict[str, list[Step]] = field(default_factory=dict)

    @property
    def spoken(self) -> list[str]:
        return [s for s in self.out.spoken if not s.startswith("<earcon")]

    def states(self) -> list[str]:
        return [e["state"] for e in self.events if e["event"] == "state"]

    def inject(self, kind: BrainEventKind, turn_id: int, **kw: object) -> None:
        """Deliver a raw brain event (e.g. a stale one) straight into the brain's event stream."""
        self.brain._out.put_nowait(BrainEvent(kind, turn_id, **kw))  # type: ignore[arg-type]


def make_cfg() -> Config:
    cfg = Config()
    cfg.listen.followup_s = 0.3
    cfg.listen.followup_after_question_s = 0.6
    cfg.listen.confirm_window_s = 0.3
    cfg.brain.narration_silence_s = 60.0
    cfg.brain.max_turn_s = 60
    return cfg


def fake_earcon(name: str) -> tuple[object, int]:
    if name not in ("wake", "listening", "done", "error", "tick"):
        raise KeyError(name)
    return np.zeros(80, dtype=np.float32), 24_000


async def make_rig(cfg: Config | None = None, outputs: dict[str, FakeOutput] | None = None,
                   segmenter: Callable[[], SentenceSegmenter] | None = None) -> tuple[Rig, asyncio.Task]:
    tts = next(iter(outputs.values())).tts if outputs else FakeTTS()
    out = outputs["local"] if outputs else FakeOutput(tts)
    speech = SpeechQueue(tts, outputs or {"local": out})
    scripts: dict[str, list[Step]] = {}
    brain = FakeBrain(lambda t: scripts.get(t, [text("Okay. ")]))
    await brain.start()
    gate, mic = FakeGate(), FakeListener("local")
    orch = Orchestrator(cfg or make_cfg(), brain=brain, gate=gate, speech=speech, earcon=fake_earcon,
                        segmenter_factory=segmenter or (lambda: SentenceSegmenter(min_chars=0)),
                        describe_tool=lambda n, i: f"Using {n}")
    events: list[dict] = []
    orch.subscribe(events.append)
    orch.add_listener(mic)
    task = asyncio.create_task(orch.run())
    await asyncio.sleep(0.01)
    return Rig(orch, brain, gate, tts, out, mic, speech, events, scripts), task


async def teardown(r: Rig, task: asyncio.Task) -> None:
    r.orch.close()
    await asyncio.wait_for(task, 2)
    for tid, _ in r.brain.asks:  # unblock scripted turns parked on a long step
        await r.brain.interrupt(tid)
    await r.brain.close()


@pytest.fixture
async def rig() -> AsyncIterator[Rig]:
    r, task = await make_rig()
    r.brain.alive = True
    try:
        yield r
    finally:
        await teardown(r, task)


async def start_turn(r: Rig, text_: str, steps: list[Step] | None = None, *, woke: bool = True) -> int:
    r.scripts[text_] = steps if steps is not None else [pause()]
    r.mic.say(text_, woke=woke)
    await wait_until(lambda: r.brain.asks and r.brain.asks[-1][1] == text_, msg="ask")
    return r.brain.asks[-1][0]


# ----------------------------------------------------------------------------- phrases (pure)


@pytest.mark.parametrize(
    ("t", "stop"),
    [("stop", True), ("Stop.", True), ("cancel", True), ("never mind", True), ("Quiet!", True), ("shut up", True),
     ("That's enough.", True), ("stop please", True), ("okay stop", True), ("stop the timer at five", False),
     ("cancel my 3pm meeting", False), ("open safari", False), ("", False)],
)
def test_is_stop(t: str, stop: bool) -> None:
    assert P.is_stop(t) is stop


@pytest.mark.parametrize(
    ("t", "verdict"),
    [("yes", "yes"), ("Yes.", "yes"), ("yeah", "yes"), ("Sure!", "yes"), ("go ahead", "yes"), ("send it", "yes"),
     ("Yes please.", "yes"), ("Okay, do it.", "yes"), ("looks good, send it", "yes"), ("that's right", "yes"),
     ("no", "no"), ("nope", "no"), ("hmm", "no"), ("", "no"), ("don't", "no"), ("not now", "no"),
     ("yes but change the subject", "steer"), ("no, send it to Bob instead", "steer"),
     ("sure, after you fix the typo", "steer"), ("yes but wait a second please", "steer")],
)
def test_classify_confirmation(t: str, verdict: str) -> None:
    assert P.classify_confirmation(t) == verdict


@pytest.mark.parametrize(
    ("t", "corr"),
    [("Actually, make it Friday", "make it Friday"), ("no, I meant Bob", "I meant Bob"),
     ("No I meant the other one", "I meant the other one"), ("instead, use the second result", "use the second result"),
     # N13: these are NOT corrections (would cancel a running turn)
     ("wait for the page to load", None), ("no, that's right", None), ("no that's fine", None),
     ("actually never mind", None), ("now open mail", None), ("no", None), ("no bob", None)],
)
def test_correction_text(t: str, corr: str | None) -> None:
    assert P.correction_text(t) == corr


@pytest.mark.parametrize(
    ("t", "off"),
    [("go to sleep", True), ("Turn off.", True), ("okay stop listening", True), ("turn off the lights", False),
     ("sleep", False), ("turn it off", False)],
)
def test_is_off_request(t: str, off: bool) -> None:
    assert P.is_off_request(t) is off


@pytest.mark.parametrize(
    ("t", "ok"),
    [("Then", False), ("um", False), ("the", False), ("Priya", False), ("", False),
     ("continue", True), ("go on", True), ("next", True), ("why", True), ("more", True), ("yes", True), ("no", True),
     ("stop", True), ("three", True), ("42", True), ("open mail", True), ("and the second one", True)],
)
def test_is_followup_intent(t: str, ok: bool) -> None:
    assert P.is_followup_intent(t) is ok


@pytest.mark.parametrize(
    ("t", "ok"),
    [("use your default voice", True), ("Hey, go back to your default voice please", True), ("reset your voice", True),
     ("use your normal voice", True), ("use a different voice", False), ("what's your voice", False), ("", False)],
)
def test_is_voice_reset_request(t: str, ok: bool) -> None:
    assert P.is_voice_reset_request(t) is ok


# ----------------------------------------------------------------------------- memory / handoff


async def test_recent_user_text(rig: Rig) -> None:
    assert rig.orch.recent_user_text() == []
    rig.scripts["first thing"] = [text("Okay. ")]
    rig.mic.say("first thing", woke=True)
    await wait_until(lambda: rig.orch.state == State.FOLLOWUP)
    await rig.orch.ask("second thing")
    await wait_until(lambda: len(rig.brain.asks) == 2)
    rig.mic.say("   ", woke=True)  # empty utterances are not remembered
    await asyncio.sleep(0.02)
    assert rig.orch.recent_user_text() == ["first thing", "second thing"]
    assert rig.orch.recent_user_text(1) == ["second thing"] and rig.orch.recent_user_text(0) == []


def _attach_handoff(rig: Rig, delay: float = 0.0) -> list[str]:
    order: list[str] = []

    async def handoff() -> None:
        order.append("handoff-start")
        await asyncio.sleep(delay)
        order.append("handoff-done")

    real_reset, real_sleep = rig.brain.reset, rig.brain.sleep

    async def reset() -> None:
        order.append("reset")
        await real_reset()

    async def sleep() -> None:
        order.append("sleep")
        await real_sleep()

    rig.brain.handoff = handoff  # type: ignore[attr-defined]
    rig.brain.reset = reset  # type: ignore[method-assign]
    rig.brain.sleep = sleep  # type: ignore[method-assign]
    return order


async def test_handoff_runs_before_reset(rig: Rig) -> None:
    order = _attach_handoff(rig)
    await rig.orch.reset()
    assert order == ["handoff-start", "handoff-done", "reset"]
    assert any(e.get("kind") == "handoff" for e in rig.events)


async def test_handoff_runs_before_off_sleep(rig: Rig) -> None:
    order = _attach_handoff(rig)
    rig.orch.set_off(True)
    await wait_until(lambda: "sleep" in order)
    assert order[:3] == ["handoff-start", "handoff-done", "sleep"]


async def test_wake_during_handoff_is_not_blocked(rig: Rig) -> None:
    order = _attach_handoff(rig, delay=0.3)
    reset_task = asyncio.create_task(rig.orch.reset())
    await wait_until(lambda: order == ["handoff-start"])
    rig.mic.say("what time is it", woke=True)  # arrives mid-handoff
    await wait_until(lambda: rig.brain.asks == [(1, "what time is it")], timeout=0.2)
    assert rig.orch.state in (State.THINKING, State.SPEAKING) and "handoff-done" not in order
    await asyncio.wait_for(reset_task, 2)
    assert order[-2:] == ["handoff-done", "reset"]


async def test_handoff_timeout_does_not_block_reset(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    from macd import orchestrator as orch_mod

    monkeypatch.setattr(orch_mod, "HANDOFF_TIMEOUT_S", 0.05)
    order = _attach_handoff(rig, delay=5.0)
    await asyncio.wait_for(rig.orch.reset(), 2)
    assert order == ["handoff-start", "reset"]


async def test_brain_without_handoff_is_fine(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(type(rig.brain), "handoff")
    assert not hasattr(rig.brain, "handoff")
    await rig.orch.reset()
    assert rig.brain.resets == 1


async def test_real_fake_brain_handoff_counts(rig: Rig) -> None:
    await rig.orch.reset()
    assert rig.brain.handoffs == 1 and rig.brain.resets == 1


async def test_spoken_rename(rig: Rig) -> None:
    names: list[str] = []

    async def rename(name: str) -> None:
        names.append(name)

    rig.orch.rename = rename
    rig.mic.say("call yourself Jarvis", woke=True)
    await wait_until(lambda: names == ["Jarvis"])
    await asyncio.sleep(0.05)
    assert rig.brain.asks == [] and rig.events[-1]["kind"] == "rename"


async def test_spoken_voice_reset(rig: Rig) -> None:
    calls: list[int] = []

    async def reset() -> dict:
        calls.append(1)
        return {}

    rig.orch.voice_reset = reset
    rig.mic.say("use your default voice", woke=True)
    await wait_until(lambda: calls == [1])
    await asyncio.sleep(0.05)
    assert rig.brain.asks == [] and rig.events[-1]["kind"] == "voice_reset"


@pytest.mark.parametrize(
    ("target", "core", "phrase"),
    [("WhatsApp:AXButton Send", "whatsapp", "in WhatsApp"), ("\u200eWhatsApp:AXButton \u200eSend", "whatsapp", "in WhatsApp"),
     ("Slack: Send", "slack", "in Slack"), ("Mail:AXButton Post", "mail", "in Mail"), ("Messages: Priya", "messages", "in Messages"),
     ("linkedin.com", "linkedin.com", "on linkedin.com"), ("https://www.linkedin.com/in/bob", "linkedin.com", "on linkedin.com"),
     ("bob@example.com", "bob@example.com", "to bob@example.com"), ("+1 415 555 0100", "+1 415 555 0100", "to +1 415 555 0100"),
     ("Priya", "priya", "to Priya"), ("", "", "")],
)
def test_humanize_target(target: str, core: str, phrase: str) -> None:
    assert P.humanize_target(target) == (core, phrase)


@pytest.mark.parametrize(
    ("summary", "target", "line"),
    [("I'll send the report to Bob", "", "I'll send the report to Bob. Send it?"),
     ("I'll tell Priya I'm late", "WhatsApp: Priya", "I'll tell Priya I'm late in WhatsApp. Send it?"),
     ("I'll message Priya on WhatsApp", "WhatsApp:AXButton Send", "I'll message Priya on WhatsApp. Send it?"),
     ("Post the update", "linkedin.com", "Post the update on linkedin.com. Post it?"),
     ("Delete the three old drafts.", "Mail: Delete", "Delete the three old drafts in Mail. Delete it?"),
     ("Order two large pizzas for tonight", "dominos.com", "Order two large pizzas for tonight on dominos.com. Go ahead and pay?"),
     ("Run the deploy script", "", "Run the deploy script. Go ahead?"),
     ("Shall I archive it?", "", "Shall I archive it. Go ahead?"),
     ('Send the Timetable WhatsApp group a message saying "Test message from Mac, please ignore."', "\u200eWhatsApp:AXButton \u200eSend",
      'Send the Timetable WhatsApp group a message saying "Test message from Mac, please ignore". Send it?'),
     (("Send Bob the quarterly numbers for the northern region saying that we are on track and asking whether he wants "
       "the detailed breakdown before Friday's meeting"), "Mail: Send",
      "Send Bob the quarterly numbers for the northern region in Mail. Send it?"),
     ("", "WhatsApp: Send", "I'm about to take an action in WhatsApp. Go ahead?")],
)
def test_confirmation_line(summary: str, target: str, line: str) -> None:
    assert P.confirmation_line(summary, target) == line


@pytest.mark.parametrize(
    ("t", "out"),
    [("a — b", "a, b"), ("a – b", "a, b"), ("a - b", "a, b"), ("well-known", "well-known"), ("x—y", "x, y"),
     ("This is taking a while — I'll keep going.", "This is taking a while, I'll keep going."), ("plain", "plain")],
)
def test_dedash(t: str, out: str) -> None:
    assert P.dedash(t) == out


def test_no_dashes_in_spoken_phrases() -> None:
    assert all("—" not in p and "–" not in p for p in P.PREWARM_PHRASES)
    assert "—" not in P.confirmation_line("Send it — to Bob", "") and "—" not in P.confirmation_line("x", "Slack: Send")


async def test_spoken_text_has_no_dashes(rig: Rig) -> None:
    rig.scripts["summarize"] = [text("First point — second point.\n"), pause()]
    rig.mic.say("summarize", woke=True)
    await wait_until(lambda: any("second point" in s for s in rig.spoken))
    assert all("—" not in s for s in rig.spoken)
    rig.orch.say("Phone session — local mic paused")
    await wait_until(lambda: "Phone session, local mic paused" in rig.spoken)


def test_listener_metrics_plumbing() -> None:
    """A's listener feeds the daemon Metrics object it is given (status then shows kws_hits etc.)."""
    from macd.eventlog import Metrics
    from macd.speech.factory import SpeechStack, make_listener

    class Src:
        name = "local"
        requires_wake_word = True

        async def start(self) -> None:
            return

        async def close(self) -> None:
            return

        def frames(self):
            return iter(())

    class Wake:
        def accept(self, frame, idx):
            return None

        def reset(self) -> None:
            return

    class Vad:
        def accept(self, frame):
            return 0.0

        def reset(self) -> None:
            return

    class Stt:
        async def transcribe(self, pcm):
            return ""

    m = Metrics()
    stack = SpeechStack(make_wake=Wake, make_vad=Vad, stt=Stt(), tts=None)  # type: ignore[arg-type]
    lst = make_listener(Src(), stack, make_cfg(), metrics=m)  # type: ignore[arg-type]
    assert lst.metrics is m
    m.incr("kws_hits")
    m.observe("listener_kws", 1.5)
    snap = m.snapshot()
    assert snap["counters"]["kws_hits"] == 1 and snap["latency_ms"]["listener_kws"]["n"] == 1


def test_visible_strips_invisible_chars() -> None:
    assert P.visible("\u200eSend\u200f  now\ufeff") == "Send now"


def test_misc_phrases() -> None:
    assert P.is_question("what's the weather?") and P.is_question("How do I bake bread") and not P.is_question("open mail")
    assert P.spoken_digits("one two three four") == "1234" and P.spoken_digits("12 34") == "1234"
    assert P.pin_matches("the code is four two oh seven", "4207") and not P.pin_matches("four two", "4207")
    assert P.elapsed_phrase(125) == "Two minutes in, still going." and P.elapsed_phrase(50) == "One minute in, still going."
    assert all(isinstance(p, str) and p for p in P.PREWARM_PHRASES)


# ----------------------------------------------------------------------------- FSM: happy path


async def test_wake_ask_answer_in_order_with_say(rig: Rig) -> None:
    rig.scripts["what's the weather"] = [text("It is sunny. Also warm"), say("Checking the forecast."), text(" Bye now.")]
    rig.mic.say("what's the weather", woke=True)
    await wait_until(lambda: rig.orch.state == State.FOLLOWUP, msg="followup")
    assert rig.spoken == [P.FILLER_SHORT_QUESTION, "It is sunny.", "Also warm.", "Bye now."]  # a non-question say is HUD-only
    assert rig.mic.reasons()[-1] == WindowReason.FOLLOWUP
    await wait_until(lambda: rig.orch.state == State.IDLE, timeout=2, msg="idle after followup")
    assert rig.states()[:3] == ["idle", "listening", "thinking"]
    assert "speaking" in rig.states() and rig.states()[-2:] == ["followup", "idle"]
    ends = [e for e in rig.events if e.get("kind") == "turn_end"]
    assert ends and "It is sunny." in ends[0]["text"]
    snap = rig.orch.metrics.snapshot()
    assert snap["latency_ms"]["turn_total"]["n"] == 1 and snap["latency_ms"]["transcript_to_first_audio"]["n"] == 1
    tr = [e for e in rig.events if e["event"] == "transcript"]
    assert tr[0]["role"] == "user" and tr[0]["final"] is True and tr[0]["remote"] is False
    assert all(e["kind"] != "filler" for e in tr if e["role"] == "assistant")  # fillers aren't transcript


async def test_short_question_filler_is_sure(rig: Rig) -> None:
    rig.scripts["what time is it"] = [text("Noon. ")]
    rig.mic.say("what time is it", woke=True)
    await wait_until(lambda: "Noon." in rig.spoken)
    assert rig.spoken[0] == P.FILLER_SHORT_QUESTION


async def test_followup_without_wake_word(rig: Rig) -> None:
    rig.scripts["open my email"] = [text("Done. ")]
    rig.mic.say("open my email", woke=True)
    await wait_until(lambda: rig.orch.state == State.FOLLOWUP)
    rig.mic.say("and reply to the first one", window=WindowReason.FOLLOWUP)
    await wait_until(lambda: len(rig.brain.asks) == 2)
    assert rig.brain.asks[1] == (2, "and reply to the first one")


async def test_followup_fragment_is_ignored_and_window_stays_open(rig: Rig) -> None:
    rig.scripts["open my email"] = [text("Done. ")]
    rig.mic.say("open my email", woke=True)
    await wait_until(lambda: rig.orch.state == State.FOLLOWUP)
    n_windows = len(rig.mic.windows)
    rig.mic.say("Then", window=WindowReason.FOLLOWUP)  # bystander fragment
    await wait_until(lambda: len(rig.mic.windows) == n_windows + 1, msg="window re-opened")
    assert rig.mic.windows[-1][1] == WindowReason.FOLLOWUP and 0 < rig.mic.windows[-1][0] <= 0.3
    assert len(rig.brain.asks) == 1 and rig.orch.state == State.FOLLOWUP
    assert rig.orch.metrics.counters["followup_ignored"] == 1
    rig.mic.say("continue", window=WindowReason.FOLLOWUP)
    await wait_until(lambda: len(rig.brain.asks) == 2)
    assert rig.brain.asks[1] == (2, "continue")


async def test_followup_longer_after_a_question(rig: Rig) -> None:
    rig.scripts["teach me french"] = [text("Bonjour means hello. Ready for the next word?")]
    rig.mic.say("teach me french", woke=True)
    await wait_until(lambda: rig.orch.state == State.FOLLOWUP)
    assert rig.mic.windows[-1] == (0.6, WindowReason.FOLLOWUP)


async def test_idle_without_wake_ignored(rig: Rig) -> None:
    rig.mic.say("open my email", woke=False)
    await asyncio.sleep(0.05)
    assert rig.brain.asks == [] and rig.orch.state == State.IDLE


async def test_no_speaking_thinking_flicker_between_sentences(rig: Rig) -> None:
    rig.scripts["tell me a story"] = [text("Once upon a time. "), text("There was a cat. ", delay=0.05),
                                      text("The end. ", delay=0.05)]
    rig.mic.say("tell me a story", woke=True)
    await wait_until(lambda: rig.orch.state == State.FOLLOWUP)
    assert rig.states().count("speaking") == 1 and rig.states().count("thinking") == 1


# ----------------------------------------------------------------------------- FSM: mid-turn policy


async def test_mid_turn_non_woke_ignored(rig: Rig) -> None:
    tid = await start_turn(rig, "open my email")
    rig.mic.say("blah blah tv in the background", woke=False)
    await asyncio.sleep(0.05)
    assert rig.brain.steers == [] and rig.brain.interrupts == [] and len(rig.brain.asks) == 1
    assert rig.orch.turn is not None and rig.orch.turn.id == tid


async def test_woke_stop_interrupts_and_drops_stale(rig: Rig) -> None:
    tid = await start_turn(rig, "open my email", [text("Opening Mail now.\n"), pause()])
    await wait_until(lambda: "Opening Mail now." in rig.spoken)
    rig.mic.say("stop", woke=True)
    await wait_until(lambda: rig.brain.interrupts == [tid])
    assert rig.gate.revokes >= 1 and rig.orch.state == State.IDLE and rig.orch.turn is None
    rig.inject(BrainEventKind.TEXT, tid, text="Here are your emails. ")
    await wait_until(lambda: P.STOPPED in rig.spoken)
    await asyncio.sleep(0.05)
    assert "Here are your emails." not in rig.spoken
    assert rig.orch.metrics.counters["stale_brain_events"] >= 1 and rig.orch.state == State.IDLE


async def test_bare_stop_while_speaking(rig: Rig) -> None:
    """UX#1: the listener's SPEAKING window delivers bare stop words while Mac talks."""
    rig.out.play_s = 0.2
    tid = await start_turn(rig, "read my email", [text("You have forty messages.\n"), pause()])
    await wait_until(lambda: rig.speech.speaking)
    rig.mic.say("stop", window=WindowReason.SPEAKING)
    await wait_until(lambda: rig.brain.interrupts == [tid])
    await wait_until(lambda: P.STOPPED in rig.spoken, timeout=3)
    assert "You have forty messages." not in rig.spoken  # cut off mid-sentence


@pytest.mark.parametrize("bystander", ["add their titles", "actually, let's get lunch", "no, make it Friday",
                                       "hey what's for dinner", "never mind the lunch plans"])
async def test_speaking_window_only_stop_words_count(rig: Rig, bystander: str) -> None:
    """DECISION 11: without the wake word, only a bare stop word is honoured while Mac talks."""
    tid = await start_turn(rig, "read my email")
    rig.mic.say(bystander, window=WindowReason.SPEAKING)
    await asyncio.sleep(0.05)
    assert rig.brain.steers == [] and rig.brain.interrupts == [] and len(rig.brain.asks) == 1
    assert rig.orch.turn is not None and rig.orch.turn.id == tid


async def test_steer_ack_jumps_the_queue(rig: Rig) -> None:
    rig.out.play_s = 0.03
    tid = await start_turn(rig, "list my contacts", [text("Alice is first. Bob is second. Carol is third. Dave is last.\n"), pause()])
    await wait_until(lambda: "Alice is first." in rig.spoken)
    rig.mic.say("also add their titles", woke=True)
    await wait_until(lambda: rig.brain.steers == [(tid, "also add their titles")])
    await wait_until(lambda: P.STEER_ACK in rig.spoken, timeout=3)
    await wait_until(lambda: "Dave is last." in rig.spoken, timeout=3)
    assert rig.spoken.index(P.STEER_ACK) < rig.spoken.index("Dave is last.")  # UX#2
    assert rig.brain.interrupts == [] and rig.orch.turn is not None and rig.orch.turn.id == tid


async def test_continuation_steers_without_ack(rig: Rig) -> None:
    tid = await start_turn(rig, "take a note")
    rig.mic.say("and the second paragraph says", continuation=True)
    await wait_until(lambda: rig.brain.steers == [(tid, "and the second paragraph says")])
    await asyncio.sleep(0.05)
    assert P.STEER_ACK not in rig.spoken


async def test_correction_interrupts_and_redirects(rig: Rig) -> None:
    tid = await start_turn(rig, "book the meeting for Thursday")
    rig.scripts["actually, make it Friday"] = [text("Booked Friday. ")]
    rig.mic.say("actually, make it Friday", woke=True)
    await wait_until(lambda: len(rig.brain.asks) == 2)
    assert rig.brain.interrupts == [tid] and rig.brain.asks[1] == (tid + 1, "actually, make it Friday")
    await wait_until(lambda: "Booked Friday." in rig.spoken)


async def test_agreement_is_a_steer_not_a_correction(rig: Rig) -> None:
    tid = await start_turn(rig, "book the meeting for Thursday")
    rig.mic.say("no, that's right", woke=True)
    await wait_until(lambda: rig.brain.steers == [(tid, "no, that's right")])
    assert rig.brain.interrupts == []


# ----------------------------------------------------------------------------- confirmation


async def _confirm(rig: Rig) -> int:
    tid = await start_turn(rig, "email Bob the report", [confirm("I'll send the report to Bob"), pause()])
    await wait_until(lambda: rig.orch.state == State.CONFIRMING)
    await wait_until(lambda: WindowReason.CONFIRM in rig.mic.reasons(), msg="confirm window")
    assert rig.spoken[-1] == "I'll send the report to Bob. Send it?"
    return tid


@pytest.mark.parametrize(("answer", "approved"), [("yes", True), ("Go ahead.", True), ("looks good, send it", True),
                                                   ("no", False), ("hmm", False)])
async def test_confirm_answers(rig: Rig, answer: str, approved: bool) -> None:
    await _confirm(rig)
    n_spoken = len(rig.spoken)
    rig.mic.say(answer, window=WindowReason.CONFIRM)
    await wait_until(lambda: rig.gate.resolved == [approved])
    assert rig.orch.state in (State.THINKING, State.SPEAKING)
    if approved:
        await asyncio.sleep(0.05)
        assert len(rig.spoken) == n_spoken  # no ack of ours after approval
    else:
        await wait_until(lambda: P.DENIED in rig.spoken)
    assert rig.brain.steers == []


async def test_confirm_yes_but_denies_and_steers(rig: Rig) -> None:
    tid = await _confirm(rig)
    rig.mic.say("yes but change the subject", window=WindowReason.CONFIRM)
    await wait_until(lambda: rig.gate.resolved == [False])
    await wait_until(lambda: rig.brain.steers == [(tid, "yes but change the subject")])
    await wait_until(lambda: P.STEER_ACK in rig.spoken)
    assert P.DENIED not in rig.spoken


async def test_confirm_timeout_and_no_speech_deny(rig: Rig) -> None:
    await _confirm(rig)
    await wait_until(lambda: rig.gate.resolved == [False], timeout=3)
    await wait_until(lambda: P.CONFIRM_TIMEOUT in rig.spoken)


async def test_confirm_no_speech_denies(rig: Rig) -> None:
    await _confirm(rig)
    rig.mic.push(ListenerEvent(ListenerEventKind.NO_SPEECH, "local", window=WindowReason.CONFIRM))
    await wait_until(lambda: rig.gate.resolved == [False])


async def test_text_ask_during_confirming_answers_it() -> None:
    """Live bug: `macctl ask yes` must resolve the confirmation, never start a new turn."""
    r, task = await make_rig()
    r.brain.alive = True
    try:
        r.scripts["email Bob the report"] = [confirm("I'll send the report to Bob"), *tool("send", duration=LONG)]
        r.mic.say("email Bob the report", woke=True)
        await wait_until(lambda: r.orch.state == State.CONFIRMING)
        assert await r.orch.ask("yes") == 1
        await wait_until(lambda: r.gate.resolved == [True])
        assert len(r.brain.asks) == 1 and r.orch.turn is not None and r.orch.turn.id == 1
        await wait_until(lambda: any(e.get("kind") == "tool_start" and e.get("tool") == "send" for e in r.events))
        res = [e for e in r.events if e.get("kind") == "confirm_result"]
        assert res[-1]["approved"] is True and res[-1]["confirm_source"] == "text"
        assert "superseded" not in [e.get("reason") for e in r.events if e["event"] == "state"]
    finally:
        await teardown(r, task)


async def test_text_no_during_confirming_denies(rig: Rig) -> None:
    await _confirm(rig)
    assert await rig.orch.ask("no") == 1
    await wait_until(lambda: rig.gate.resolved == [False])
    await wait_until(lambda: P.DENIED in rig.spoken)
    assert len(rig.brain.asks) == 1 and rig.orch.turn is not None and rig.orch.turn.id == 1


async def test_confirm_command(rig: Rig) -> None:
    with pytest.raises(ValueError, match="nothing pending"):
        rig.orch.confirm(True)
    await _confirm(rig)
    assert rig.orch.confirm(True) == 1
    await wait_until(lambda: rig.gate.resolved == [True])
    res = [e for e in rig.events if e.get("kind") == "confirm_result"][-1]
    assert res["confirm_source"] == "command" and res["approved"] is True
    with pytest.raises(ValueError, match="nothing pending"):
        rig.orch.confirm(False)


async def test_text_ask_in_followup_and_mid_turn_follows_policy(rig: Rig) -> None:
    tid = await start_turn(rig, "list my contacts")
    assert await rig.orch.ask("also add their titles") == tid  # mid-turn text → steer, not a new turn
    await wait_until(lambda: rig.brain.steers == [(tid, "also add their titles")])
    assert await rig.orch.ask("stop") == tid
    await wait_until(lambda: rig.brain.interrupts == [tid])
    assert rig.orch.turn is None


async def test_confirm_speaks_humanized_target_once(rig: Rig) -> None:
    live = 'Send the Timetable WhatsApp group a message saying "Test message from Mac, please ignore."'
    await start_turn(rig, "message the group", [confirm(live, "\u200eWhatsApp:AXButton \u200eSend"), pause()])
    await wait_until(lambda: rig.orch.state == State.CONFIRMING)
    await wait_until(lambda: any(s.endswith("Send it?") for s in rig.spoken))
    line = next(s for s in rig.spoken if s.endswith("Send it?"))
    assert line == 'Send the Timetable WhatsApp group a message saying "Test message from Mac, please ignore". Send it?'
    assert "AXButton" not in line and "\u200e" not in line
    tr = [e for e in rig.events if e["event"] == "transcript" and e.get("kind") == "confirm"]
    assert len(tr) == 1 and tr[0]["text"] == line
    assert not any(s.startswith("<earcon") for s in rig.out.spoken) and P.SLOW_FILLER not in rig.spoken


async def test_second_confirm_while_pending_is_ignored(rig: Rig) -> None:
    tid = await _confirm(rig)
    rig.inject(BrainEventKind.CONFIRM, tid, text="Another thing")
    await asyncio.sleep(0.05)
    assert sum(1 for e in rig.events if e.get("kind") == "confirm") == 1
    assert rig.orch.metrics.counters["confirms"] == 1


# ----------------------------------------------------------------------------- errors / liveness


async def test_error_event_spoken_from_one_table(rig: Rig) -> None:
    await start_turn(rig, "open my email", [error(BrainErrorKind.AUTH, "")])
    await wait_until(lambda: MESSAGES[BrainErrorKind.AUTH] in rig.spoken)
    errs = [e for e in rig.events if e["event"] == "error"]
    assert errs and errs[0]["kind"] == "auth"
    assert any(s.startswith("<earcon") for s in rig.out.spoken)
    await wait_until(lambda: rig.orch.state == State.IDLE)


async def test_tool_steps_shown_on_hud_never_spoken() -> None:
    cfg = make_cfg()
    cfg.brain.narration_silence_s = 0.1
    r, task = await make_rig(cfg)
    r.brain.alive = True
    try:
        r.scripts["find Bob on linkedin"] = [*tool("navigate", {"url": "linkedin.com"}, duration=LONG)]
        r.mic.say("find Bob on linkedin", woke=True)
        await wait_until(lambda: any(e["event"] == "activity" and e["kind"] == "narration"
                                     and e["text"] == "Using navigate" for e in r.events), timeout=2)
        await asyncio.sleep(0.35)
        assert "Using navigate" not in r.spoken  # step lines are HUD-only
        assert not any(s.startswith("Still using") for s in r.spoken)
    finally:
        await teardown(r, task)


async def test_slow_start_tick_and_still_working(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(P, "SLOW_BLIP_S", 0.05)
    monkeypatch.setattr(P, "SLOW_SAY_S", 0.2)
    rig.brain.alive = False  # claude process asleep: connect latency ahead
    tid = await start_turn(rig, "what's the weather")
    await wait_until(lambda: rig.spoken)
    assert rig.spoken[0] == P.SLOW_FILLER
    await wait_until(lambda: any(s.startswith("<earcon") for s in rig.out.spoken), msg="tick")
    await wait_until(lambda: P.STILL_WORKING in rig.spoken, msg="still working")
    assert rig.spoken.count(P.STILL_WORKING) == 1
    rig.inject(BrainEventKind.TEXT, tid, text="Sunny.\n")
    await wait_until(lambda: "Sunny." in rig.spoken)


async def test_fast_brain_gets_no_slow_start_chatter(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(P, "SLOW_BLIP_S", 0.05)
    monkeypatch.setattr(P, "SLOW_SAY_S", 0.1)
    rig.scripts["open mail"] = [text("Opening Mail.\n"), pause()]
    rig.mic.say("open mail", woke=True)
    await wait_until(lambda: "Opening Mail." in rig.spoken)
    await asyncio.sleep(0.2)
    assert P.STILL_WORKING not in rig.spoken and rig.spoken[0] in P.FILLER_ACTION


async def test_watchdog_resets_a_silent_brain() -> None:
    cfg = make_cfg()
    cfg.brain.max_turn_s = -59  # limit = max_turn_s + 60 s → 1 s
    r, task = await make_rig(cfg)
    r.brain.alive = True
    try:
        await start_turn(r, "hang forever")
        await wait_until(lambda: P.VOICE_LOST in r.spoken, timeout=3)
        await wait_until(lambda: r.brain.resets == 1)
        assert r.orch.metrics.counters["watchdog_resets"] == 1
        await wait_until(lambda: r.orch.state == State.IDLE)
    finally:
        await teardown(r, task)


async def test_stalled_output_still_ends_the_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """MINOR-3: play() never completes → the turn ends with an error, never hangs."""
    from macd import orchestrator as orch_mod

    monkeypatch.setattr(orch_mod, "SPEECH_STALL_S", 0.2)
    r, task = await make_rig()
    r.brain.alive = True

    async def never(pcm: object, sr: int) -> None:
        await asyncio.Event().wait()

    r.out.play = never  # type: ignore[method-assign]
    try:
        r.scripts["say hi"] = [text("Hi there.\n")]
        r.mic.say("say hi", woke=True)
        await wait_until(lambda: r.brain.asks)
        await wait_until(lambda: r.orch.state == State.IDLE and r.orch.turn is None, timeout=3)
        errs = [e for e in r.events if e["event"] == "error"]
        assert errs and errs[0]["kind"] == "timeout" and "stalled" in errs[0]["message"]
        assert r.orch.metrics.counters["output_stalls"] == 1
    finally:
        await teardown(r, task)


async def test_stalled_output_cannot_stick_confirming(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    from macd import orchestrator as orch_mod

    monkeypatch.setattr(orch_mod, "CONFIRM_SPEECH_SLACK_S", 0.1)

    async def never(pcm: object, sr: int) -> None:
        await asyncio.Event().wait()

    rig.out.play = never  # type: ignore[method-assign]
    await start_turn(rig, "email Bob", [confirm("I'll email Bob"), pause()])
    await wait_until(lambda: rig.orch.state == State.CONFIRMING)
    await wait_until(lambda: rig.gate.resolved == [False], timeout=3)  # window 0.3 + slack 0.1
    assert rig.orch.state != State.CONFIRMING


async def test_trailing_done_after_interrupt_is_ignored(rig: Rig) -> None:
    """MINOR-3: a DONE for a turn the orchestrator already closed must not touch the new turn."""
    tid = await start_turn(rig, "first task")
    rig.scripts["actually, do the second task"] = [pause()]
    rig.mic.say("actually, do the second task", woke=True)
    await wait_until(lambda: len(rig.brain.asks) == 2)
    rig.inject(BrainEventKind.DONE, tid, data={"interrupted": True})
    rig.inject(BrainEventKind.TEXT, tid, text="late text.\n")
    await asyncio.sleep(0.05)
    assert rig.orch.turn is not None and rig.orch.turn.id == tid + 1 and not rig.orch.turn.brain_done
    assert rig.orch.state in (State.THINKING, State.SPEAKING) and "late text." not in rig.spoken
    assert rig.orch.metrics.counters["stale_brain_events"] >= 2


async def test_push_to_talk_counts_as_wake(rig: Rig) -> None:
    assert rig.orch.listen() == "local"
    assert rig.mic.reasons()[-1] == WindowReason.PTT and rig.orch.state == State.LISTENING
    rig.mic.say("what time is it", window=WindowReason.PTT)
    await wait_until(lambda: rig.brain.asks == [(1, "what time is it")])


async def test_wake_blip_only_after_a_pause(rig: Rig) -> None:
    rig.mic.wake()
    rig.mic.push(ListenerEvent(ListenerEventKind.SPEECH_START, "local"))  # same breath
    await asyncio.sleep(0.4)
    assert not any(s.startswith("<earcon") for s in rig.out.spoken)
    rig.mic.wake()
    await wait_until(lambda: any(s.startswith("<earcon") for s in rig.out.spoken), timeout=1)


async def test_screen_lock_events(rig: Rig) -> None:
    ctx: list[dict] = []

    async def set_context(**kw: object) -> None:
        ctx.append(dict(kw))

    rig.brain.set_context = set_context  # type: ignore[attr-defined]
    await rig.orch.power("screen_locked")
    assert rig.orch.locked is True and rig.orch.status()["locked"] is True and ctx == [{"locked": True}]
    st = [e for e in rig.events if e["event"] == "state"][-1]
    assert st["locked"] is True and st["reason"] == "lock" and not rig.mic.muted
    await rig.orch.power("screen_locked")  # repeat (app re-sends on connect): brain told again, no new event
    assert ctx == [{"locked": True}, {"locked": True}] and [e for e in rig.events if e["event"] == "state"][-1] is st
    tid = await start_turn(rig, "what time is it")  # listening and turns continue while locked
    assert rig.orch.turn is not None and rig.orch.turn.id == tid
    await rig.orch.power("screen_unlocked")
    assert rig.orch.locked is False and ctx[-1] == {"locked": False}
    assert [e for e in rig.events if e["event"] == "state"][-1]["locked"] is False


async def test_screen_lock_without_brain_support(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(type(rig.brain), "set_context")
    await rig.orch.power("screen_locked")
    assert rig.orch.locked is True


async def test_screen_lock_reaches_real_fake_brain(rig: Rig) -> None:
    await rig.orch.power("screen_locked")
    assert rig.orch.locked is True  # B1's FakeBrain.set_context(locked=...) accepted the call


async def test_mute_and_power(rig: Rig) -> None:
    rig.orch.set_muted(True)
    assert rig.mic.muted and rig.events[-1]["event"] == "state" and rig.events[-1]["muted"] is True
    rig.orch.set_muted(False)
    assert not rig.mic.muted
    await rig.orch.power("will_sleep")
    assert rig.mic.muted
    await rig.orch.power("did_wake")
    assert not rig.mic.muted


# ----------------------------------------------------------------------------- off switch


async def test_off_and_on_commands(rig: Rig) -> None:
    config.update_state(session_id="abc")  # the brain's keys must survive (STATE_PATH is per-test)
    await start_turn(rig, "open my email")
    rig.orch.set_off(True)
    assert rig.orch.state == State.OFF and rig.mic.muted
    await wait_until(lambda: rig.brain.sleeps >= 1 and rig.brain.interrupts == [1])
    await wait_until(lambda: config.read_state().get("listening_off") is True, msg="persisted")
    assert config.read_state() == {"session_id": "abc", "listening_off": True}
    st = rig.orch.status()
    assert st["off"] is True and st["state"] == "off" and st["brain_alive"] is False and rig.orch.listen() is None
    await asyncio.sleep(0.02)
    assert rig.orch.state == State.OFF  # the cancelled turn must not flip us back to idle
    warms = rig.brain.warms
    rig.orch.set_off(False)
    assert rig.orch.state == State.IDLE and not rig.mic.muted and rig.brain.warms == warms
    await wait_until(lambda: config.read_state().get("listening_off") is False)
    await wait_until(lambda: rig.out.spoken and rig.out.spoken[-1].startswith("<earcon"))  # UX#21 feedback


async def test_off_reissues_sleep_until_brain_is_asleep(rig: Rig) -> None:
    """N8: the brain ignores sleep() while a turn drains; the orchestrator keeps asking."""
    real_sleep = rig.brain.sleep
    calls = 0

    async def stubborn() -> None:
        nonlocal calls
        calls += 1
        if calls >= 2:
            await real_sleep()

    rig.brain.sleep = stubborn  # type: ignore[method-assign]
    await start_turn(rig, "open my email")
    rig.orch.set_off(True)
    await wait_until(lambda: rig.brain.alive is False, timeout=3)
    assert calls == 2


async def test_spoken_off_request(rig: Rig) -> None:
    rig.mic.say("go to sleep", woke=True)
    await wait_until(lambda: P.GOING_OFF in rig.spoken)
    await wait_until(lambda: rig.orch.state == State.OFF)
    await wait_until(lambda: rig.brain.sleeps >= 1)  # after the (instant) handoff
    assert rig.brain.asks == [] and rig.brain.handoffs == 1


async def test_restored_off_is_kept_at_start() -> None:
    r, task = await make_rig()
    try:
        r.events.clear()
        r.orch.close()
        await asyncio.wait_for(task, 2)
        r.orch.set_off(True, persist=False)
        r.orch._stop.clear()
        task = asyncio.create_task(r.orch.run())
        await asyncio.sleep(0.05)
        assert "idle" not in r.states() and r.states()[-1] == "off"
        assert r.orch.status()["off"] is True and r.mic.muted and r.orch.listen() is None
    finally:
        await teardown(r, task)


async def test_warm_on_wake(rig: Rig) -> None:
    rig.mic.wake()
    await wait_until(lambda: rig.brain.warms == 1)
    rig.orch.cfg.brain.warm_on_wake = False
    rig.mic.wake()
    await asyncio.sleep(0.02)
    assert rig.brain.warms == 1


# ----------------------------------------------------------------------------- remote (N7)


async def test_remote_pin_gate_and_preemption() -> None:
    tts = FakeTTS()
    local, remote_out = FakeOutput(tts, "local"), FakeOutput(tts, "remote")
    r, task = await make_rig(outputs={"local": local, "remote": remote_out})
    r.brain.alive = True
    phone = FakeListener("remote:phone")
    try:
        await start_turn(r, "play some music")
        r.orch.on_remote_session("4207")
        assert r.events[-1] == {"event": "remote", "active": True, "present": False, "verified": False, "pin": "4207",
                                "identity": None}
        r.orch.add_listener(phone, requires_wake=False)
        r.orch.set_remote_present(True, "phone")
        await wait_until(lambda: r.brain.interrupts == [1])
        await wait_until(lambda: P.PHONE_PREEMPTED in local.spoken)
        phone.say("what's on my calendar")  # before the PIN: refused
        await wait_until(lambda: P.PIN_PROMPT in remote_out.spoken)
        assert len(r.brain.asks) == 1
        phone.say("one two three four")
        await wait_until(lambda: P.PIN_WRONG in remote_out.spoken)
        phone.say("four two zero seven")
        await wait_until(lambda: P.PIN_OK in remote_out.spoken)
        assert r.orch.remote["verified"] is True
        r.scripts["what's on my calendar"] = [text("You have two meetings.\n"), pause()]
        phone.say("what's on my calendar")
        await wait_until(lambda: len(r.brain.asks) == 2)
        assert r.orch.turn is not None and r.orch.turn.output == "remote"
        await wait_until(lambda: "You have two meetings." in remote_out.spoken)
        assert "You have two meetings." not in local.spoken
        mirrored = [e for e in r.events if e["event"] == "transcript" and e.get("remote")]
        assert any(e["role"] == "assistant" and e["text"] == "You have two meetings." for e in mirrored)
        r.mic.say("stop", woke=True)  # local voice is ignored while the phone is present
        await asyncio.sleep(0.05)
        assert r.orch.turn is not None and r.orch.turn.id == 2
        r.orch.on_remote_session(None)
        assert r.orch.remote["active"] is False and r.orch.remote["present"] is False
    finally:
        await teardown(r, task)


async def test_phone_connected_spoken_when_idle(rig: Rig) -> None:
    from macd.orchestrator import PHONE_SESSION_DETAIL

    rig.orch.on_remote_session("1111")
    rig.orch.set_remote_present(True, "phone")
    await wait_until(lambda: P.PHONE_CONNECTED in rig.spoken)
    st = [e for e in rig.events if e["event"] == "state"][-1]
    assert st["reason"] == "remote" and st["voice"] is True and st["voice_reason"] == PHONE_SESSION_DETAIL
    rig.orch.set_remote_present(False)
    assert [e for e in rig.events if e["event"] == "state"][-1]["voice_reason"] == ""


# ----------------------------------------------------------------------------- commands / notify / supervise


async def test_commands(rig: Rig) -> None:
    rig.scripts["hello there"] = [pause()]
    tid = await rig.orch.ask("hello there")
    assert tid == 1
    await wait_until(lambda: rig.brain.asks == [(1, "hello there")])
    assert rig.orch.status()["turn_id"] == 1 and rig.orch.status()["remote"]["active"] is False
    assert await rig.orch.stop() is True and rig.brain.interrupts == [1]
    rig.orch.say("Hello world.")
    await wait_until(lambda: "Hello world." in rig.spoken)
    await rig.orch.reset()
    assert rig.brain.resets == 1
    with pytest.raises(ValueError):
        await rig.orch.ask("   ")
    with pytest.raises(ValueError):
        await rig.orch.power("explode")
    rig.orch.set_voice(False, "speech models missing")
    rig.orch.set_muted(False)
    assert rig.events[-1]["voice"] is False and rig.events[-1]["voice_reason"] == "speech models missing"


async def test_supervise_restarts_after_crash() -> None:
    calls = 0
    errors: list[str] = []

    async def flaky() -> None:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RuntimeError("boom")

    await asyncio.wait_for(supervise("flaky", flaky, backoff=(0.001, 0.01), on_error=lambda n, e: errors.append(n)), 2)
    assert calls == 3 and errors == ["flaky", "flaky"]


async def test_listener_crash_is_supervised() -> None:
    class Crashy(FakeListener):
        runs = 0

        async def run(self) -> None:
            Crashy.runs += 1
            if Crashy.runs < 2:
                raise RuntimeError("audio exploded")
            await asyncio.Event().wait()

    r, task = await make_rig()
    r.orch.add_listener(Crashy("crashy"))
    try:
        await wait_until(lambda: Crashy.runs >= 2, timeout=3)
        assert any(e["event"] == "error" for e in r.events)
    finally:
        await teardown(r, task)


def test_state_json_isolated_from_home(tmp_path) -> None:
    assert str(config.STATE_PATH).startswith(str(tmp_path))
    json.dumps(config.read_state())


async def test_step_lines_before_tools_are_not_spoken(rig: Rig) -> None:
    rig.scripts["send it to the team"] = [text("Note's done, opening WhatsApp now."), *tool("open_app", {"name": "WhatsApp"}),
                                          text("Typing the message into the compose field."), *tool("ui_type", {"text": "hi"}),
                                          text("Sent.")]
    rig.mic.say("send it to the team", woke=True)
    await wait_until(lambda: rig.orch.state == State.FOLLOWUP, msg="followup")
    assert "Sent." in rig.spoken
    assert not any("WhatsApp now" in x or "compose" in x for x in rig.spoken)
    narr = [e["text"] for e in rig.events if e["event"] == "activity" and e["kind"] == "narration"]
    assert any("opening WhatsApp" in t for t in narr)  # shown on the HUD instead


async def test_question_say_is_spoken(rig: Rig) -> None:
    rig.scripts["book it"] = [say("Which day works, Monday or Tuesday?"), pause()]
    rig.mic.say("book it", woke=True)
    await wait_until(lambda: "Which day works, Monday or Tuesday?" in rig.spoken, msg="question")
