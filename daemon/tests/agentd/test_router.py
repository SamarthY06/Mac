"""Router (W3.5): follow-up, exact names, Spotlight escaping (N-a), skip list, the warm model's last
JSON object, the threshold, and the route question."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentd import router as rt
from agentd.workspaces import WorkspaceStore, spotlight_find, spotlight_query

NOW = 1_800_000_000.0


class Model:
    def __init__(self, reply: str | Exception = "") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def sess(cwd: Path, sid: str, title: str, days_ago: float = 1) -> SimpleNamespace:
    return SimpleNamespace(cwd=str(cwd), session_id=sid, custom_title=title, summary="",
                           last_modified=(NOW - days_ago * 86400) * 1000)


@pytest.fixture
def home(tmp_path: Path) -> Path:
    h = tmp_path / "home"
    for d in ("Desktop/Personal/Resume", "Desktop/Personal/Blog", "Code/Mac", "Library/Mail",
              "Code/proj/node_modules/Resume"):
        (h / d).mkdir(parents=True)
    (h / "Desktop/Personal/Resume/CLAUDE.md").write_text("# Job search and resumes\n")
    return h


def make(home: Path, reply: str | Exception = "", sessions: list | None = None, mdfind: str = ""):
    ws = WorkspaceStore(home.parent / "ws.json", home, seeds=())
    ws.approved[str(home / "Desktop/Personal/Resume")] = {"hash": "x", "aliases": ["cv"]}
    calls: list[list[str]] = []

    def find(args: list[str]) -> str:
        calls.append(args)
        return mdfind

    def ls(directory: str | None = None, limit: int | None = None, include_worktrees: bool = True):
        assert include_worktrees is False or directory is None  # F9: never worktrees per folder
        items = sessions or []
        if directory:
            items = [i for i in items if i.cwd == directory]
        return items[:limit]

    model = Model(reply)
    r = rt.Router(ws, model, list_sessions=ls, mdfind=find, clock=lambda: NOW)
    return r, model, calls


async def test_followup_within_30_min_keeps_folder_and_session(home):
    r, model, _ = make(home)
    out = await r.route("and the next one", folder=home / "Code/Mac", last_answer_at=NOW - 600, fresh=False, named=None)
    assert out == rt.Route(home / "Code/Mac", "same")
    assert model.prompts == []


async def test_new_session_skips_followup(home):
    r, model, _ = make(home, '{"folder": "general", "session": "new", "confidence": 0.9}')
    out = await r.route("hello", folder=home / "Code/Mac", last_answer_at=NOW - 60, fresh=True, named=None)
    assert out.folder is None and out.session == "new" and model.prompts


async def test_exact_name_in_text_wins_without_model(home):
    r, model, _ = make(home)
    out = await r.route("find jobs in Resume please", folder=None, last_answer_at=0, fresh=False, named=None)
    assert out.folder == home / "Desktop/Personal/Resume" and out.session == "new" and model.prompts == []


async def test_alias_and_named_folder_from_new_session(home):
    r, _, _ = make(home)
    out = await r.route("anything", folder=None, last_answer_at=0, fresh=True, named="CV")
    assert out.folder == home / "Desktop/Personal/Resume"


async def test_unknown_name_goes_to_spotlight_with_escaping(home):
    r, _, calls = make(home, mdfind=f"{home}/Code/Mac\n{home}/Library/Mail\n/etc\n")
    out = await r.route("x", folder=None, last_answer_at=0, fresh=False, named='Ma"c\\')
    assert calls[0][:3] == ["mdfind", "-onlyin", str(home)]
    assert calls[0][3] == 'kMDItemContentType == "public.folder" && kMDItemFSName == "Ma\\"c\\\\"c'
    assert out.folder == home / "Code/Mac"  # Library and /etc dropped


@pytest.mark.parametrize("name", ["Res*", "Res?me", "x" * 81, "a\nb", ""])
def test_spotlight_rejects_globs_long_and_control_names(name):
    assert spotlight_query(name) is None


def test_spotlight_drops_skip_list_hits(home):
    out = f"{home}/Code/proj/node_modules/Resume\n{home}/Desktop/Personal/Resume\n"
    assert spotlight_find("Resume", home, lambda a: out) == [home / "Desktop/Personal/Resume"]


async def test_several_hits_ask_by_path_tail(home):
    (home / "Old/Blog").mkdir(parents=True)
    r, _, _ = make(home, mdfind=f"{home}/Desktop/Personal/Blog\n{home}/Old/Blog\n")
    out = await r.route("edit the post in the Blog folder", folder=None, last_answer_at=0, fresh=False, named=None)
    assert out.question.startswith("Which Blog do you mean")
    assert [o.label for o in out.options] == ["Personal/Blog", "Old/Blog"]


async def test_model_pick_above_threshold_continue_session(home):
    s1 = sess(home / "Desktop/Personal/Resume", "s-r1", "AI job hunt")
    r, model, _ = make(home, 'Sure. {"folder": "x"} then {"folder": "Resume", "session": "continue:s-r1", '
                             '"confidence": 0.7}', sessions=[s1])
    out = await r.route("20 more jobs like last time", folder=None, last_answer_at=0, fresh=False, named=None)
    assert out == rt.Route(home / "Desktop/Personal/Resume", "continue:s-r1")
    assert "s-r1" in model.prompts[0] and "Job search and resumes" in model.prompts[0]


def test_threshold_is_pinned():
    assert rt.CONFIDENCE == 0.7


async def test_low_confidence_asks_a_route_question(home):
    r, _, _ = make(home, '{"folder": "Resume", "session": "new", "confidence": 0.5}')
    out = await r.route("tidy things up", folder=None, last_answer_at=0, fresh=False, named=None)
    assert out.question == "Should I do this in Resume, or just generally?"
    assert [o.folder for o in out.options] == [home / "Desktop/Personal/Resume", None]


async def test_low_confidence_continue_asks_continue_or_fresh(home):
    s1 = sess(home / "Desktop/Personal/Resume", "s-r1", "the job search")
    r, _, _ = make(home, '{"folder": "Resume", "session": "continue:s-r1", "confidence": 0.4}', sessions=[s1])
    out = await r.route("jobs", folder=None, last_answer_at=0, fresh=False, named=None)
    assert out.question.startswith('Continue "the job search" from ') and out.question.endswith(", or start fresh?")


@pytest.mark.parametrize("reply", ["no json here", '{"folder": "Nowhere", "session": "new", "confidence": 1}',
                                   '{"folder": "Resume", "session": "continue:fake", "confidence": 1}',
                                   '{"folder": "Resume", "session": "maybe", "confidence": 1}',
                                   RuntimeError("down")])
async def test_bad_model_reply_falls_back(home, reply):
    r, _, _ = make(home, reply)
    out = await r.route("hm", folder=None, last_answer_at=0, fresh=False, named=None)
    assert out == rt.Route(None, "new")
    out = await r.route("hm", folder=home / "Desktop/Personal/Resume", last_answer_at=0, fresh=False, named=None)
    assert out.question == "Should I do this in Resume, or just generally?"


async def test_candidates_recent_folders_filtered(home):
    sessions = [sess(home / "Code/Mac", "m1", "agent"), sess(home / "Library/Mail", "x", "no"),
                sess(Path("/etc"), "y", "no"), sess(home / "Gone", "z", "no"),
                sess(home / "Desktop/Personal/Blog", "old", "too old", days_ago=40)]
    r, _, _ = make(home, sessions=sessions)
    names = [c.name for c in r.candidates()]
    assert names == ["Resume", "Mac"]


async def test_no_home_in_route_strings(home):
    r, _, _ = make(home, '{"folder": "Resume", "session": "new", "confidence": 0.1}')
    out = await r.route("x", folder=None, last_answer_at=0, fresh=False, named=None)
    assert "Home" not in out.question and all("Home" not in o.label for o in out.options)


def test_parse_last_json():
    assert rt.parse_last_json('I think {"a": 1} and {"b": {"c": 2}}') == {"b": {"c": 2}}
    assert rt.parse_last_json("{broken") is None
    assert rt.parse_last_json('[1] {"a": [1, 2]} trailing') == {"a": [1, 2]}


def test_router_prompt_shape(home):
    p = rt._prompt("do x", [rt.Candidate("Resume", home, "jobs", [("s1", "AI job hunt", "Nov 9")])], time.time())
    assert p.startswith("Folders:\n- Resume: jobs. Sessions: Nov 9 'AI job hunt' (id s1)") and "Request: do x" in p
