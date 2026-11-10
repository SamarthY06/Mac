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
    for text in ("find jobs in Resume", "find jobs in the Resume folder please", "find jobs in my resume project",
                 "summarize ~/Desktop/Personal/Resume"):
        out = await r.route(text, folder=None, last_answer_at=0, fresh=False, named=None)
        assert out.folder == home / "Desktop/Personal/Resume" and out.session == "new", text
    assert model.prompts == []


async def test_name_inside_prose_is_only_a_hint_for_the_model(home):
    """AG2: "a cover letter in resume style" is not a folder choice; the model decides."""
    r, model, _ = make(home, '{"folder": "general", "session": "new", "confidence": 0.9}')
    out = await r.route("write a cover letter in resume style", folder=None, last_answer_at=0, fresh=False,
                        named=None)
    assert out == rt.Route(None, "new")
    assert len(model.prompts) == 1 and "Hint: the request mentions 'Resume'" in model.prompts[0]


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
    r, _, _ = make(home, '{"folder": "Blog", "session": "new", "confidence": 0.9}',
                   mdfind=f"{home}/Desktop/Personal/Blog\n{home}/Old/Blog\n")
    out = await r.route("edit the post in the Blog folder", folder=None, last_answer_at=0, fresh=False, named=None)
    assert out.question.startswith("Which Blog do you mean")
    assert [o.label for o in out.options] == ["Personal/Blog", "Old/Blog"]


async def test_model_pick_above_threshold_continue_session(home):
    s1 = sess(home / "Desktop/Personal/Resume", "s-r1", "AI job hunt")
    r, model, _ = make(home, 'Sure. {"folder": "x"} then {"folder": "Resume", "session": "continue:s-r1", '
                             '"confidence": 0.75}', sessions=[s1])
    out = await r.route("20 more jobs like last time", folder=None, last_answer_at=0, fresh=False, named=None)
    assert out == rt.Route(home / "Desktop/Personal/Resume", "continue:s-r1")
    assert "s-r1" in model.prompts[0] and "Job search and resumes" in model.prompts[0]


def test_threshold_is_pinned():
    assert rt.CONFIDENCE == 0.75  # AG8, as in the doc


async def test_low_confidence_asks_a_route_question(home):
    r, _, _ = make(home, '{"folder": "Resume", "session": "new", "confidence": 0.5}')
    out = await r.route("tidy things up", folder=None, last_answer_at=0, fresh=False, named=None)
    assert out.question == "Should I do this in Resume? Say no to do it generally."
    assert [o.folder for o in out.options] == [home / "Desktop/Personal/Resume", None]


async def test_low_confidence_continue_asks_continue_or_fresh(home):
    s1 = sess(home / "Desktop/Personal/Resume", "s-r1", "the job search")
    r, _, _ = make(home, '{"folder": "Resume", "session": "continue:s-r1", "confidence": 0.4}', sessions=[s1])
    out = await r.route("jobs", folder=None, last_answer_at=0, fresh=False, named=None)
    assert out.question.startswith('Continue "the job search" from ') and out.question.endswith("? Say no to start fresh.")


@pytest.mark.parametrize("reply", ["no json here", '{"folder": "Bad*", "session": "new", "confidence": 1}',
                                   '{"folder": "Resume", "session": "continue:fake", "confidence": 1}',
                                   '{"folder": "Resume", "session": "maybe", "confidence": 1}',
                                   RuntimeError("down")])
async def test_bad_model_reply_falls_back(home, reply):
    r, _, _ = make(home, reply)
    out = await r.route("hm", folder=None, last_answer_at=0, fresh=False, named=None)
    assert out == rt.Route(None, "new")
    out = await r.route("hm", folder=home / "Desktop/Personal/Resume", last_answer_at=0, fresh=False, named=None)
    assert out.question == "Should I do this in Resume? Say no to do it generally."


async def test_candidates_recent_folders_filtered(home):
    sessions = [sess(home / "Code/Mac", "m1", "agent"), sess(home / "Library/Mail", "x", "no"),
                sess(Path("/etc"), "y", "no"), sess(home / "Gone", "z", "no"),
                sess(home / "Desktop/Personal/Blog", "old", "too old", days_ago=40)]
    r, _, _ = make(home, sessions=sessions)
    names = [c.name for c in await r.candidates()]
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


async def test_model_names_a_folder_outside_the_candidates(home):
    r, model, calls = make(home, '{"folder": "Mac", "session": "new", "confidence": 0.9}', mdfind=f"{home}/Code/Mac\n")
    out = await r.route("fix the bug in my voice agent code", folder=None, last_answer_at=0, fresh=False, named=None)
    assert out == rt.Route(home / "Code/Mac", "new") and model.prompts and calls  # model chose, Spotlight resolved


async def test_model_named_folder_not_found_goes_general(home):
    r, _, _ = make(home, '{"folder": "Nowhere", "session": "new", "confidence": 0.9}')
    out = await r.route("x", folder=None, last_answer_at=0, fresh=False, named=None)
    assert out == rt.Route(None, "new")


async def test_model_words_the_route_question(home):
    r, _, _ = make(home, '{"folder": "Resume", "session": "new", "confidence": 0.4, '
                         '"ask": "Is this \u2014 for your job search?", "yes": "Resume"}')
    out = await r.route("update it", folder=None, last_answer_at=0, fresh=False, named=None)
    assert out.question == "Is this, for your job search?" and "\u2014" not in out.question
    assert [o.folder for o in out.options] == [home / "Desktop/Personal/Resume", None] and out.yes == 0


async def test_model_question_yes_means_what_the_model_says(home):
    """AG3: "Should I start fresh, or pick up the job search?" with yes = fresh binds yes to fresh."""
    sessions = [SimpleNamespace(cwd=str(home / "Desktop/Personal/Resume"), last_modified=NOW * 1000,
                                session_id="s1", custom_title="the job search", summary="")]
    r, _, _ = make(home, '{"folder": "Resume", "session": "continue:s1", "confidence": 0.4, '
                         '"ask": "Should I start fresh, or pick up the job search?", "yes": "fresh"}', sessions)
    out = await r.route("more jobs", folder=None, last_answer_at=0, fresh=False, named=None)
    assert [o.label for o in out.options] == ["continue", "fresh"] and out.yes == 1
    assert out.question == "Should I start fresh, or pick up the job search?"
    r, _, _ = make(home, '{"folder": "Resume", "session": "continue:s1", "confidence": 0.4, '
                         '"ask": "Fresh or the job search?"}', sessions)
    out = await r.route("more jobs", folder=None, last_answer_at=0, fresh=False, named=None)
    assert out.yes is None and out.question == "Fresh or the job search? (1 continue, 2 fresh)"


async def test_answer_to_a_route_question_goes_back_to_the_model(home):
    """AG4: the user's own words decide; the model is told the question and the answer."""
    r, model, _ = make(home, '{"folder": "Resume", "session": "new", "confidence": 0.3}')
    out = await r.route("update it", folder=None, last_answer_at=NOW - 60, fresh=False, named=None,
                        answer=("Is this for your job search?", "the job one"))
    assert out == rt.Route(home / "Desktop/Personal/Resume", "new")  # never asks twice
    assert "They answered: 'the job one'" in model.prompts[0]
    r, _, _ = make(home, "no json")
    out = await r.route("x", folder=None, last_answer_at=0, fresh=False, named=None, answer=("?", "hmm"))
    assert out == rt.Route(None, "new")


async def test_unsure_general_asks_about_the_current_folder(home):
    """AG9: a low-confidence general with an ask offers [current folder, general]."""
    cur = home / "Desktop/Personal/Resume"
    r, _, _ = make(home, '{"folder": "general", "session": "new", "confidence": 0.4, '
                         '"ask": "Is this about your CV?", "yes": "Resume"}')
    out = await r.route("tweak it", folder=cur, last_answer_at=0, fresh=False, named=None)
    assert out.question == "Is this about your CV?" and [o.folder for o in out.options] == [cur, None]
    assert out.yes == 0


async def test_no_keyword_intent_rules_without_the_model(home):
    """Agentic: "in the X folder" is not parsed by a rule; only an exact known name is a fast path."""
    r, model, calls = make(home, '{"folder": "general", "session": "new", "confidence": 0.9}',
                           mdfind=f"{home}/Code/Mac\n")
    out = await r.route("edit things in the Mac folder", folder=None, last_answer_at=0, fresh=False, named=None)
    assert out == rt.Route(None, "new") and model.prompts and calls == []
