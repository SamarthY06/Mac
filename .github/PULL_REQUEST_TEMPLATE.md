## What

One or two sentences: what changes for the user, or for the code.

## Why

The problem, the issue number if there is one, and why this is the right fix.

## Tests run

- [ ] `cd daemon && uv run ruff check src tests`
- [ ] `cd daemon && uv run pytest -q -m "not live and not models and not perf"`
- [ ] `cd app && swift test` (if `app/` changed)
- [ ] `go vet ./... && go build ./...` in `bridges/whatsapp` (if the bridge changed; this also needs a `VERSION` bump and a release, see CONTRIBUTING.md)
- [ ] Tried it end to end (say what you said to Mac and what it did)

## Checks

- [ ] No message text, group names, phone numbers or secrets in code, tests, fixtures or logs.
- [ ] User-facing text (spoken lines, HUD, CLI output, errors, docs) is plain and short, American spelling, no em dashes.
- [ ] The gate is not widened. If a legitimate action was blocked, the fix narrows the rule and a test shows both the old block and the new allow.
- [ ] Docs updated where behaviour changed (README, docs/USAGE.md, CHANGELOG.md).
