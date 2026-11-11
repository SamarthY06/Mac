# Security

OpenMac runs an agent with access to your Mac, your browser and your WhatsApp. We take reports
seriously and would rather hear about a problem privately first.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting on the repository: **Security**, then **Report a
vulnerability**, at <https://github.com/SamarthY06/OpenMac/security/advisories/new>. Please do
not open a public issue for a security problem, and do not post it in a pull request.

Include what you did, what happened, and what you expected. A command, a message or a page that
reproduces it is the most useful thing you can send. You should hear back within a week.

## Supported version

The latest release. Fixes land on `main` and ship in the next release; older releases are not
patched.

## Scope

In scope, and worth reporting:

- A way for a tool call to perform an irreversible action (send, post, pay, delete, upload)
  without a spoken or written yes from the owner to a summary of that action.
- A way for content Mac reads (a web page, a document, a group message, a quoted message) to
  act as an instruction that reaches the gate's allow path.
- A way to read anything under the never-readable set: `~/.ssh`, `~/.gnupg`, keychains,
  browser profiles, `~/.aws` and similar credential stores.
- A way for a tool to read or write anything under `~/.mac-agent/` other than Mac's own
  workspace and memory (config, profiles, pairing, the services' frozen copies, the app token).
- A way for someone other than the owner to trigger a turn in a WhatsApp group, or for a
  never-listed chat to be read or written.
- Message text or audio reaching logs, the repo, or any destination other than Anthropic as
  part of the owner's Claude session.

Out of scope, by design:

- **The gate is a rule-based filter, not a sandbox.** It classifies tool calls (parsed shell
  commands, script contents, paths, the frontmost app and the current tab). A model running
  with the owner's own user account can still do what that account can do within the rules.
  Reports that amount to "the model ran an allowed command" are not vulnerabilities.
- **Group members are trusted.** Their messages reach Claude as framed, untrusted context, and
  only the owner's messages are requests, but a group the owner adds is a group the owner
  trusts. Social engineering of the owner through a group is out of scope.
- The owner's own Claude account, Chrome profile and WhatsApp account. OpenMac uses them as
  the owner; it does not add protection they do not already have.
- Denial of service against the owner's own machine (a long-running task, a full disk).

## What the project does to stay safe

The model can only ask. With `brain.confirm_outward = true` (the default) the confirmation token is minted by the orchestrator when the owner
says yes to the summary, is bound to that tool and destination, expires in 120 seconds, and is
checked in a `PreToolUse` hook outside the model's control; the model never sees it. The
never-readable set and the `~/.mac-agent/` rule are checked in the same hook. Independently of that switch, Claude Code's own permission classifier (`permission_mode = "auto"`) runs on every tool call and anything it escalates is put to the owner as a confirmation. With `confirm_outward = false` the owner delegates the send/post/pay decision to the model's judgement; the dangerous set stays enforced. Group
context is capped (50 messages, 8,000 characters) and wrapped as untrusted. Logs carry ids,
counts and timings, never text. Tests under `daemon/tests/test_gate_security_audit.py` encode
the findings of the last internal audit.

Installs on Macs without Go download the prebuilt WhatsApp bridge from the GitHub release of the
same version and check it against the release's `SHA256SUMS`. That is a transfer check, not a
signature: both files come from the same release, so the trust root is the GitHub repository,
the same as for the `git clone` the installer performs. With Go installed the bridge is built
from source instead (`MAC_BRIDGE=build` forces this).
