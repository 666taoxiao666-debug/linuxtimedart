---
name: push-linuxtimedart
description: >-
  Commit and push local changes to the linuxtimedart GitHub repository
  (https://github.com/666taoxiao666-debug/linuxtimedart.git). Use when the
  user invokes /push-linuxtimedart, asks to push to linuxtimedart, or wants
  to sync modified code to that remote.
disable-model-invocation: true
---

# Push to linuxtimedart

Push modified code to the target repository:

**Remote URL**: `https://github.com/666taoxiao666-debug/linuxtimedart.git`
**Default remote name**: `origin`
**Default branch**: `main`

## Preconditions

1. Run from the project root (the repo that tracks linuxtimedart).
2. Verify the remote:

```bash
git remote get-url origin
```

Expected output contains `666taoxiao666-debug/linuxtimedart`. If wrong, fix before pushing:

```bash
git remote set-url origin https://github.com/666taoxiao666-debug/linuxtimedart.git
```

## Workflow

Execute these steps in order. Do not skip verification.

### Step 1: Inspect state (parallel)

```bash
git status
git diff
git diff --staged
git log -5 --oneline
```

### Step 2: Decide what to commit

- **No changes and branch is up to date** → report "nothing to push" and stop.
- **Uncommitted changes** → stage and commit (Step 3), then push (Step 4).
- **Committed but not pushed** → push directly (Step 4).

**Do not commit**:
- Secrets (`.env`, credentials, API keys, tokens)
- Large binaries or datasets unless the user explicitly requests them
- `.cursor/` unless the user explicitly asks to include it

Respect existing `.gitignore`.

### Step 3: Commit (only if needed)

Stage relevant files:

```bash
git add <paths>
```

Draft a concise commit message (1–2 sentences, focus on *why*). Match recent `git log` style.

```bash
git commit -m "$(cat <<'EOF'
Your commit message here.

EOF
)"
```

If the commit hook fails, fix the issue and create a **new** commit — never amend a failed commit.

### Step 4: Push

Use the bundled auth script (reads token from `.github-token` in this skill directory). **Never echo or print the token.**

```bash
bash .cursor/skills/push-linuxtimedart/push.sh -u origin main
```

If the current branch is not `main`:

```bash
bash .cursor/skills/push-linuxtimedart/push.sh -u origin HEAD
```

If `git push` fails with auth errors and `.github-token` is missing or invalid, tell the user to update `.cursor/skills/push-linuxtimedart/.github-token` locally — do not ask them to paste tokens into chat.

### Step 5: Verify

```bash
git status
```

Confirm the branch is up to date with `origin/main` (or the pushed branch).

## Safety rules

- NEVER update git config
- NEVER force-push to `main`/`master` unless the user explicitly requests it — warn them if they do
- NEVER use `--no-verify` unless the user explicitly requests it
- NEVER push secrets; warn if staged files look sensitive
- Do not push unless the user invoked this skill or explicitly asked to push

## Authentication

Token is stored locally at:

`.cursor/skills/push-linuxtimedart/.github-token`

This file is gitignored. The agent must use `push.sh` for all pushes — never embed the token in shell commands or commit it.

If push fails with auth errors:

1. Check remote URL is HTTPS (above).
2. Confirm `.github-token` exists and contains a valid GitHub PAT with `repo` scope.
3. Ask the user to regenerate the token on GitHub and update `.github-token` locally if needed.
4. Retry with `push.sh`. Do not ask the user to paste tokens into chat.

## Success output

Report to the user:
- Branch pushed
- Commit hash and message (if a new commit was created)
- Link: https://github.com/666taoxiao666-debug/linuxtimedart

## Examples

**User**: `/push-linuxtimedart`

Agent: inspect → commit if needed → push → confirm with repo link.

**User**: "把改动推到 linuxtimedart"

Same workflow as above.

**User**: "只 push，不要 commit"

Skip Step 3; only push existing commits. If there are uncommitted changes, warn and ask whether to commit first.
