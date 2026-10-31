---
name: pr-ready
description: "Make a branch ready for a PR to main: scope + report-hygiene check, ruff without auto-fix, tests (pytest + ckpt_sync self-test + bash -n), docs row, PR body with the test tally. Usage: /pr-ready [branch]"
---

# pr-ready

## Procedure

1. Scope. `git diff --stat main...HEAD`. A PR carries implementation, tests and docs only. Move out (or drop):
   experiment launch scripts, logs, PDFs, reports, `*.bak*`, scratch / self-test outputs, archives of other repos,
   and personal paths (`/local/aliangdw`, `/datasets/aliangdw`, `/home/...`) in code or docs. Paths become env vars,
   flags or `$HF_LEROBOT_HOME`-style placeholders. Shared S3 prefixes are fine.
   ```bash
   git diff main...HEAD | grep -nE '^\+.*(aliang|/home/|/local/|jobs/)' | cut -c1-160
   ```
2. Lint without fixing, on the changed Python files only:
   ```bash
   uvx ruff check --no-fix $(git diff --name-only --diff-filter=AMR main -- '*.py')
   ```
   `yam/` scripts already carry E701/E702 style findings on main; fix new real findings (unused imports, bugs,
   unused noqa) by hand and do not reformat untouched code.
3. Tests:
   ```bash
   uv run pytest -q --tb=line -p no:warnings <touched test files>   # first
   uv run pytest -q --tb=line -p no:warnings                       # then the suite
   bash yam/sagemaker/test_ckpt_sync.sh
   for f in $(git ls-files '*.sh'); do bash -n $f || echo BAD $f; done
   ```
   Report failures that also fail on main as pre-existing (name them), not as passes.
4. Docs in the same PR: a new config, flag or script gets a row/paragraph in `docs/YAM_PRETRAIN.md` or
   `docs/STAGE2.md`; renames sweep docs, CLAUDE.md and skills.
5. PR body: summary per change, evidence for anything trained (S3 path of the committed `final/<step>/params`,
   W&B run), test tally (counts, not logs). Squash-merge; delete the branch afterwards.

## Receipt (mandatory)

```
branch : <name> @ <sha> vs main <sha>  files <n> (+<a>/-<d>)
scope  : code+tests+docs only | REMOVED: <paths>
lint   : ruff --no-fix: <n> new findings (fixed by hand) | 0
tests  : pytest <p> passed <f> failed (pre-existing: <names>)  ckpt_sync: all ok  bash -n: ok
docs   : <files>
pr     : <url>
```
