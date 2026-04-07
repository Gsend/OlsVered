---
name: git-commit
description: >
  Commit and push OlsveredKFAC repo changes to GitHub. Trigger this skill
  whenever the user says "commit", "push", "save to git", "commit and push",
  "push my changes", or any variation. Handles the Windows index.lock issue
  automatically by running the repo's commit script. Always use this skill
  instead of running raw git commands.
---

# Git Commit & Push — OlsveredKFAC

When the user asks to commit and/or push, follow these steps exactly.

## Step 1 — Check which environment is available

Run this to detect the situation:
```bash
cd /sessions/friendly-gallant-allen/mnt/OlsVered && ls .git/index.lock 2>/dev/null && echo "LOCKED" || echo "NO_LOCK"
```

## Step 2a — If NO_LOCK (VM can access git freely)

Run the Linux commit script directly:
```bash
cd /sessions/friendly-gallant-allen/mnt/OlsVered
bash scripts/commit_and_push.sh "COMMIT_MESSAGE_HERE"
```

If the user provided a commit message, use it. If not, generate a short one
describing what changed (e.g. "Add LR scheduler; fix K-FAC convergence").

## Step 2b — If LOCKED (Windows index.lock is blocking the VM)

Tell the user:

> The git index is locked by Windows. Please run this in PowerShell from your OlsVered folder:
> ```powershell
> powershell -ExecutionPolicy Bypass -File scripts\commit_and_push.ps1
> ```
> Or with a custom message:
> ```powershell
> powershell -ExecutionPolicy Bypass -File scripts\commit_and_push.ps1 -Message "your message here"
> ```
> Once done, let me know and I'll confirm it worked.

## Files the scripts always stage

Both scripts stage these files (skipping any that don't exist yet):
- `benchmark/gpu_benchmark.py`
- `benchmark/training_benchmark.py`
- `benchmark/economic_analysis.py`
- `benchmark/theoretical_analysis.py`
- `benchmark/results/`
- `optimizer/hooks.py`
- `optimizer/olsvered_kfac.py`
- `optimizer/classic_kfac.py`
- `run_benchmark.sh`
- `scripts/`
- `.gitignore`

## After the commit

Confirm success by running:
```bash
cd /sessions/friendly-gallant-allen/mnt/OlsVered && git log --oneline -3
```
Show the user the last 3 commits so they can confirm the push went through.

## If the user wants to add extra files

If they mention files not in the default list, add them explicitly before running
the script:
```bash
git add path/to/extra/file.py
bash scripts/commit_and_push.sh "message"
```
