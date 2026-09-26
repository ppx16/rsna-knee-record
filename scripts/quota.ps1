# Read the accelerator quota.
#
# WHY THIS IS A SEPARATE SCRIPT: the repository's primary CLI (kaggle 2.1.2 at D:\python\Scripts)
# has NO `quota` subcommand -- `kaggle quota` exits with "invalid choice: 'quota'" (FACTS A77), which
# is why an earlier note in this repo wrongly concluded the numbers were unreadable.  They ARE
# readable; 2.2.4 exposes them.  Rather than upgrade the working install (2.2.x needs a different
# auth path, FACTS A19), 2.2.4 lives in an isolated venv used only for this.
#
# WHAT IT SETTLED (2026-09-26):
#   * Submission scoring reruns do NOT charge our GPU quota.  Two scoring reruns ran for ~1.7 h
#     while the measured weekly total matched our own runs to within 0.15 h.
#   * It also gives the instrument for the long-open question FACTS 158.5 (whether an N-hour run on
#     a 2xT4 machine is billed as N or 2N): read before and after a timed run and subtract.
#
# Usage:  powershell -File scripts\quota.ps1
$env:KAGGLE_API_TOKEN = [Environment]::GetEnvironmentVariable('KAGGLE_API_TOKEN', 'User')
$env:PYTHONUTF8 = '1'
& D:\kaggle\venv-kaggle\Scripts\kaggle.exe quota
