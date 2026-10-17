"""Redact local machine identifiers from the two PUBLIC repos (rsna-knee-record, arc-agi-3-ledger).

Three kinds, handled differently on purpose:

  * a private-key path      -> `<local-ssh-key>`  (it only ever meant "the key is on this machine");
  * the Windows home path    -> `<user-home>`      (keeps the meaning, drops the account name);
  * the account name in prose-> `<local-user>`     (ditto);
  * the kaggle.exe fallback in scripts/safety_net_watch.ps1 -> $env:USERPROFILE, which keeps the
    script WORKING; a placeholder there would have broken it.

Idempotent: prints what it changed and finishes by reporting any file that still mentions the name.
"""
import io
from pathlib import Path

NAME = '<local-user>'
REPLACEMENTS = [
    # order matters: the longest, most specific forms first
    ('C:\\Users\\%s\\.ssh\\autodl_ed25519' % NAME, '<local-ssh-key>'),
    ('C:/Users/%s/.gitconfig' % NAME, '<user-home>/.gitconfig'),
    ('C:\\Users\\%s\\anaconda3\\Scripts\\kaggle.exe' % NAME, '$env:USERPROFILE\\anaconda3\\Scripts\\kaggle.exe'),
    ('`%s`' % NAME, '`<local-user>`'),
    (NAME, '<local-user>'),
]

REPOS = {
    Path(r'D:\kaggle\rsna-knee'): ['docs/HANDOFF.md', 'state/FACTS.md', 'scripts/safety_net_watch.ps1'],
    Path(r'D:\kaggle\arc-agi-3-ledger'): ['findings/FINDINGS.md', 'STATUS-2026-10-13.md',
                                          'STATUS-2026-10-14.md', 'STATUS-2026-10-15.md'],
}

total = 0
for repo, rels in REPOS.items():
    print('== %s' % repo)
    for rel in rels:
        p = repo / rel
        if not p.is_file():
            print('   %-34s (missing)' % rel)
            continue
        s = p.read_text(encoding='utf-8', newline='')
        changed = 0
        for old, new in REPLACEMENTS:
            n = s.count(old)
            if n:
                s = s.replace(old, new)
                changed += n
                print('   %-34s %-44s -> %-40s x%d' % (rel, old[:44], new[:40], n))
        if changed:
            p.write_text(s, encoding='utf-8', newline='')
        total += changed
print('\ntotal replacements:', total)

# final sweep: walk every tracked-ish text file in both repos for the name
import subprocess
for repo in REPOS:
    r = subprocess.run(['git', '-C', str(repo), 'grep', '-l', NAME], capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    hits = [l for l in (r.stdout or '').splitlines() if l.strip()]
    print('%-32s files still mentioning %r: %s' % (repo.name, NAME, hits or 'none'))
