# Run a command (or a whole bash script) on the rented AutoDL instance over SSH.
#
# WHY A FILE: this project has lost time four separate times to PowerShell eating quotes on
# inline commands (`python -c "..."` inside a double-quoted PS string).  PowerShell 5.1 does
# not even accept \" as an escape.  So put the remote command in a .sh file: it is sent as a
# base64 word and decoded with `base64 -d | bash`, so neither CRLF nor stdin mangling can
# corrupt it (both did, on 2026-09-24 -- see HANDOFF 146.2).
#
#   & scripts\remote.ps1 -File .testtmp\health.sh
#   & scripts\remote.ps1 -Cmd 'nvidia-smi'
#   & scripts\remote.ps1 -Cmd 'ls' -Out .testtmp\ls.txt
#
# Reads state/autodl.json for host/port/key, so a restart only needs that one file updated.
#
# NOTE: use `& .\scripts\remote.ps1 ...`.  This box has NO `pwsh` (only Windows PowerShell 5.1),
# so `pwsh -File ...` fails with CommandNotFoundException.
param(
    [string]$Cmd,
    [string]$File,
    [string]$Out,
    [int]$TimeoutSec = 600,
    [switch]$Quiet
)
$ErrorActionPreference = 'Continue'
$here = $PSScriptRoot
$cfg = Get-Content (Join-Path (Split-Path -Parent $here) 'state\autodl.json') -Raw | ConvertFrom-Json

$key = $cfg.key -replace '^~', $env:USERPROFILE
$target = "$($cfg.user)@$($cfg.host)"

$sshArgs = @('-i', $key, '-p', "$($cfg.port)",
             '-o', 'BatchMode=yes',
             '-o', 'StrictHostKeyChecking=accept-new',
             '-o', 'ServerAliveInterval=30',
             '-o', "ConnectTimeout=20",
             $target)

if ($File) {
    if (-not (Test-Path $File)) { Write-Error "no such script: $File"; exit 2 }
    $body = Get-Content $File -Raw
    # Two Windows transport bugs, both hit on 2026-09-24:
    #  1) a .sh written on Windows has CRLF -> bash dies on the trailing CR (`$'\r': command not found`)
    #  2) piping a string into a native command's stdin makes PowerShell append a platform newline,
    #     which puts a bare CR on the LAST line even after (1) is fixed
    # So do not pipe at all: base64 the (LF-normalised) body and send it as one argv word.
    $body = $body -replace "`r`n", "`n"
    $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($body))
    $sshArgs += "echo $b64 | base64 -d | bash"
    $output = & ssh @sshArgs 2>&1
} elseif ($Cmd) {
    $sshArgs += $Cmd
    $output = & ssh @sshArgs 2>&1
} else {
    Write-Error 'need -Cmd or -File'; exit 2
}
$text = ($output | Out-String)
# ssh's first-connection "Permanently added ... to the list of known hosts" goes to stderr and
# makes PowerShell report a non-zero exit for a perfectly good command.  Filter it out.
$text = ($text -split "`n" | Where-Object { $_ -notmatch 'Permanently added .* to the list of known hosts' }) -join "`n"
if (-not $Quiet) { Write-Output $text }
if ($Out) { $text | Set-Content -Path $Out -Encoding UTF8 }
exit 0
