# scp to/from the rented AutoDL instance, using the same state/autodl.json config as remote.ps1.
#
#   & scripts\rcopy.ps1 -Push D:\kaggle\corpus\packs\rsna-knee-code-20260924.zip -To /root/
#   & scripts\rcopy.ps1 -Push D:\kaggle\corpus\zips -To /root/autodl-tmp/corpus/ -Recurse
#   & scripts\rcopy.ps1 -Pull /root/rsna-knee/out/raptor_oof_x_fold0.npz -To D:\kaggle\rsna-knee\downloads\rent\
param(
    [string]$Push,
    [string]$Pull,
    [string]$To,
    [switch]$Recurse
)
$ErrorActionPreference = 'Continue'
$here = $PSScriptRoot
$cfg = Get-Content (Join-Path (Split-Path -Parent $here) 'state\autodl.json') -Raw | ConvertFrom-Json
$key = $cfg.key -replace '^~', $env:USERPROFILE
$target = "$($cfg.user)@$($cfg.host)"

$common = @('-i', $key, '-P', "$($cfg.port)",
            '-o', 'BatchMode=yes',
            '-o', 'StrictHostKeyChecking=accept-new',
            '-o', 'ServerAliveInterval=30')

if ($Push) {
    if (-not (Test-Path $Push)) { Write-Error "no such local path: $Push"; exit 2 }
    if (-not $To) { Write-Error '-To is required with -Push'; exit 2 }
    $a = $common + @('-r', $Push, "${target}:$To")
    if (-not $Recurse -and (Get-Item $Push).PSIsContainer) {
        Write-Host "(note: $Push is a directory -- -r is being used implicitly)" -ForegroundColor Yellow
    }
    & scp @a
} elseif ($Pull) {
    if (-not $To) { Write-Error '-To is required with -Pull'; exit 2 }
    $parent = Split-Path -Parent $To
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Force $parent | Out-Null }
    $a = $common + @('-r', "${target}:$Pull", $To)
    & scp @a
} else {
    Write-Error 'need -Push or -Pull'; exit 2
}
exit $LASTEXITCODE
