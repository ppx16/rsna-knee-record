# Safety-net watcher for the two RSNA Knee Kaggle runs.
#
# Why this exists
# ---------------
# The primary watchers (scripts/watch_train.sh, scripts/watch_v8.sh) were started
# by an earlier session and are still running as orphans. They are probably fine,
# but they share one failure mode: they `echo` on every status *change*, and the
# terminal transition is a status change. If the stdout pipe of an orphaned
# watcher was closed when its session died, that final echo raises SIGPIPE and
# the watcher dies before it ever calls `kaggle kernels output`. Kaggle only ever
# serves the output of the LATEST version of a kernel, so a missed download is
# permanent - eight hours of T4 quota gone with no way to fetch it back.
#
# This script is the insurance. It is deliberately independent:
#   * it lands output in outputs/_landing/<name>_<stamp>/, never in the primary
#     directories, so the two can never interleave a partial extraction;
#   * it writes nothing to stdout, so it cannot die the way described above;
#   * it treats "downloaded but zero files" as failure and retries, because
#     `kaggle kernels output` during RUNNING returns an empty directory with
#     exit code 0 and no error.
#
# Launch detached so it outlives any agent session:
#   Start-Process pwsh -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','D:\kaggle\rsna-knee\scripts\safety_net_watch.ps1' -WindowStyle Hidden
#
# Watch it with:  Get-Content D:\kaggle\rsna-knee\outputs\_landing\safety_net.log -Tail 20

param(
    # Defaults are the real run. Override them to rehearse the DOWNLOAD branch against an
    # already-finished kernel, which is otherwise untestable without waiting hours:
    #   powershell -File safety_net_watch.ps1 -OutRoot outputs\_landing_selftest `
    #       -PollSeconds 5 -KernelsSpec "smoke=ppxl16/rsna-knee-tpu-smoke"
    [string]$OutRoot = "outputs\_landing",
    [int]$PollSeconds = 180,
    [string]$KernelsSpec = ""
)

$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"          # kaggle CLI 2.1.2 mis-decodes UTF-8 notebooks on zh-CN Windows

$Root = "D:\kaggle\rsna-knee"
$Land = Join-Path $Root $OutRoot
New-Item -ItemType Directory -Force -Path $Land | Out-Null
$Log = Join-Path $Land "safety_net.log"

function Say([string]$m) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
    Add-Content -Path $Log -Value $line -Encoding UTF8
}

$Kaggle = (Get-Command kaggle -ErrorAction SilentlyContinue).Source
if (-not $Kaggle) {
    foreach ($c in @("$env:APPDATA\Python\Python312\Scripts\kaggle.exe",
                     "$env:LOCALAPPDATA\Programs\Python\Python312\Scripts\kaggle.exe",
                     "$env:USERPROFILE\anaconda3\Scripts\kaggle.exe")) {
        if (Test-Path $c) { $Kaggle = $c; break }
    }
}
if (-not $Kaggle) {
    Say "FATAL: cannot resolve the kaggle CLI; aborting"
    exit 1
}

# name -> kernel ref. The name is only used to label the landing directory.
$Kernels = [ordered]@{}
if ($KernelsSpec) {
    foreach ($pair in $KernelsSpec.Split(";")) {
        $kv = $pair.Split("=", 2)
        if ($kv.Count -eq 2 -and $kv[0].Trim() -and $kv[1].Trim()) {
            $Kernels[$kv[0].Trim()] = $kv[1].Trim()
        }
    }
} else {
    $Kernels["gpu_v4_224x12"] = "ppxl16/rsna-knee-train-v1"
    $Kernels["tpu_v19"]       = "ppxl16/rsna-knee-tpu-train-v1"
}
if ($Kernels.Count -eq 0) {
    Say "FATAL: no kernels to watch (KernelsSpec='$KernelsSpec')"
    exit 1
}

Say "safety net start (pid $PID); kaggle=$Kaggle; poll=${PollSeconds}s; landing=$Land; watching $($Kernels.Count) kernel(s)"

$pending = @{}
foreach ($k in $Kernels.Keys) { $pending[$k] = $Kernels[$k] }

$prev = @{}
while ($pending.Count -gt 0) {
    foreach ($name in @($pending.Keys)) {
        $ref = $pending[$name]
        $raw = (& $Kaggle kernels status $ref 2>&1 | Out-String)
        $s = "UNKNOWN"
        if ($raw -match "KernelWorkerStatus\.([A-Z]+)") { $s = $Matches[1] }
        if ($prev[$ref] -ne $s) { Say "$ref status=$s"; $prev[$ref] = $s }

        if ($s -eq "COMPLETE" -or $s -eq "ERROR" -or $s -eq "CANCELLED") {
            # Relative path from $Root: keeps the CLI on a flat, relative -p argument.
            $rel  = "{0}\{1}_{2}" -f $OutRoot, $name, (Get-Date -Format "yyyyMMdd_HHmmss")
            $dest = Join-Path $Root $rel
            New-Item -ItemType Directory -Force -Path $dest | Out-Null
            Say "$ref TERMINAL ($s) - downloading to $rel"
            Start-Sleep -Seconds 30          # let Kaggle finish writing the archive

            $ok = $false
            for ($try = 1; $try -le 5; $try++) {
                Push-Location $Root
                $null = (& $Kaggle kernels output $ref -p $rel 2>&1 | Out-String)
                Pop-Location
                $files = @(Get-ChildItem -Path $dest -Recurse -File -ErrorAction SilentlyContinue)
                Say "  attempt $try of 5 : files=$($files.Count)"
                if ($files.Count -gt 0) { $ok = $true; break }
                Start-Sleep -Seconds 60
            }

            if ($ok) {
                Say "  LANDED $($files.Count) file(s):"
                foreach ($f in $files) {
                    Say ("    {0,10}  {1}" -f $f.Length, $f.Name)
                }
            } else {
                Say "  FAILED: 5 attempts produced zero files - check the primary watcher's dir"
            }
            $pending.Remove($name)
        }
    }
    if ($pending.Count -gt 0) {
        # Heartbeat every poll. Without it the log only moves on a status change,
        # so a dead safety net would look exactly like a quiet one - which is the
        # failure this whole script exists to guard against.
        $hb = @()
        foreach ($k in $Kernels.Keys) { $hb += "$k=$($prev[$Kernels[$k]])" }
        Say ("  poll ok: " + ($hb -join " "))
        Start-Sleep -Seconds $PollSeconds
    }
}

Say "safety net done: every watched kernel reached a terminal state"
