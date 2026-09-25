# Keep this machine awake while a long transfer runs.
#
# WHY: the dataset upload runs ON THIS MACHINE.  Sleep/hibernate suspends the process, which stalls
# the upload -- and across a process restart the CLI's resumable-upload session may not survive (the
# dataset is not finalised yet, so `kaggle datasets list` cannot even see it).  So an unattended
# "screen goes black and it sleeps" is a real way to lose the transfer.
#
# SetThreadExecutionState with ES_CONTINUOUS|ES_SYSTEM_REQUIRED tells Windows "do not go to automatic
# sleep while this thread lives".  ES_DISPLAY_REQUIRED is deliberately NOT set: the monitor is allowed
# to blank, which is what the user expects -- we only need the SYSTEM to stay up.
#
#   powershell -File scripts\keep_awake.ps1 -MaxMinutes 120
param([int]$MaxMinutes = 120)
$ErrorActionPreference = 'Continue'

Add-Type -Namespace KeepAwake -Name Native -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@

# ⚠️ `[uint32]0x80000000` THROWS in PowerShell: the hex literal is parsed as Int32 -2147483648 and the
# cast fails, leaving $flags empty -- so the API call would pass 0 and CLEAR the inhibition instead of
# setting it.  Use the decimal literal 2147483648 (parsed as Int64, so the cast succeeds).
$ES_CONTINUOUS = [uint32]2147483648
$ES_SYSTEM_REQUIRED = [uint32]1
$flags = $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED
if ($flags -ne [uint32]2147483649) {
    throw "bad flags: $flags (expected 2147483649 = ES_CONTINUOUS|ES_SYSTEM_REQUIRED)"
}

# also push the automatic timeouts out, in case a policy overrides the API (works per-user; may need
# admin, so failures are reported rather than fatal)
foreach ($kv in @(@('standby-timeout-ac', 0), @('hibernate-timeout-ac', 0))) {
    $r = & powercfg /change $kv[0] $kv[1] 2>&1
    if ($LASTEXITCODE -ne 0) { Write-Host "powercfg $($kv[0]) failed (not fatal): $r" }
    else { Write-Host "powercfg $($kv[0]) = $($kv[1])" }
}

$deadline = (Get-Date).AddMinutes($MaxMinutes)
Write-Host "$(Get-Date -Format HH:mm:ss) keep-awake START (system-required, display may blank)"
$n = 0
while ((Get-Date) -lt $deadline) {
    $rc = [KeepAwake.Native]::SetThreadExecutionState($flags)
    $n++
    if ($n % 12 -eq 1) {
        Write-Host ("{0} tick {1} rc={2} (system sleep inhibited)" -f (Get-Date -Format 'HH:mm:ss'), $n, $rc)
    }
    Start-Sleep -Seconds 45
}
Write-Host "$(Get-Date -Format HH:mm:ss) keep-awake END (window elapsed)"
Write-Host "KEEPAWAKE-DONE"
