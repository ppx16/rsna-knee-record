@echo off
REM Wrapper so the Windows Task Scheduler can launch the safety-net watcher.
REM schtasks /tr does not accept a command line with embedded quotes reliably,
REM so the quoting lives here instead.
REM
REM   schtasks /create /tn "RSNAKneeSafetyNet" /tr "D:\kaggle\rsna-knee\scripts\safety_net.cmd" /sc ONCE /st 23:59 /f
REM   schtasks /run /tn "RSNAKneeSafetyNet"
REM   schtasks /delete /tn "RSNAKneeSafetyNet" /f      (when the runs have landed)
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\kaggle\rsna-knee\scripts\safety_net_watch.ps1"
