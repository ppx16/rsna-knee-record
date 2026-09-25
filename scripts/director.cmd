@echo off
REM Wrapper so the Windows Task Scheduler can run one director pass on a timer.
REM `schtasks /tr` does not take a command line with embedded quotes reliably,
REM so the quoting lives here instead.
REM
REM   schtasks /create /tn "RSNAKneeDirector" /tr "D:\kaggle\rsna-knee\scripts\director.cmd" /sc MINUTE /mo 15 /f
REM   schtasks /run    /tn "RSNAKneeDirector"
REM   schtasks /query  /tn "RSNAKneeDirector" /v /fo LIST
REM   schtasks /delete /tn "RSNAKneeDirector" /f        (when the runs have landed)
REM
REM A pass takes seconds when every kernel is still running and runs the
REM download/submit/score chain when one has finished. It self-locks, so it is
REM safe to overlap with a `director.py --loop` background session.
REM Read the token OUT OF THE FILE instead of trusting the environment.
REM   READ paths (kernels.get, competitions.submissions) need KAGGLE_API_TOKEN in the
REM   ENVIRONMENT; WRITE paths (kernels.push) use ~/.kaggle/access_token. `setx` only
REM   affects NEW logons, but Task Scheduler inherits the env of the CURRENT session -- so
REM   after the token was rotated, every scheduled read path silently used the OLD token and
REM   failed with "Permission 'kernels.get' was denied" / 403 on the submissions endpoint.
REM   Reading the file makes both paths agree no matter when the token last changed.
set /p KAGGLE_API_TOKEN=<"%USERPROFILE%\.kaggle\access_token"
set PYTHONUTF8=1
REM Capture the pass output. queue.cmd has always redirected ("so a failure is diagnosable
REM rather than silent"); the director did NOT, so every error it hit vanished into the Task
REM Scheduler's void -- which is exactly why a 403 on the submissions endpoint had to be
REM pasted in by hand instead of being found in a log. Same reason, same fix.
cd /d "D:\kaggle\rsna-knee"
"D:\python\python.exe" "D:\kaggle\rsna-knee\scripts\director.py" --once >> "D:\kaggle\rsna-knee\state\director_task.log" 2>&1
