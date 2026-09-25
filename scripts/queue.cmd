@echo off
REM Wrapper so the Windows Task Scheduler can run one QUEUE pass on a timer.
REM Mirrors director.cmd: `schtasks /tr` does not take a command line with embedded
REM quotes reliably, so the quoting lives here instead.
REM
REM WHY THIS EXISTS (2026-09-14)
REM   The director had a scheduled task but the QUEUE did not, so after a reboot the GPU
REM   slots stayed IDLE until a human or agent happened to run `cv_queue.py --once`.  The
REM   whole point of the queue is to keep both slots full, so it has to run unattended.
REM
REM   schtasks /create /tn "RSNAKneeQueue" /tr "D:\kaggle\rsna-knee\scripts\queue.cmd" /sc MINUTE /mo 20 /f
REM   schtasks /run    /tn "RSNAKneeQueue"
REM   schtasks /query  /tn "RSNAKneeQueue" /v /fo LIST
REM   schtasks /delete /tn "RSNAKneeQueue" /f          (when the experiments are done)
REM
REM Safe to overlap: cv_queue.py is idempotent (never re-pushes something it already pushed or
REM that is already running), stops the pass on "GPU full", retries transient credential/network
REM failures instead of permanently holding a target, and only registers ENSEMBLE variants with
REM the director (training folds produce no submission.csv by design).
REM
REM Requires KAGGLE_API_TOKEN in the environment (set at User scope on 2026-09-14) and the
REM token file ~/.kaggle/access_token for the push path.  Its output goes to
REM state\queue_task.log so a failure is diagnosable rather than silent.
REM Read the token OUT OF THE FILE rather than trusting the environment -- see director.cmd.
REM   A stale session env after a token rotation makes every READ path 403 while pushes keep
REM   working off this same file, which looks like a permissions bug and is not one.
set /p KAGGLE_API_TOKEN=<"%USERPROFILE%\.kaggle\access_token"
set PYTHONUTF8=1
cd /d "D:\kaggle\rsna-knee"
"D:\python\python.exe" "D:\kaggle\rsna-knee\scripts\cv_queue.py" --once >> "D:\kaggle\rsna-knee\state\queue_task.log" 2>&1
