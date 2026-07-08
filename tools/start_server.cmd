@echo off
REM ATANOR SPLATRA — GPU generation server launcher (auto-start on login).
REM Binds IPv4 0.0.0.0:8000. Point the Cloudflare tunnel ingress at http://127.0.0.1:8000.
cd /d "C:\0.ASKIM ALL-VIN\26.SPLATRA"
set SPLATRA_SD=1
set SPLATRA_MV=1
set SPLATRA_TRIPOSR=1
set SPLATRA_SD_BESTOF=2
set SPLATRA_LOWVRAM=1
set SPLATRA_TRIPOSR_DIR=%USERPROFILE%\.cache\splatra\TripoSR
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
REM Port 8088 (Docker Desktop grabs 8000 on reboot). LOWVRAM=1 streams SDXL
REM submodels CPU<->GPU so we coexist with Docker without thrashing (peak ~7.6GB).
"C:\ProgramData\miniconda3\python.exe" -m uvicorn apps.plugin_api:app --host 0.0.0.0 --port 8088 --log-level warning
