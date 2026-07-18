@echo off
cd /d "%~dp0"
set PY=python
where python >nul 2>nul || set PY=py
echo === 1/2 : Generation des donnees demo Safran + seed base CBR ===
%PY% generate_demo_safran.py --seed-db
echo.
echo === 2/2 : Lancement de LI90 (laisser cette fenetre ouverte) ===
%PY% -m streamlit run landing.py
pause
