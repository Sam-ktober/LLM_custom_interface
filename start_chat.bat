@echo off
title Lanceur IA Locale - AI Studio
echo ==========================================
echo   LANCEMENT DE L'INTERFACE AI STUDIO
echo ==========================================
echo.

:: 1. Initialisation de Conda
:: On utilise le chemin detecte dans tes logs precedents
echo [+] Initialisation de Anaconda...
set CONDA_PATH=C:\Users\sampl\anaconda3\Scripts\activate.bat

if not exist "%CONDA_PATH%" (
    echo [ERREUR] Impossible de trouver Anaconda a l'emplacement prevu.
    echo Verifiez le chemin : %CONDA_PATH%
    pause
    exit
)

:: 2. Activation de l'environnement specifique
echo [+] Activation de l'environnement 'ai_studio_env'...
call "%CONDA_PATH%" ai_studio_env

:: 3. Lancement de l'interface
echo [+] Demarrage de Streamlit...
echo.
streamlit run chat_interface.py

:: Si Streamlit s'arrete, on garde la fenetre ouverte pour voir l'erreur
echo.
echo Interface arretee.
pause