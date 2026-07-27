@echo off
chcp 65001 >nul
setlocal
title Instalacao - Analise Mandibular
cd /d "%~dp0"

echo.
echo ============================================================
echo    ANALISE MANDIBULAR - INSTALACAO
echo ============================================================
echo.
echo  Este processo prepara o programa no computador.
echo  Leva de 3 a 10 minutos e precisa de conexao com a internet.
echo  Voce so precisa fazer isso UMA VEZ.
echo.
pause

REM ---------------------------------------------------------------
REM  1. Procurar o Python instalado
REM ---------------------------------------------------------------
echo.
echo [1/5] Procurando o Python...
set "PYEXE="

py -3 -c "print(1)" >nul 2>&1
if not errorlevel 1 set "PYEXE=py -3"

if not defined PYEXE (
    python -c "print(1)" >nul 2>&1
    if not errorlevel 1 set "PYEXE=python"
)

if not defined PYEXE goto SEM_PYTHON

%PYEXE% -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if errorlevel 1 goto PYTHON_ANTIGO

echo       Python encontrado.

REM ---------------------------------------------------------------
REM  2. Criar o ambiente isolado do programa
REM ---------------------------------------------------------------
echo.
echo [2/5] Preparando o ambiente do programa...
if exist ".venv\Scripts\python.exe" (
    echo       Ambiente ja existia, reaproveitando.
) else (
    %PYEXE% -m venv .venv
    if errorlevel 1 goto ERRO_VENV
)

REM ---------------------------------------------------------------
REM  3. Instalar as bibliotecas
REM ---------------------------------------------------------------
echo.
echo [3/5] Instalando os componentes (parte mais demorada)...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r requirements.txt --quiet
if errorlevel 1 goto ERRO_PIP

REM ---------------------------------------------------------------
REM  4. Baixar o modelo de reconhecimento facial
REM ---------------------------------------------------------------
echo.
echo [4/5] Baixando o arquivo de reconhecimento facial (4 MB)...
".venv\Scripts\python.exe" download_model.py
if errorlevel 1 goto ERRO_MODELO

REM ---------------------------------------------------------------
REM  5. Criar o atalho na Area de Trabalho
REM ---------------------------------------------------------------
echo.
echo [5/5] Criando o atalho na Area de Trabalho...
".venv\Scripts\python.exe" criar_atalho.py
if errorlevel 1 goto ERRO_ATALHO

echo.
echo ============================================================
echo    PRONTO! Instalacao concluida.
echo ============================================================
echo.
echo  Procure na sua Area de Trabalho o icone:
echo.
echo        Analise Mandibular
echo.
echo  Clique duas vezes nele para abrir o programa.
echo  Nao e mais necessario usar esta janela.
echo.
pause
exit /b 0

REM ---------------------------------------------------------------
REM  Mensagens de erro
REM ---------------------------------------------------------------
:SEM_PYTHON
echo.
echo ------------------------------------------------------------
echo   FALTA INSTALAR O PYTHON
echo ------------------------------------------------------------
echo.
echo  O programa precisa do Python, que e gratuito.
echo.
echo  Vou abrir a pagina de download no seu navegador.
echo.
echo  Na pagina, clique no botao amarelo "Download Python".
echo  Ao abrir o instalador, MARQUE a caixinha
echo.
echo        [X] Add python.exe to PATH
echo.
echo  que aparece na parte de baixo da primeira tela, e clique
echo  em "Install Now".
echo.
echo  Terminada a instalacao do Python, execute este
echo  INSTALAR novamente.
echo.
pause
start "" "https://www.python.org/downloads/"
exit /b 1

:PYTHON_ANTIGO
echo.
echo   A versao do Python instalada e antiga demais.
echo   E necessario o Python 3.10 ou mais novo.
echo.
echo   Vou abrir a pagina de download. Instale a versao mais
echo   recente (marcando "Add python.exe to PATH") e execute
echo   este INSTALAR novamente.
echo.
pause
start "" "https://www.python.org/downloads/"
exit /b 1

:ERRO_VENV
echo.
echo   Nao foi possivel preparar o ambiente do programa.
echo   Tente executar este arquivo novamente. Se o erro
echo   continuar, envie uma foto desta tela ao suporte tecnico.
echo.
pause
exit /b 1

:ERRO_PIP
echo.
echo   Nao foi possivel instalar os componentes.
echo   Verifique a conexao com a internet e execute este
echo   arquivo novamente.
echo.
pause
exit /b 1

:ERRO_MODELO
echo.
echo   Nao foi possivel baixar o arquivo de reconhecimento facial.
echo   Verifique a conexao com a internet e execute este
echo   arquivo novamente.
echo.
pause
exit /b 1

:ERRO_ATALHO
echo.
echo   O programa foi instalado, mas nao consegui criar o atalho
echo   na Area de Trabalho.
echo.
echo   Voce ainda pode abrir o programa clicando duas vezes no
echo   arquivo "Analise Mandibular.bat", nesta mesma pasta.
echo.
pause
exit /b 1
