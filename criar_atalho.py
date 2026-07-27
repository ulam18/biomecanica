"""
Cria o icone do programa e o atalho na Area de Trabalho (Windows).

Executado pelo INSTALAR.bat no fim da instalacao. O atalho aponta para o
pythonw.exe do ambiente isolado (.venv), de modo que o menu abra sem janela
preta de terminal atras.

Nao usa dependencias externas alem do Pillow (que ja vem com o matplotlib):
o atalho .lnk e criado por um script VBScript temporario, evitando exigir
pywin32 so para isso.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(BASE_DIR, "assets")
ICONE = os.path.join(ASSETS, "icone.ico")
NOME_ATALHO = "Analise Mandibular"

# Cores do icone (mesma identidade visual do menu).
AZUL = (31, 78, 121)
BRANCO = (255, 255, 255)


def criar_icone(destino: str = ICONE) -> str:
    """Desenha um icone simples (dente estilizado sobre fundo azul)."""
    from PIL import Image, ImageDraw

    os.makedirs(os.path.dirname(destino), exist_ok=True)
    tamanho = 256
    img = Image.new("RGBA", (tamanho, tamanho), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    d.rounded_rectangle([0, 0, tamanho - 1, tamanho - 1], radius=48, fill=AZUL)

    # Coroa do dente (retangulo de cantos arredondados) e duas raizes.
    d.rounded_rectangle([50, 46, 206, 152], radius=46, fill=BRANCO)
    d.polygon([(64, 142), (114, 142), (100, 216), (80, 216)], fill=BRANCO)
    d.polygon([(142, 142), (192, 142), (176, 216), (156, 216)], fill=BRANCO)

    img.save(destino, format="ICO",
             sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
    return destino


def caminho_pythonw() -> str:
    """pythonw.exe do ambiente isolado; cai para o interpretador atual."""
    venv = os.path.join(BASE_DIR, ".venv", "Scripts", "pythonw.exe")
    return venv if os.path.exists(venv) else sys.executable


def area_de_trabalho() -> str:
    """
    Pasta da Area de Trabalho do usuario.

    Em contas com OneDrive a Area de Trabalho costuma estar redirecionada;
    por isso a pasta do OneDrive e verificada antes do perfil local.
    """
    candidatos = []
    onedrive = os.environ.get("OneDrive") or os.environ.get("OneDriveConsumer")
    if onedrive:
        candidatos += [os.path.join(onedrive, "Desktop"),
                       os.path.join(onedrive, "Area de Trabalho")]
    perfil = os.environ.get("USERPROFILE", os.path.expanduser("~"))
    candidatos += [os.path.join(perfil, "Desktop"),
                   os.path.join(perfil, "Area de Trabalho")]
    for c in candidatos:
        if os.path.isdir(c):
            return c
    return perfil


def criar_atalho_lnk(pasta_destino: str, icone: str | None) -> str:
    """Cria o .lnk via WScript.Shell e retorna o caminho do atalho."""
    destino = os.path.join(pasta_destino, f"{NOME_ATALHO}.lnk")
    alvo = caminho_pythonw()
    script = os.path.join(BASE_DIR, "launcher.py")

    vbs = f'''Set oWS = WScript.CreateObject("WScript.Shell")
Set oLink = oWS.CreateShortcut("{destino}")
oLink.TargetPath = "{alvo}"
oLink.Arguments = """{script}"""
oLink.WorkingDirectory = "{BASE_DIR}"
oLink.Description = "Avaliacao do movimento mandibular por camera"
'''
    if icone and os.path.exists(icone):
        vbs += f'oLink.IconLocation = "{icone}"\n'
    vbs += "oLink.Save\n"

    with tempfile.NamedTemporaryFile("w", suffix=".vbs", delete=False,
                                     encoding="mbcs") as f:
        f.write(vbs)
        caminho_vbs = f.name
    try:
        subprocess.run(["cscript", "//nologo", caminho_vbs], check=True,
                       capture_output=True, text=True)
    finally:
        os.unlink(caminho_vbs)
    return destino


def criar_bat_local() -> str:
    """
    Cria um .bat na pasta do projeto, alternativa ao atalho da Area de
    Trabalho (util se o programa for copiado para um pen drive).
    """
    caminho = os.path.join(BASE_DIR, "Analise Mandibular.bat")
    conteudo = (
        "@echo off\r\n"
        "cd /d \"%~dp0\"\r\n"
        "if exist \".venv\\Scripts\\pythonw.exe\" (\r\n"
        "    start \"\" \".venv\\Scripts\\pythonw.exe\" \"launcher.py\"\r\n"
        ") else (\r\n"
        "    echo O programa ainda nao foi instalado.\r\n"
        "    echo Clique duas vezes em INSTALAR.bat primeiro.\r\n"
        "    pause\r\n"
        ")\r\n"
    )
    with open(caminho, "w", encoding="mbcs", newline="") as f:
        f.write(conteudo)
    return caminho


def main() -> None:
    try:
        icone = criar_icone()
        print(f"Icone criado: {icone}")
    except Exception as exc:  # noqa: BLE001 - icone e opcional
        icone = None
        print(f"Aviso: nao foi possivel criar o icone ({exc}).")

    print(f"Arquivo de abertura: {criar_bat_local()}")

    try:
        atalho = criar_atalho_lnk(area_de_trabalho(), icone)
        print(f"Atalho criado: {atalho}")
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"Erro ao criar o atalho na Area de Trabalho: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
