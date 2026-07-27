"""
Menu principal do sistema (janela com botoes) -- ponto de entrada para uso
clinico, sem linha de comando.

Este arquivo NAO contem logica de medicao: ele apenas monta a interface e
chama `run.py` (avaliacao ao vivo) e `analyze_video.py` (analise de video
gravado) como processos separados, com os parametros ja preenchidos. Assim a
mesma logica atende tanto quem usa o terminal quanto quem usa so o mouse.

Executado pelo atalho "Analise Mandibular" criado por INSTALAR.bat.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTADOS = os.path.join(BASE_DIR, "resultados")
MODELO = os.path.join(BASE_DIR, "models", "face_landmarker.task")

COR_FUNDO = "#f4f5f7"
COR_TITULO = "#1f4e79"
COR_PRIMARIO = "#1b7f3b"
COR_TEXTO = "#222222"

# Distancia media entre os cantos externos dos olhos em adultos, usada como
# valor inicial do campo. E apenas um ponto de partida: o profissional deve
# medir o paciente com regua/paquimetro para a leitura em milimetros valer.
REF_MM_PADRAO = "63"


def python_executavel() -> str:
    """
    Interpretador usado para os subprocessos.

    O launcher costuma ser aberto por pythonw.exe (sem console). Para os
    subprocessos preferimos python.exe, para que erros apareçam no log.
    """
    pasta, arquivo = os.path.split(sys.executable)
    if arquivo.lower().startswith("pythonw"):
        alternativo = os.path.join(pasta, arquivo.lower().replace("pythonw", "python", 1))
        if os.path.exists(alternativo):
            return alternativo
    return sys.executable


class Launcher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Análise Mandibular")
        self.configure(bg=COR_FUNDO)
        self.resizable(False, False)
        self._processo: subprocess.Popen | None = None

        icone = os.path.join(BASE_DIR, "assets", "icone.ico")
        if os.path.exists(icone):
            try:
                self.iconbitmap(icone)
            except tk.TclError:
                pass  # icone invalido nao deve impedir o uso do programa

        self._montar()
        self.after(300, self._verificar_instalacao)

    # -- Interface ------------------------------------------------------------
    def _montar(self) -> None:
        cab = tk.Frame(self, bg=COR_TITULO, padx=24, pady=18)
        cab.pack(fill="x")
        tk.Label(cab, text="Análise Mandibular", font=("Segoe UI", 20, "bold"),
                 bg=COR_TITULO, fg="white").pack(anchor="w")
        tk.Label(cab, text="Avaliação do movimento da mandíbula por câmera",
                 font=("Segoe UI", 10), bg=COR_TITULO, fg="#cfe0f0").pack(anchor="w")

        corpo = tk.Frame(self, bg=COR_FUNDO, padx=24, pady=18)
        corpo.pack(fill="both", expand=True)

        # -- Dados do paciente
        tk.Label(corpo, text="Nome do paciente (opcional)", font=("Segoe UI", 10),
                 bg=COR_FUNDO, fg=COR_TEXTO).pack(anchor="w")
        self.var_paciente = tk.StringVar()
        tk.Entry(corpo, textvariable=self.var_paciente, font=("Segoe UI", 12),
                 width=40).pack(anchor="w", pady=(2, 14), ipady=4)

        # -- Medida em milimetros
        self.var_mm = tk.BooleanVar(value=True)
        linha_mm = tk.Frame(corpo, bg=COR_FUNDO)
        linha_mm.pack(anchor="w", fill="x")
        tk.Checkbutton(linha_mm, text="Mostrar as medidas em milímetros",
                       variable=self.var_mm, bg=COR_FUNDO, font=("Segoe UI", 10),
                       command=self._alternar_mm).pack(side="left")
        self.var_ref = tk.StringVar(value=REF_MM_PADRAO)
        self.campo_ref = tk.Entry(linha_mm, textvariable=self.var_ref, width=6,
                                  font=("Segoe UI", 11), justify="center")
        self.campo_ref.pack(side="left", padx=(8, 4))
        tk.Label(linha_mm, text="mm", bg=COR_FUNDO, font=("Segoe UI", 10)).pack(side="left")
        tk.Label(corpo, text="Meça com régua a distância entre os cantos externos "
                             "dos olhos\ndo paciente e digite o valor acima.",
                 font=("Segoe UI", 9), bg=COR_FUNDO, fg="#666666",
                 justify="left").pack(anchor="w", pady=(2, 14))

        # -- Camera
        linha_cam = tk.Frame(corpo, bg=COR_FUNDO)
        linha_cam.pack(anchor="w", pady=(0, 16))
        tk.Label(linha_cam, text="Câmera:", bg=COR_FUNDO,
                 font=("Segoe UI", 10)).pack(side="left")
        self.var_cam = tk.StringVar(value="Câmera 1")
        ttk.Combobox(linha_cam, textvariable=self.var_cam, state="readonly", width=12,
                     values=["Câmera 1", "Câmera 2", "Câmera 3"],
                     font=("Segoe UI", 10)).pack(side="left", padx=8)
        tk.Label(linha_cam, text="(troque se a imagem não aparecer)", bg=COR_FUNDO,
                 font=("Segoe UI", 9), fg="#666666").pack(side="left")

        # -- Botao principal
        tk.Button(corpo, text="INICIAR AVALIAÇÃO", command=self.iniciar_avaliacao,
                  font=("Segoe UI", 15, "bold"), bg=COR_PRIMARIO, fg="white",
                  activebackground="#166030", activeforeground="white",
                  relief="flat", cursor="hand2", height=2).pack(fill="x", pady=(0, 14))

        # -- Botoes secundarios
        for texto, comando in [
            ("Analisar um vídeo já gravado", self.analisar_video),
            ("Abrir a pasta dos resultados", self.abrir_resultados),
            ("Ver a evolução do paciente", self.ver_evolucao),
            ("Como usar (passo a passo)", self.mostrar_ajuda),
        ]:
            tk.Button(corpo, text=texto, command=comando, font=("Segoe UI", 11),
                      bg="white", fg=COR_TITULO, relief="solid", bd=1,
                      cursor="hand2", height=2).pack(fill="x", pady=3)

        self.var_status = tk.StringVar(value="Pronto para iniciar.")
        tk.Label(self, textvariable=self.var_status, bg="#e8eaed", fg="#444444",
                 font=("Segoe UI", 9), anchor="w", padx=24,
                 pady=6).pack(fill="x", side="bottom")

        self._alternar_mm()

    def _alternar_mm(self) -> None:
        self.campo_ref.configure(state="normal" if self.var_mm.get() else "disabled")

    def _status(self, texto: str) -> None:
        self.var_status.set(texto)
        self.update_idletasks()

    # -- Verificacoes ---------------------------------------------------------
    def _verificar_instalacao(self) -> None:
        if os.path.exists(MODELO):
            return
        baixar = messagebox.askyesno(
            "Primeiro uso",
            "Falta baixar o arquivo de reconhecimento facial (cerca de 4 MB).\n\n"
            "Deseja baixar agora? É necessário estar conectado à internet.",
        )
        if not baixar:
            self._status("Programa incompleto: falta o arquivo de reconhecimento facial.")
            return
        self._status("Baixando o arquivo de reconhecimento facial...")
        try:
            subprocess.run([python_executavel(), "download_model.py"], cwd=BASE_DIR,
                           check=True, capture_output=True, text=True)
            self._status("Arquivo baixado. Pronto para iniciar.")
        except subprocess.CalledProcessError as exc:
            self._erro("Não foi possível baixar o arquivo de reconhecimento facial.",
                       exc.stderr or str(exc))

    def _ref_mm(self) -> str | None:
        """Valida o campo de milimetros; retorna None se a opcao estiver desligada."""
        if not self.var_mm.get():
            return None
        texto = self.var_ref.get().strip().replace(",", ".")
        try:
            valor = float(texto)
        except ValueError:
            messagebox.showwarning(
                "Medida inválida",
                "Digite a distância entre os cantos externos dos olhos em "
                "milímetros (por exemplo: 63).",
            )
            return "invalido"
        if not 30.0 <= valor <= 120.0:
            messagebox.showwarning(
                "Medida fora do esperado",
                f"O valor {valor:g} mm está fora da faixa esperada (30 a 120 mm).\n"
                "Confira a medida entre os cantos externos dos olhos.",
            )
            return "invalido"
        return f"{valor:g}"

    def _erro(self, mensagem: str, detalhe: str = "") -> None:
        texto = mensagem
        if detalhe:
            texto += "\n\nDetalhe técnico:\n" + detalhe.strip()[-800:]
        messagebox.showerror("Erro", texto)
        self._status(mensagem)

    # -- Acoes ----------------------------------------------------------------
    def iniciar_avaliacao(self) -> None:
        ref = self._ref_mm()
        if ref == "invalido":
            return
        if not os.path.exists(MODELO):
            self._verificar_instalacao()
            if not os.path.exists(MODELO):
                return

        cam = self.var_cam.get().split()[-1]
        args = [python_executavel(), "run.py", "--simples",
                "--camera", str(int(cam) - 1),
                "--width", "960", "--height", "540"]
        if ref:
            args += ["--ref-mm", ref]
        if self.var_paciente.get().strip():
            args += ["--paciente", self.var_paciente.get().strip()]

        self._status("Avaliação em andamento. Feche a janela da câmera para voltar.")
        self.withdraw()
        threading.Thread(target=self._rodar_avaliacao, args=(args,), daemon=True).start()

    def _rodar_avaliacao(self, args: list[str]) -> None:
        try:
            proc = subprocess.run(args, cwd=BASE_DIR, capture_output=True, text=True)
        except OSError as exc:
            self.after(0, self._fim_avaliacao, 1, str(exc))
            return
        self.after(0, self._fim_avaliacao, proc.returncode,
                   (proc.stderr or "") + (proc.stdout or ""))

    def _fim_avaliacao(self, codigo: int, saida: str) -> None:
        self.deiconify()
        if codigo != 0:
            dica = ""
            if "camera" in saida.lower():
                dica = ("\n\nDica: verifique se a webcam está conectada e se nenhum "
                        "outro programa a está usando. Se houver mais de uma câmera, "
                        "tente 'Câmera 2' no menu.")
            self._erro("A avaliação terminou com erro." + dica, saida)
        else:
            self._status("Avaliação encerrada. Os arquivos estão na pasta de resultados.")

    def analisar_video(self) -> None:
        ref = self._ref_mm()
        if ref == "invalido":
            return
        caminho = filedialog.askopenfilename(
            title="Escolha o vídeo do paciente",
            filetypes=[("Vídeos", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                       ("Todos os arquivos", "*.*")],
        )
        if not caminho:
            return
        args = [python_executavel(), "analyze_video.py", caminho, "--abrir-relatorio"]
        if ref:
            args += ["--ref-mm", ref]
        if self.var_paciente.get().strip():
            args += ["--paciente", self.var_paciente.get().strip()]

        self._status("Analisando o vídeo... isso pode levar alguns minutos.")
        threading.Thread(target=self._rodar_video, args=(args,), daemon=True).start()

    def _rodar_video(self, args: list[str]) -> None:
        proc = subprocess.run(args, cwd=BASE_DIR, capture_output=True, text=True)
        if proc.returncode == 0:
            self.after(0, self._status, "Análise concluída. O relatório em PDF foi aberto.")
        else:
            self.after(0, self._erro, "Não foi possível analisar esse vídeo.",
                       (proc.stderr or "") + (proc.stdout or ""))

    def abrir_resultados(self) -> None:
        os.makedirs(RESULTADOS, exist_ok=True)
        os.startfile(RESULTADOS)  # noqa: S606 - abrir pasta no Explorer (Windows)

    def ver_evolucao(self) -> None:
        paciente = self.var_paciente.get().strip()
        if not paciente:
            messagebox.showinfo(
                "Informe o paciente",
                "Digite o nome do paciente no campo acima para ver a evolução "
                "dele entre as sessões.",
            )
            return
        self._status("Gerando o gráfico de evolução...")
        threading.Thread(target=self._gerar_evolucao, args=(paciente,),
                         daemon=True).start()

    def _gerar_evolucao(self, paciente: str) -> None:
        # Executado em processo separado para nao carregar matplotlib dentro da
        # janela do menu (importacao lenta e conflitos de backend com o Tk).
        codigo = (
            "import sys, os, webbrowser; sys.path.insert(0, 'src');"
            "from mandibular.evolution import evolution_path, load_evolution;"
            "from mandibular.plotting import plot_evolution;"
            f"p = evolution_path({RESULTADOS!r}, {paciente!r});"
            "linhas = load_evolution(p);"
            "sys.exit(2) if len(linhas) < 1 else None;"
            f"destino = os.path.join({RESULTADOS!r}, 'evolucao_grafico.png');"
            f"plot_evolution(linhas, destino, {paciente!r});"
            "os.startfile(destino)"
        )
        proc = subprocess.run([python_executavel(), "-c", codigo], cwd=BASE_DIR,
                              capture_output=True, text=True)
        if proc.returncode == 2:
            self.after(0, messagebox.showinfo, "Sem histórico",
                       f"Ainda não há sessões salvas para {paciente}.\n\n"
                       "Faça uma avaliação com esse nome preenchido e salve ao final.")
            self.after(0, self._status, "Sem histórico para esse paciente.")
        elif proc.returncode != 0:
            self.after(0, self._erro, "Não foi possível gerar o gráfico de evolução.",
                       (proc.stderr or "") + (proc.stdout or ""))
        else:
            self.after(0, self._status, "Gráfico de evolução aberto.")

    def mostrar_ajuda(self) -> None:
        janela = tk.Toplevel(self)
        janela.title("Como usar")
        janela.configure(bg="white")
        janela.resizable(False, False)

        texto = tk.Text(janela, width=74, height=30, font=("Segoe UI", 10),
                        bg="white", relief="flat", wrap="word", padx=20, pady=16)
        texto.pack()
        texto.insert("1.0", AJUDA)
        texto.configure(state="disabled")
        tk.Button(janela, text="Fechar", command=janela.destroy,
                  font=("Segoe UI", 10), width=14).pack(pady=(0, 14))


AJUDA = """COMO FAZER UMA AVALIAÇÃO

Antes de começar
  1. Sente o paciente de frente para a câmera, a cerca de 50-70 cm.
  2. Deixe o rosto bem iluminado, sem luz forte atrás dele.
  3. Peça que mantenha a cabeça parada e o rosto voltado para a câmera.
  4. (Opcional) Meça com régua a distância entre os cantos externos dos
     olhos e digite o valor no campo "mm" do menu. Sem essa medida, os
     resultados saem em proporção, e não em milímetros.

Durante a avaliação
  A janela da câmera tem cinco botões na parte de baixo. Basta clicar.

  1. CALIBRAR - o programa pede para o paciente ficar de boca fechada e
     depois abrir bem a boca. Siga as instruções que aparecem na tela.
     Esse passo ensina ao programa qual é a faixa de movimento do paciente.

  2. GRAVAR - começa a registrar. Peça ao paciente que abra e feche a boca
     algumas vezes, no ritmo dele. O contador de repetições sobe sozinho.

  3. SALVAR - encerra o registro, guarda os dados e abre o relatório em
     PDF. Ele traz cada medida com a faixa de referência e a leitura, os
     gráficos e as limitações. É o arquivo para anexar ao prontuário.

  RECOMEÇAR - apaga o que foi registrado e começa de novo.
  SAIR - fecha a janela da câmera e volta a este menu.

Se algo não funcionar
  - Imagem preta ou "câmera não encontrada": feche outros programas que
    usem a webcam (Teams, Zoom, Meet) e tente "Câmera 2" no menu.
  - Aviso "rosto muito distante": aproxime o paciente da câmera.
  - Aviso "pouca luz": acenda mais luzes ou vire o paciente para a janela.

Onde ficam os arquivos
  Tudo fica na pasta "resultados", uma pasta por sessão, com o relatório
  (relatorio.pdf), a versão em HTML, os gráficos e a planilha de dados.
  Use o botão "Abrir a pasta dos resultados" no menu.

Importante
  Este programa é uma ferramenta de APOIO e documentação. Os valores são
  estimativas feitas por câmera comum e não substituem o exame clínico,
  a medição direta nem o diagnóstico profissional.
"""


def main() -> None:
    app = Launcher()
    app.mainloop()


if __name__ == "__main__":
    main()
