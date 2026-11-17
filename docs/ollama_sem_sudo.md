# Ollama sem sudo (Linux) — como foi configurado na máquina de desenvolvimento

O instalador oficial (`curl -fsSL https://ollama.com/install.sh | sh`) precisa de `sudo`:
ele instala em `/usr/local` e cria um serviço do sistema. Aqui o Ollama foi instalado
só para o usuário, sem alterar o sistema operacional.

```bash
# 1. Baixar o pacote oficial (versão usada: v0.40.16) e conferir o SHA-256
mkdir -p ~/.local/opt/ollama-download && cd ~/.local/opt/ollama-download
curl -LO https://github.com/ollama/ollama/releases/download/v0.40.16/ollama-linux-amd64.tar.zst
curl -L https://github.com/ollama/ollama/releases/download/v0.40.16/sha256sum.txt | grep linux-amd64.tar.zst
sha256sum ollama-linux-amd64.tar.zst      # deve ser igual à linha acima

# 2. Extrair em ~/.local/opt/ollama e expor o binário em ~/.local/bin
mkdir -p ~/.local/opt/ollama
tar --zstd -xf ollama-linux-amd64.tar.zst -C ~/.local/opt/ollama
ln -sf ~/.local/opt/ollama/bin/ollama ~/.local/bin/ollama

# 3. Serviço do usuário (inicia sozinho no login; só escuta em 127.0.0.1)
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/ollama.service <<'EOF'
[Unit]
Description=Ollama (usuário) — LLM local do LendingRisk AI
After=network-online.target

[Service]
ExecStart=%h/.local/opt/ollama/bin/ollama serve
Environment=OLLAMA_HOST=127.0.0.1:11434
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now ollama.service

# 4. Baixar o modelo (~2 GB, fica em ~/.ollama/models)
ollama pull llama3.2:3b
ollama list
```

Comandos úteis:

```bash
systemctl --user status ollama        # ver se está rodando
systemctl --user stop ollama          # parar (libera memória)
systemctl --user disable --now ollama # não iniciar mais no login
journalctl --user -u ollama -f        # logs
```

Para desinstalar: `systemctl --user disable --now ollama`, depois remova
`~/.config/systemd/user/ollama.service`, `~/.local/opt/ollama`, `~/.local/bin/ollama` e
`~/.ollama`.
