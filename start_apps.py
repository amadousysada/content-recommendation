#!/usr/bin/env python3
import os, sys, time, signal, platform, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FUNC_DIR = ROOT / "app"
STREAMLIT_DIR = ROOT / "streamlit"

FUNC_VENV = FUNC_DIR / ".venv"
ST_VENV = STREAMLIT_DIR / ".venv"

API_PORT = int(os.getenv("FUNC_PORT", "7071"))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
API_BASE = os.getenv("API_BASE", f"http://localhost:{API_PORT}/api")

procs = []

def bash_with_venv(venv_path: Path, cmd: list[str], cwd: Path, env: dict[str, str] = None) -> None:
    """Run a command with 'source <venv>/bin/activate' on Unix."""
    activate = venv_path / "bin" / "activate"
    shell_cmd = f"source {activate} && " + " ".join(cmd)
    return subprocess.Popen(
        ["bash", "-lc", shell_cmd],
        cwd=str(cwd),
        stdout=sys.stdout,
        stderr=sys.stderr,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        env=env,
    )

def win_with_venv(venv_path: Path, cmd: list[str], cwd: Path):
    """Windows variant using activate.bat."""
    activate = venv_path / "Scripts" / "activate.bat"
    shell_cmd = f'"{activate}" && ' + " ".join(cmd)
    return subprocess.Popen(
        ["cmd", "/c", shell_cmd],
        cwd=str(cwd),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

def start(name, venv_path, cmd, cwd, env=None):
    print(f"→ {name}: {' '.join(cmd)} (cwd={cwd})")
    p = (bash_with_venv if platform.system() != "Windows" else win_with_venv)(venv_path, cmd, cwd, env)
    procs.append((name, p))
    return p

def shutdown(*_):
    print("\n⚠️  stopping…")
    for name, p in procs[::-1]:
        try:
            if hasattr(os, "killpg"):
                os.killpg(p.pid, signal.SIGTERM)
            else:
                p.terminate()
        except Exception:
            pass
    time.sleep(2)
    for name, p in procs[::-1]:
        if p.poll() is None:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(p.pid, signal.SIGKILL)
                else:
                    p.kill()
            except Exception:
                pass
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

# Sécurité: vérifier local.settings.json
if not (FUNC_DIR / "local.settings.json").exists():
    print("⚠️  app/local.settings.json manquant (le host Functions ne lira pas tes secrets).")

# 1) Azure Functions (port = 7071 par défaut)
#   IMPORTANT: ne pas injecter AzureWebJobsStorage ici -> lu depuis local.settings.json
func_cmd = ["func", "start", "--port", str(API_PORT)]
start("azure-functions", FUNC_VENV, func_cmd, FUNC_DIR)

# 2) Streamlit (consomme l'API via localhost)
st_env = os.environ.copy()
st_env.setdefault("FUNC_BASE_URL", API_BASE)

# Lance Streamlit dans son venv
st_cmd = ["streamlit", "run", "streamlit_app.py", "--server.port", str(STREAMLIT_PORT), "--server.address", "0.0.0.0"]
print(f"→ streamlit: {' '.join(st_cmd)} (cwd={STREAMLIT_DIR})")
p = (bash_with_venv if platform.system() != "Windows" else win_with_venv)(ST_VENV, st_cmd, STREAMLIT_DIR, st_env)
procs.append(("streamlit", p))

print(f"\n🔗 API base : {API_BASE}")
print(f"🌐 Streamlit : http://localhost:{STREAMLIT_PORT}")
print("Ctrl+C pour quitter.")

while True:
    time.sleep(1)
    for name, p in procs:
        if p.poll() is not None:
            print(f"\n✖ {name} s'est arrêté (code {p.returncode}).")
            shutdown()
