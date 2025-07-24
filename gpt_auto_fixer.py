import os
import subprocess

CODE_DIR = "."
FIXED_DIR = "fixed_modules"
PROMPT = "Fix the following code:"

EXCLUDED_DIRS = ["venv", ".venv", "__pycache__", "site-packages", "dist", "build", "fixed_modules"]

if not os.path.exists(FIXED_DIR):
    os.makedirs(FIXED_DIR)

def is_excluded(path):
    return any(ex in path for ex in EXCLUDED_DIRS)

def scan_python_files():
    py_files = []
    for root, _, files in os.walk(CODE_DIR):
        if is_excluded(root):
            continue
        for file in files:
            if file.endswith(".py") and not file.startswith("gpt_auto_fixer"):
                full_path = os.path.join(root, file)
                py_files.append(full_path)
    return py_files

def run_codex_fix(file_path):
    print(f"🔍 Checking: {file_path}")
    output_file = os.path.join(FIXED_DIR, os.path.basename(file_path))

    command = [
        "codex",
        PROMPT,
        "--file",
        file_path,
        "--output",
        output_file,
        "--model",
        "codex-mini-latest",
        "--temp",
        "0.1",
        "--top_p",
        "0.9"
    ]

    try:
        subprocess.run(command, check=True)
        print(f"✅ Fixed: {file_path} → {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Codex Fix Failed: {file_path}")
        print(e)

if __name__ == "__main__":
    files = scan_python_files()
    for f in files:
        run_codex_fix(f)

