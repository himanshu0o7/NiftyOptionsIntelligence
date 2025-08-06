import os
import subprocess

EXCLUDED_DIRS = {"venv", ".git", "__pycache__"}
LOG_FILE = "codex_fix_log.txt"

def should_exclude(path):
    return any(excluded in path for excluded in EXCLUDED_DIRS)

def fix_with_codex(file_path):
    print(f"🔧 Fixing: {file_path}")
    try:
        result = subprocess.run(
            [
                "codex",
                "Fix this Python module to be clean, runnable, and complete.",
                "--file", file_path,
                "--save"
            ],
            capture_output=True,
            text=True,
            check=False
        )
        with open(LOG_FILE, "a") as log:
            log.write(f"=== {file_path} ===\n")
            log.write(result.stdout + "\n")
            log.write(result.stderr + "\n")

        if result.returncode == 0:
            print(f"✅ Fixed: {file_path}")
        else:
            print(f"⚠️ Error fixing {file_path}. Check log.")
    except Exception as e:
        print(f"❌ Exception for {file_path}: {e}")

def main():
    print("🔍 Scanning for Python modules to fix...")
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                if not should_exclude(full_path):
                    fix_with_codex(full_path)
    print(f"\n✅ All fix attempts complete. Log saved to {LOG_FILE}")

if __name__ == "__main__":
    main()
