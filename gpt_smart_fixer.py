import os
import subprocess
import difflib

PROMPT = "Fix the following code:"
MODEL = "codex-mini-latest"
INPUT_FILE = "input_tmp.py"
OUTPUT_FILE = "output_tmp.py"

def fix_with_codex(filepath):
    with open(filepath, "r") as f:
        original_code = f.read()

    with open(INPUT_FILE, "w") as f:
        f.write(original_code)

    command = [
        "codex",
        PROMPT,
        "--",  # ✅ THIS IS THE FIX
        "--file", INPUT_FILE,
        "--output", OUTPUT_FILE,
        "--model", MODEL,
        "--temp", "0.1",
        "--top_p", "0.9"
    ]

    subprocess.run(command, check=True)

    with open(OUTPUT_FILE, "r") as f:
        fixed_code = f.readlines()

    original_lines = original_code.splitlines(keepends=True)

    print("\n🔁 DIFF PREVIEW:")
    diff = difflib.unified_diff(original_lines, fixed_code, fromfile='original', tofile='fixed')
    print("".join(diff))

    choice = input("\n✅ Overwrite original file? (y/n): ").strip().lower()
    if choice == "y":
        with open(filepath, "w") as f:
            f.writelines(fixed_code)
        print("✔ File updated.")
    else:
        print("❌ File not changed.")

if __name__ == "__main__":
    file = input("Enter path to Python file: ").strip()
    if os.path.exists(file):
        fix_with_codex(file)
    else:
        print("❌ File not found.")

