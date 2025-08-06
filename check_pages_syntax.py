# Example: How to identify syntax error in a file
# Run this in your project root directory to check all .py files under /pages

import os
import py_compile

pages_dir = "pages"
for fname in os.listdir(pages_dir):
    if fname.endswith(".py"):
        fpath = os.path.join(pages_dir, fname)
        try:
            py_compile.compile(fpath, doraise=True)
            print(f"✅ {fname} - OK")
        except py_compile.PyCompileError as e:
            print(f"❌ {fname} - SYNTAX ERROR:")
            print(e.msg)

# Run this script using:
# python check_pages_syntax.py
