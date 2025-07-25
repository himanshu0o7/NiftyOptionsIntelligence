--- pages/strategy_config.py	2025-07-24
+++ pages/strategy_config.py	2025-07-24
@@
+import sys
+from pathlib import Path
+
+ROOT = Path(__file__).resolve().parent.parent
+if str(ROOT) not in sys.path:
+    sys.path.append(str(ROOT))
+
 import streamlit as st
