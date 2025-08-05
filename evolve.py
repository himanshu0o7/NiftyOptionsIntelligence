# evolve.py

import os
import json
from strategies.breakout_strategy import BreakoutStrategy
from utils.instrument_downloader import InstrumentDownloader

def audit_and_suggest():
    strategy = BreakoutStrategy("NIFTY", [])
    print("🔍 Loaded Strategy:", strategy.__class__.__name__)
    
    # Dummy audit logic (you can integrate Codex later)
    suggestion = {
        "improvement": "Add vega filter > 1.0 in should_enter",
        "reason": "Too many false positives on low vega contracts",
        "file_to_edit": "strategies/breakout_strategy.py"
    }

    with open("evolve_log.json", "w") as f:
        json.dump(suggestion, f, indent=2)
    
    print("✅ Audit Complete. Suggestion saved to evolve_log.json")

if __name__ == "__main__":
    audit_and_suggest()
