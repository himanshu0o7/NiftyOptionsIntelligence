import os

REQUIRED_MODULES = [
    "login_manager.py",
    "session_manager.py",
    "order_executor.py",
    "greeks_handler.py",
    "risk_management/position_manager.py"
]

def check_modules():
    missing = []
    for module in REQUIRED_MODULES:
        if not os.path.exists(module):
            missing.append(module)

    if not missing:
        print("✅ All required modules are present.")
    else:
        print("❌ Missing Modules:")
        for m in missing:
            print("   -", m)
        return missing

if __name__ == "__main__":
    check_modules()

