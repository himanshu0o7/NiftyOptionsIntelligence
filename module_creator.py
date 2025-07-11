import os

TEMPLATES = {
    "session_manager.py": '''# Handles session re-use and token management
import json

def load_session():
    with open("session_tokens.json") as f:
        return json.load(f)
''',
    "order_executor.py": '''# Executes orders using SmartAPI
def place_order(order_data, token):
    # TODO: implement Angel One order placement
    pass
''',
    "greeks_handler.py": '''# Placeholder for Greeks calculations
def fetch_greeks(option_symbol):
    # TODO: Use Open Interest & Greeks here
    return {}
''',
    "risk_management/position_manager.py": '''# Manages risk and position limits
def check_position_limits():
    # TODO: Add risk control logic here
    pass
'''
}

def create_modules(missing_files):
    for file in missing_files:
        dir_name = os.path.dirname(file)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        with open(file, "w") as f:
            f.write(TEMPLATES.get(file, "# Empty Module\n"))
        print(f"🛠 Created: {file}")

if __name__ == "__main__":
    from autocode_checker import check_modules
    missing = check_modules()
    if missing:
        create_modules(missing)

