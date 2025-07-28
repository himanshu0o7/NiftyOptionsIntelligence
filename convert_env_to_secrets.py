import os

def convert_env_to_secrets(env_path='.env', secrets_path=os.path.expanduser('~/.streamlit/secrets.toml')):
    if not os.path.exists(env_path):
        print(f"❌ File not found: {env_path}")
        return

    os.makedirs(os.path.dirname(secrets_path), exist_ok=True)
    secrets_dict = {}

    with open(env_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            
            # Group keys into TOML sections if needed
            if "angel" in key.lower():
                secrets_dict.setdefault("angel", {})[key] = value
            elif "news" in key.lower():
                secrets_dict.setdefault("newsapi", {})[key] = value
            elif "telegram" in key.lower() or "chat" in key.lower():
                secrets_dict.setdefault("telegram", {})[key] = value
            else:
                secrets_dict.setdefault("general", {})[key] = value

    with open(secrets_path, 'w') as f:
        for section, kv in secrets_dict.items():
            f.write(f"[{section}]\n")
            for k, v in kv.items():
                f.write(f'{k} = "{v}"\n')
            f.write("\n")
    
    print(f"✅ Converted `{env_path}` → `{secrets_path}` successfully!")

# Run the function
convert_env_to_secrets()

