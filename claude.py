import anthropic
client = anthropic.Anthropic(api_key="sk-ant-your-key-here")
def ask_claude(prompt):
    response = client.messages.create(model="claude-3.5-sonnet-20241022", max_tokens=1000, messages=[{"role": "user", "content": prompt}])
    return response.content[0].text
print(ask_claude("Write a Python script to fetch BTC/INR price from CoinSwitch API"))

