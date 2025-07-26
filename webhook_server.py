from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "✅ Webhook server is live!"

@app.route('/', methods=['POST'])
def webhook():
    data = request.json
    print("📩 Webhook Received:", data)
    return jsonify({"status": "success", "data": data}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)


