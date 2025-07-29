# 📊 NiftyOptionsIntelligence

Live Nifty Options Dashboard + CE/PE Trigger Bot using Angel One API and Sensibull OI/Greeks.

---

## 🚀 Features
- ✅ ATM CE/PE token auto-detection (AngelOne master contract)
- 📡 Live LTP, OI, Volume stream via SmartWebSocketV2
- 🧠 Option chain trend detector (OI + News sentiment)
- ⚡ Trigger bot for CE/PE trades (Volume + OI based)
- 🖥️ Mobile-friendly Streamlit UI
- ☑️ Plug-and-play for VPS, Replit, or Colab

---

## 🔧 Setup

### 1. Clone Repo
```bash
git clone https://github.com/himanshu0o7/NiftyOptionsIntelligence.git
cd NiftyOptionsIntelligence
```

### 2. Create `.env` file
```env
API_KEY=your_angel_api_key
CLIENT_CODE=your_client_code
PASSWORD=your_password
TOTP_KEY=your_totp_secret
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Run Dashboard
```bash
streamlit run main.py
```

---

## 📂 Project Structure
```bash
.
├── main.py                     # Unified Streamlit UI
├── angel_utils.py             # Session + MasterContract
├── smart_websocket_handler.py # WebSocket manager
├── option_stream_ui.py        # Live CE/PE LTP+OI table
├── ce_pe_trigger_bot.py       # Trigger logic engine
├── trend_detector.py          # OI + Sentiment logic
├── dashboard_ce_pe_launcher.py # UI Bot starter
├── README.md
├── requirements.txt
```

---

## 📞 Contact
**Built by:** Himanshu (Chaudhary saab)
- 📬 Telegram: [@your_channel_here](https://t.me/your_channel_here)
- 💼 GitHub: [himanshu0o7](https://github.com/himanshu0o7)
- 💡 Questions/Suggestions welcome!
