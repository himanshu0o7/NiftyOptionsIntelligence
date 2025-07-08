# ML Trading Bot - मशीन लर्निंग ट्रेडिंग बॉट

## 🎯 **Overview**
यह एक independent ML bot है जो आपके main trading system के साथ communicate करता है। यह अलग से run होता है और ML signals generate करके main system को भेजता है।

## 📁 **Structure**

```
ml_bot/
├── ml_trading_bot.py          # Main ML bot code
├── start_ml_bot.py            # Bot startup script
├── config.json                # Configuration file
├── requirements.txt           # Python dependencies
├── models/                    # Trained ML models (auto-created)
│   ├── ensemble_model.pkl
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   ├── logistic_model.pkl
│   ├── neural_network_model.pkl
│   └── scaler.pkl
├── logs/                      # Log files (auto-created)
│   └── ml_bot.log
└── README.md                  # This file
```

## 🔧 **Features**

### **1. ML Models**
- **Random Forest**: Tree-based ensemble model
- **SVM**: Support Vector Machine for pattern recognition
- **Logistic Regression**: Linear classification model
- **Neural Network**: Multi-layer perceptron
- **Ensemble**: Combined voting classifier

### **2. Technical Features**
- **RSI**: Relative Strength Index
- **EMA**: Exponential Moving Averages
- **VWAP**: Volume Weighted Average Price
- **Volume Analysis**: Volume ratios and patterns
- **Price Momentum**: Short-term price movements
- **Volatility**: Market volatility measures
- **Greeks**: Delta and Gamma changes
- **IV Percentile**: Implied Volatility percentile

### **3. Communication**
- **WebSocket**: Real-time communication with main system
- **HTTP API**: Fallback communication method
- **JSON**: Structured data exchange format

## 🚀 **Installation & Setup**

### **Step 1: Install Dependencies**
```bash
cd ml_bot
pip install -r requirements.txt
```

### **Step 2: Configure Settings**
Edit `config.json` for your requirements:
```json
{
  "model_params": {
    "random_forest": {"n_estimators": 100, "max_depth": 10}
  },
  "training_params": {
    "min_confidence": 0.65
  },
  "main_system_api": "http://localhost:5000"
}
```

### **Step 3: Train Models**
```bash
python start_ml_bot.py --mode train --symbol NIFTY
```

### **Step 4: Start Bot**
```bash
python start_ml_bot.py --mode continuous
```

## 🎮 **Usage Modes**

### **1. Training Mode**
```bash
python start_ml_bot.py --mode train --symbol NIFTY
```
- Fetches historical data
- Trains all ML models
- Saves models to disk
- Shows accuracy metrics

### **2. Prediction Mode**
```bash
python start_ml_bot.py --mode predict --symbol NIFTY
```
- Loads trained models
- Generates single prediction
- Shows signal and confidence

### **3. Continuous Mode**
```bash
python start_ml_bot.py --mode continuous
```
- Runs continuous analysis
- Generates signals every 30 seconds
- Sends signals to main system
- Logs all activities

## 📊 **Signal Format**

ML Bot generates signals in this format:
```json
{
  "type": "ml_signal",
  "data": {
    "symbol": "NIFTY",
    "action": "BUY_CE",
    "confidence": 0.78,
    "source": "ML Ensemble",
    "timestamp": "2025-07-08T15:30:00",
    "reasoning": "Bullish signal with 78% confidence",
    "target_strike": 23500,
    "expected_move": 0.02
  }
}
```

## 🔗 **Integration with Main System**

### **WebSocket Connection**
```python
# Main system receives ML signals
{
  "type": "ml_signal",
  "action": "BUY_CE",
  "confidence": 0.78,
  "reasoning": "ML ensemble prediction"
}
```

### **HTTP API Endpoint**
```python
# Fallback HTTP communication
POST /api/ml_signal
{
  "symbol": "NIFTY",
  "action": "BUY_CE",
  "confidence": 0.78
}
```

## 📈 **Performance Metrics**

Bot tracks these metrics:
- **Accuracy**: Percentage of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confidence**: Model's confidence in predictions

## 🛠️ **Customization**

### **Add New Features**
```python
# In ml_trading_bot.py
self.feature_columns = [
    'rsi', 'ema_short', 'ema_long', 
    'your_new_feature'  # Add here
]
```

### **Modify Models**
```python
# In config.json
"model_params": {
  "random_forest": {
    "n_estimators": 200,  # Increase trees
    "max_depth": 15       # Increase depth
  }
}
```

### **Change Symbols**
```python
# In config.json
"symbols": ["NIFTY", "BANKNIFTY", "FINNIFTY"]
```

## 🔍 **Monitoring**

### **Log Files**
- `ml_bot.log`: All bot activities
- Model training progress
- Signal generation logs
- Error tracking

### **Model Files**
- `models/`: Trained model files
- Automatically saved after training
- Loaded on bot startup

## 🔄 **Maintenance**

### **Model Retraining**
```bash
# Retrain models with fresh data
python start_ml_bot.py --mode train --symbol NIFTY
```

### **Performance Check**
```bash
# Check single prediction
python start_ml_bot.py --mode predict --symbol NIFTY
```

### **Clear Models**
```bash
# Delete old models
rm -rf ml_bot/models/
```

## 🐛 **Troubleshooting**

### **Common Issues**
1. **Import Error**: Install requirements.txt
2. **Connection Error**: Check main system is running
3. **Model Error**: Retrain models with fresh data
4. **WebSocket Error**: Falls back to HTTP API

### **Debug Mode**
```bash
# Run with debug logging
python start_ml_bot.py --mode continuous --debug
```

## 📞 **Support**

For issues or questions:
1. Check log files in `logs/`
2. Verify configuration in `config.json`
3. Ensure main system is running
4. Retrain models if needed

---

## 🎯 **Quick Start Commands**

```bash
# Install and setup
pip install -r requirements.txt

# Train models
python start_ml_bot.py --mode train

# Start bot
python start_ml_bot.py --mode continuous

# Test prediction
python start_ml_bot.py --mode predict
```

**ML Bot ready to use! 🚀**