# ML Bot User Guide

## How to Use the Self-Evolving ML Bot

### 🚀 Quick Start

**Two Systems Running:**
- **Main Trading System**: http://localhost:5000 (Full trading dashboard)
- **ML Bot GUI**: http://localhost:8501 (AI evolution features)

### 🤖 ML Bot Features

#### 1. **AI-Powered Analysis** 
Uses GPT-4o to analyze your trading performance and suggest improvements.

**How to use:**
- Go to "Evolution Dashboard" tab
- Click "📊 Run Performance Analysis" 
- AI will analyze your data and suggest specific improvements

#### 2. **Auto Error Fixing**
Automatically detects and fixes code errors in ML models.

**How to use:**
- Go to "Auto Error Fixing" tab
- Click "🔍 Scan for Errors"
- System will detect issues and generate fixes automatically

#### 3. **AI Strategy Generator**
Creates new trading strategies based on market conditions.

**How to use:**
- Go to "Strategy Generator" tab
- Select market condition (Bullish/Bearish/Sideways)
- Choose risk tolerance (1-10)
- Pick index (NIFTY, BANKNIFTY, etc.)
- Click "🚀 Generate Strategy"

#### 4. **Continuous Evolution**
Bot automatically improves every 30 minutes using AI analysis.

**How to enable:**
- Go to "Evolution Dashboard"
- Check "🔄 Enable Continuous Evolution"
- Set interval (15-120 minutes)
- System will auto-improve based on market feedback

### 📊 Performance Monitoring

The ML Bot tracks:
- **Accuracy**: How often predictions are correct
- **Confidence**: How sure the bot is about predictions
- **Health Score**: Overall system performance (0-100)
- **Evolution Cycles**: Number of AI improvements made

### 🔧 Required Setup

**For Full Features:**
- OpenAI API key needed for AI analysis
- Without it: Basic ML features only

**With OpenAI:**
- Full AI analysis and strategy generation
- Automatic error fixing
- Continuous learning and improvement

### 🎯 Integration with Main Trading

**Data Flow:**
1. Main system sends market data to ML Bot
2. ML Bot analyzes and creates enhanced signals
3. AI evolution improves the models automatically
4. Better predictions sent back to main system

**Enhanced Signals Include:**
- News sentiment analysis
- Web data insights
- Technical indicator combinations
- AI-generated confidence scores

### 💡 Best Practices

1. **Start with Training**
   - Let the bot train on historical data first
   - Check accuracy before live trading

2. **Monitor Performance**
   - Watch the health score (aim for >80)
   - Review evolution logs regularly

3. **Use AI Analysis**
   - Run performance analysis weekly
   - Implement suggested improvements

4. **Enable Auto Evolution**
   - Set 30-60 minute intervals
   - Let AI continuously improve strategies

### 🚨 Troubleshooting

**Common Issues:**

1. **"Bot Needs Training"**
   - Go to ML Bot GUI
   - Click train models
   - Wait for completion

2. **"OpenAI Not Connected"**
   - Add OPENAI_API_KEY to environment
   - Restart ML Bot for full features

3. **Low Accuracy**
   - Run AI performance analysis
   - Follow GPT-4o recommendations
   - Enable continuous evolution

### 🔄 Daily Workflow

**Morning Setup:**
1. Check ML Bot health score
2. Review overnight evolution results
3. Run AI performance analysis if needed

**During Trading:**
1. Monitor enhanced signals from ML Bot
2. Check confidence scores before trades
3. Let auto-evolution run in background

**Evening Review:**
1. Check trading performance
2. Review ML Bot improvements
3. Adjust evolution settings if needed

### 📈 Advanced Features

**Custom Strategy Generation:**
- AI creates strategies based on your specific market conditions
- Combines technical analysis with sentiment data
- Adapts to different market regimes automatically

**Performance Optimization:**
- AI suggests specific parameter changes
- Automatic feature engineering improvements
- Dynamic risk management adjustments

**Error Prevention:**
- Detects potential issues before they occur
- Suggests code improvements
- Monitors system health continuously

### 🎯 Success Tips

1. **Start Simple**: Use basic features first, then enable advanced AI
2. **Monitor Results**: Check accuracy and health scores daily
3. **Trust the AI**: Let evolution system make improvements automatically
4. **Review Logs**: Check what AI changes are being made
5. **Stay Updated**: AI learns from market changes continuously