# Single Trade System Configuration ✅

## New Trading Rules Implemented

Based on your requirement: **₹17,000 max per single trade, no parallel trades, intraday only**

### ✅ Key Changes Made:

#### 1. **Single Trade Mode Activated**
- **Before**: Multiple parallel trades (up to 5 positions)
- **After**: Only 1 trade at a time
- **Logic**: System checks for active positions before placing new trades

#### 2. **Capital Limits Updated**
- **Before**: ₹17,000 total limit, ₹3,400 per trade
- **After**: ₹17,000 maximum per single trade
- **No parallel trading**: Wait for current trade to close

#### 3. **Trading Strategy Modified**
- **Signal Selection**: System picks BEST signal (highest confidence) instead of multiple
- **Position Monitoring**: Monitors single active position until closure
- **Intraday Auto-Close**: Automatically closes positions at 3:25 PM

#### 4. **Capital Manager Updated**
```python
# New single trade validation
if active_positions > 0:
    return False, "Single trade mode: Wait for current position to close"

if order_value > 17000:  # ₹17k max per trade
    return False, "Exceeds ₹17k single trade limit"
```

## How It Works Now

### Step 1: Signal Generation
- System generates multiple signals as before
- **NEW**: Selects only the BEST signal (highest confidence)
- Ignores other signals until current trade closes

### Step 2: Single Trade Execution
```
🎯 Best signal selected: BUY NIFTY 23500 CE (Confidence: 78.5%)
💰 Capital required: ₹13,500 / ₹17,000 max
🔥 LIVE TRADE: Executing BUY NIFTY 23500 CE
✅ Single trade executed: NIFTY23500CE
```

### Step 3: Position Monitoring
- **During Trade**: System monitors the single active position
- **No New Trades**: Blocks all new signals until current trade closes
- **Auto-Close**: Closes position at 3:25 PM (intraday)

### Step 4: Trade Closure
- Position closed manually or by stop-loss/target
- **After Closure**: System ready for next single trade

## Expected Behavior

### ✅ When NO Active Positions:
```
⏳ Scanning for best trading opportunity...
🎯 Best signal: BUY NIFTY 23450 CE (Confidence: 72.3%)
💰 Capital: ₹12,000 / ₹17,000 limit
🔥 Executing single intraday trade...
✅ Order placed: NIFTY23450CE
📱 Telegram: Single Trade executed - BUY NIFTY 23450 CE ₹12,000
```

### ✅ When Active Position EXISTS:
```
📊 Single trade mode: 1 active position - monitoring current trade
Current Position: NIFTY 23450 CE (+₹2,400 unrealized P&L)
⏳ Waiting for current trade to close before new signals
```

### ✅ Market Close (3:25 PM):
```
🕒 Auto-closing intraday positions at market close
✅ Position closed: NIFTY 23450 CE (+₹1,800 profit)
📱 Telegram: Intraday position auto-closed at 3:25 PM
💰 Total P&L: +₹1,800
🔄 Ready for next trading day
```

## Telegram Notifications

Now you'll receive focused notifications:
- **Signal Selection**: "Best signal identified: NIFTY 23500 CE"
- **Trade Execution**: "Single Trade: BUY NIFTY 23500 CE - ₹15,000"
- **Position Updates**: "Current P&L: +₹2,100 (14% gain)"
- **Auto-Close**: "Intraday position closed at 3:25 PM: +₹1,800"

## Capital Usage Examples

### Example 1: NIFTY Trade
- **Strike**: 23500 CE
- **Premium**: ₹180
- **Lot Size**: 75
- **Total**: 75 × ₹180 = ₹13,500 ✅ (Within ₹17k limit)

### Example 2: BANKNIFTY Trade  
- **Strike**: 50200 CE
- **Premium**: ₹800
- **Lot Size**: 15
- **Total**: 15 × ₹800 = ₹12,000 ✅ (Within ₹17k limit)

### Example 3: Rejected Trade
- **Strike**: NIFTY 23500 CE
- **Premium**: ₹250 (high premium)
- **Lot Size**: 75
- **Total**: 75 × ₹250 = ₹18,750 ❌ (Exceeds ₹17k limit)

## Key Benefits

1. **Risk Control**: Maximum ₹17k exposure per trade
2. **Focus**: Single position monitoring instead of multiple
3. **Capital Efficiency**: Full capital available for each trade
4. **Intraday Safety**: Auto-close at market end
5. **Simple Tracking**: Easy P&L monitoring for single position

## System Ready

Your trading system now operates in **Single Trade Mode**:
- ✅ Maximum ₹17,000 per trade
- ✅ No parallel positions
- ✅ Intraday options only
- ✅ 1 lot quantity
- ✅ Auto market close
- ✅ Best signal selection
- ✅ Live trading enabled

**Ready to start single trade automation!** 🚀