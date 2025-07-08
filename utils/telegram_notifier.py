"""
Telegram Notification System for Trading Alerts
"""
import requests
from typing import Dict, Optional
from datetime import datetime
from utils.logger import Logger

class TelegramNotifier:
    """Send trading alerts via Telegram"""
    
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.logger = Logger()
        self.enabled = bool(bot_token and chat_id)
        
        if not self.enabled:
            self.logger.warning("Telegram notifications disabled - missing bot_token or chat_id")
    
    def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """Send message to Telegram"""
        if not self.enabled:
            self.logger.info(f"Telegram (disabled): {message}")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                self.logger.info("Telegram message sent successfully")
                return True
            else:
                self.logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Telegram send error: {e}")
            return False
    
    def send_trade_alert(self, trade_data: Dict) -> bool:
        """Send trade execution alert"""
        try:
            action = trade_data.get('action', 'UNKNOWN')
            symbol = trade_data.get('symbol', 'UNKNOWN')
            price = trade_data.get('price', 0)
            quantity = trade_data.get('quantity', 0)
            confidence = trade_data.get('confidence', 0)
            
            message = f"""
🤖 *AUTOMATED TRADE EXECUTED*

📊 *Symbol:* {symbol}
🎯 *Action:* {action}
💰 *Price:* ₹{price:,.2f}
📦 *Quantity:* {quantity:,}
🎲 *Confidence:* {confidence:.1%}

⏰ *Time:* {datetime.now().strftime('%H:%M:%S')}
💼 *System:* Live Trading Bot
            """
            
            return self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending trade alert: {e}")
            return False
    
    def send_pnl_update(self, pnl_data: Dict) -> bool:
        """Send P&L update alert"""
        try:
            current_pnl = pnl_data.get('current_pnl', 0)
            daily_pnl = pnl_data.get('daily_pnl', 0)
            total_trades = pnl_data.get('total_trades', 0)
            win_rate = pnl_data.get('win_rate', 0)
            
            emoji = "📈" if daily_pnl >= 0 else "📉"
            
            message = f"""
{emoji} *DAILY P&L UPDATE*

💰 *Current P&L:* ₹{current_pnl:,.0f}
📊 *Daily P&L:* ₹{daily_pnl:,.0f}
🎯 *Total Trades:* {total_trades}
🏆 *Win Rate:* {win_rate:.1f}%

⏰ *Updated:* {datetime.now().strftime('%H:%M:%S')}
            """
            
            return self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending P&L update: {e}")
            return False
    
    def send_risk_alert(self, risk_data: Dict) -> bool:
        """Send risk management alert"""
        try:
            alert_type = risk_data.get('type', 'UNKNOWN')
            message_text = risk_data.get('message', '')
            loss_amount = risk_data.get('loss_amount', 0)
            loss_percentage = risk_data.get('loss_percentage', 0)
            
            if alert_type == 'MTM_LOSS_THRESHOLD':
                emoji = "⚠️"
                title = "MTM LOSS ALERT"
            elif alert_type == 'DAILY_LOSS_LIMIT':
                emoji = "🛑"
                title = "DAILY LOSS LIMIT"
            elif alert_type == 'TRADING_HALTED':
                emoji = "🚨"
                title = "TRADING HALTED"
            else:
                emoji = "⚠️"
                title = "RISK ALERT"
            
            message = f"""
{emoji} *{title}*

🔴 *Alert:* {message_text}
💸 *Loss Amount:* ₹{loss_amount:,.0f}
📊 *Loss %:* {loss_percentage:.1f}% of capital

⏰ *Time:* {datetime.now().strftime('%H:%M:%S')}
🤖 *Action:* Review positions immediately
            """
            
            return self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending risk alert: {e}")
            return False
    
    def send_ml_signal_alert(self, signal_data: Dict) -> bool:
        """Send ML signal generation alert"""
        try:
            symbol = signal_data.get('symbol', 'UNKNOWN')
            action = signal_data.get('action', 'UNKNOWN')
            confidence = signal_data.get('confidence', 0)
            signal_type = signal_data.get('signal_type', 'ML')
            
            message = f"""
🧠 *ML SIGNAL GENERATED*

📊 *Symbol:* {symbol}
🎯 *Action:* {action}
🎲 *Confidence:* {confidence:.1%}
🤖 *Type:* {signal_type}

⏰ *Time:* {datetime.now().strftime('%H:%M:%S')}
📈 *Status:* Signal under review
            """
            
            return self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending ML signal alert: {e}")
            return False
    
    def send_system_status(self, status_data: Dict) -> bool:
        """Send system status update"""
        try:
            status = status_data.get('status', 'UNKNOWN')
            uptime = status_data.get('uptime', 'N/A')
            active_positions = status_data.get('active_positions', 0)
            api_status = status_data.get('api_status', 'UNKNOWN')
            
            message = f"""
🤖 *SYSTEM STATUS UPDATE*

🟢 *Status:* {status}
⏱️ *Uptime:* {uptime}
💼 *Active Positions:* {active_positions}
🔌 *API Status:* {api_status}

⏰ *Timestamp:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            return self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending system status: {e}")
            return False
    
    def send_daily_summary(self, summary_data: Dict) -> bool:
        """Send end-of-day summary"""
        try:
            total_trades = summary_data.get('total_trades', 0)
            winning_trades = summary_data.get('winning_trades', 0)
            daily_pnl = summary_data.get('daily_pnl', 0)
            win_rate = summary_data.get('win_rate', 0)
            best_trade = summary_data.get('best_trade', 0)
            worst_trade = summary_data.get('worst_trade', 0)
            
            emoji = "🎉" if daily_pnl >= 0 else "📉"
            
            message = f"""
{emoji} *DAILY TRADING SUMMARY*

📊 *Total Trades:* {total_trades}
🏆 *Winning Trades:* {winning_trades}
📈 *Win Rate:* {win_rate:.1f}%
💰 *Daily P&L:* ₹{daily_pnl:,.0f}

🥇 *Best Trade:* ₹{best_trade:,.0f}
🥉 *Worst Trade:* ₹{worst_trade:,.0f}

📅 *Date:* {datetime.now().strftime('%Y-%m-%d')}
🤖 *System:* Automated Options Trading
            """
            
            return self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending daily summary: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test Telegram connection"""
        if not self.enabled:
            return False
        
        test_message = f"""
🧪 *TEST MESSAGE*

✅ Telegram notifications are working correctly!
⏰ Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🤖 Options Trading System
        """
        
        return self.send_message(test_message)