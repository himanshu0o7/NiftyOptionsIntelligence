#!/usr/bin/env python3
"""
ML Bot Startup Script
Easy way to start the ML bot with different modes
"""

import sys
import argparse
from ml_trading_bot import MLTradingBot

def main():
    parser = argparse.ArgumentParser(description='ML Trading Bot')
    parser.add_argument('--mode', choices=['train', 'predict', 'continuous'], 
                       default='continuous', help='Bot operation mode')
    parser.add_argument('--symbol', default='NIFTY', help='Trading symbol')
    parser.add_argument('--config', default='ml_bot/config.json', help='Config file path')
    
    args = parser.parse_args()
    
    # Initialize bot
    bot = MLTradingBot(config_path=args.config)
    bot.initialize_models()
    
    if args.mode == 'train':
        print(f"🔥 Training ML models for {args.symbol}...")
        results = bot.train_models(args.symbol)
        print("✅ Training completed!")
        
        for model_name, result in results.items():
            print(f"📊 {model_name}: {result['accuracy']:.3f} accuracy")
    
    elif args.mode == 'predict':
        print(f"🔮 Generating prediction for {args.symbol}...")
        
        # Load models
        bot.load_models()
        
        # Get current data (you can modify this)
        current_data = bot._get_current_market_data()
        signal = bot.generate_prediction(current_data)
        
        print(f"🎯 Signal: {signal.action}")
        print(f"📈 Confidence: {signal.confidence:.1%}")
        print(f"💡 Reasoning: {signal.reasoning}")
    
    elif args.mode == 'continuous':
        print("🤖 Starting continuous ML analysis...")
        import asyncio
        asyncio.run(bot.run_continuous_analysis())

if __name__ == "__main__":
    main()