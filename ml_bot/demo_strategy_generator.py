#!/usr/bin/env python3
"""
Demo: AI Strategy Generator with OpenAI Integration
Shows how the ML Bot creates trading strategies using GPT-4o
"""

import os
import json
from datetime import datetime
import pandas as pd
import numpy as np

# OpenAI integration (requires API key)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = bool(os.environ.get("OPENAI_API_KEY"))
    if OPENAI_AVAILABLE:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except ImportError:
    OPENAI_AVAILABLE = False

def analyze_market_conditions():
    """Analyze current market conditions"""
    # Simulate current market data
    market_data = {
        "nifty_trend": "Bullish",
        "volatility": "Medium (VIX: 18.5)",
        "sentiment": "Positive (0.72)",
        "volume": "Above Average",
        "support_levels": [23400, 23350],
        "resistance_levels": [23600, 23700],
        "news_sentiment": "Bullish on IT stocks",
        "global_cues": "Mixed signals from US markets"
    }
    return market_data

def generate_ai_strategy(market_condition, focus_index, risk_level):
    """Generate trading strategy using AI analysis"""
    
    if not OPENAI_AVAILABLE:
        return generate_rule_based_strategy(market_condition, focus_index, risk_level)
    
    # Get market analysis
    market_data = analyze_market_conditions()
    
    # Create AI prompt for strategy generation
    prompt = f"""
    As an expert options trading strategist, create a detailed trading strategy based on:
    
    Market Conditions:
    - Primary Trend: {market_condition}
    - Focus Index: {focus_index}
    - Risk Tolerance: {risk_level}/10
    - Current Data: {json.dumps(market_data, indent=2)}
    
    Generate a comprehensive options trading strategy with:
    1. Strategy name and description
    2. Entry conditions (technical + sentiment)
    3. Strike selection logic (ATM/ITM/OTM)
    4. Risk management (SL, TP, position sizing)
    5. Expected accuracy and reasoning
    6. Implementation steps
    
    Focus on {focus_index} options with proper Greeks validation.
    Capital limit: ₹17,000 total, max ₹3,400 per position.
    
    Respond in JSON format.
    """
    
    try:
        # Use GPT-4o for strategy generation
        response = client.chat.completions.create(
            model="gpt-4o",  # Latest OpenAI model
            messages=[
                {"role": "system", "content": "You are an expert Indian options trading strategist with deep knowledge of NIFTY derivatives."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        strategy = json.loads(response.choices[0].message.content)
        strategy["ai_generated"] = True
        strategy["generation_time"] = datetime.now().isoformat()
        
        return strategy
        
    except Exception as e:
        print(f"AI strategy generation failed: {e}")
        return generate_rule_based_strategy(market_condition, focus_index, risk_level)

def generate_rule_based_strategy(market_condition, focus_index, risk_level):
    """Generate strategy using rule-based logic (fallback)"""
    
    market_data = analyze_market_conditions()
    
    strategy = {
        "name": f"{market_condition} {focus_index} Strategy",
        "description": f"Rule-based {market_condition.lower()} strategy for {focus_index}",
        "market_analysis": market_data,
        "entry_conditions": {
            "primary": f"Market trend: {market_condition}",
            "technical": "RSI < 70 for bullish, RSI > 30 for bearish",
            "sentiment": "News sentiment > 0.6 for long positions",
            "volume": "Above average volume confirmation"
        },
        "strike_selection": {
            "bullish": "ATM CE or slightly ITM",
            "bearish": "ATM PE or slightly ITM", 
            "sideways": "ATM straddle or strangle"
        },
        "risk_management": {
            "position_size": f"₹{min(3400, 17000 * risk_level // 10)}",
            "stop_loss": f"{max(2, 8 - risk_level)}%",
            "take_profit": f"{min(15, 8 + risk_level)}%",
            "time_decay": "Exit if theta > 0.05"
        },
        "expected_accuracy": f"{65 + risk_level}%",
        "implementation": [
            "Monitor market opening for trend confirmation",
            "Check sentiment indicators",
            "Select appropriate strikes based on Greeks",
            "Place orders with proper SL/TP"
        ],
        "ai_generated": False,
        "generation_time": datetime.now().isoformat()
    }
    
    return strategy

def demonstrate_error_fixing():
    """Demonstrate auto error fixing capability"""
    
    if not OPENAI_AVAILABLE:
        return {
            "status": "Limited",
            "message": "OpenAI API key required for auto error fixing",
            "available_fixes": ["Basic error detection", "Rule-based corrections"]
        }
    
    # Simulate common ML errors
    errors = [
        {
            "type": "ModelTrainingError",
            "message": "ValueError: Feature dimension mismatch",
            "context": "X_train shape (100, 12) vs expected (100, 10)",
            "severity": "High"
        },
        {
            "type": "PerformanceWarning", 
            "message": "Model accuracy dropped to 62%",
            "context": "Previous accuracy: 74%",
            "severity": "Medium"
        }
    ]
    
    fixes = []
    
    for error in errors:
        try:
            # Generate AI fix
            fix_prompt = f"""
            Error Analysis:
            Type: {error['type']}
            Message: {error['message']}
            Context: {error['context']}
            
            Provide a specific code fix and explanation for this ML trading bot error.
            Focus on practical solutions for options trading applications.
            
            Respond in JSON format with: root_cause, fix_code, explanation, prevention.
            """
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert ML engineer specializing in trading systems."},
                    {"role": "user", "content": fix_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            fix_data = json.loads(response.choices[0].message.content)
            fix_data["error"] = error
            fix_data["ai_generated"] = True
            fixes.append(fix_data)
            
        except Exception as e:
            # Fallback fix
            fixes.append({
                "error": error,
                "root_cause": "Common ML pipeline issue",
                "fix_code": "# Add validation: assert X.shape[1] == expected_features",
                "explanation": "Add feature validation before model training",
                "prevention": "Implement feature consistency checks",
                "ai_generated": False
            })
    
    return {
        "status": "Active",
        "errors_detected": len(errors),
        "fixes_generated": len(fixes),
        "fixes": fixes
    }

def demonstrate_performance_analysis():
    """Demonstrate AI performance analysis"""
    
    # Simulate trading performance data
    performance_data = {
        "total_trades": 45,
        "winning_trades": 31,
        "accuracy": 68.9,
        "avg_profit": 2.3,
        "avg_loss": -1.8,
        "max_drawdown": 8.5,
        "sharpe_ratio": 1.42,
        "recent_performance": [0.72, 0.68, 0.71, 0.65, 0.74],
        "strategy_breakdown": {
            "breakout": {"trades": 20, "accuracy": 75},
            "oi_analysis": {"trades": 15, "accuracy": 67}, 
            "ml_signals": {"trades": 10, "accuracy": 60}
        }
    }
    
    if not OPENAI_AVAILABLE:
        return {
            "status": "Limited",
            "analysis": "Basic performance metrics available",
            "recommendations": [
                "Improve ML signal accuracy",
                "Focus on breakout strategy",
                "Reduce position size during drawdown"
            ]
        }
    
    try:
        analysis_prompt = f"""
        Trading Performance Analysis:
        {json.dumps(performance_data, indent=2)}
        
        As an expert trading system analyst, provide:
        1. Performance assessment
        2. Specific improvement recommendations
        3. Parameter optimization suggestions
        4. Risk management enhancements
        5. Strategy refinements
        
        Focus on practical, actionable improvements for Indian options trading.
        Respond in JSON format.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert quantitative analyst specializing in options trading performance optimization."},
                {"role": "user", "content": analysis_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        analysis = json.loads(response.choices[0].message.content)
        analysis["performance_data"] = performance_data
        analysis["ai_generated"] = True
        
        return analysis
        
    except Exception as e:
        return {
            "status": "Error",
            "message": f"AI analysis failed: {e}",
            "basic_analysis": "Performance metrics indicate moderate success with room for improvement"
        }

def run_demo():
    """Run complete ML Bot demonstration"""
    print("🤖 ML Bot Self-Evolution Demo")
    print("=" * 50)
    
    # Check OpenAI status
    if OPENAI_AVAILABLE:
        print("✅ OpenAI GPT-4o: Connected")
    else:
        print("⚠️ OpenAI GPT-4o: Not Available (Limited features)")
    
    print("\n1. 📈 AI Strategy Generation")
    print("-" * 30)
    
    # Generate strategy
    strategy = generate_ai_strategy("Bullish", "NIFTY", 6)
    print(f"Strategy: {strategy.get('name', 'Generated Strategy')}")
    print(f"AI Generated: {strategy.get('ai_generated', False)}")
    print(json.dumps(strategy, indent=2)[:500] + "...")
    
    print("\n2. 🔧 Auto Error Fixing")
    print("-" * 30)
    
    # Demonstrate error fixing
    error_analysis = demonstrate_error_fixing()
    print(f"Status: {error_analysis['status']}")
    print(f"Errors Detected: {error_analysis.get('errors_detected', 0)}")
    
    print("\n3. 📊 Performance Analysis")
    print("-" * 30)
    
    # Demonstrate performance analysis
    analysis = demonstrate_performance_analysis()
    print(f"Analysis Status: {analysis.get('status', 'Unknown')}")
    print(f"AI Powered: {analysis.get('ai_generated', False)}")
    
    print("\n✅ Demo Complete!")
    print("Access full features at: http://localhost:8501")

if __name__ == "__main__":
    run_demo()