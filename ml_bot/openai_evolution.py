"""
OpenAI-Powered Self-Evolving ML Bot
Automatically improves strategies, fixes errors, and enhances performance
"""

import os
import json
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from openai import OpenAI

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user

class SelfEvolvingBot:
    """
    OpenAI-powered self-evolving ML bot that:
    1. Analyzes performance data to suggest improvements
    2. Auto-fixes code errors in ML models
    3. Generates new trading strategies
    4. Continuously evolves based on market feedback
    """
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.evolution_log = []
        self.performance_history = []
        self.error_fixes = []
        self.strategy_improvements = []
        
        # Setup logging
        logging.basicConfig(
            filename='ml_bot/evolution.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def analyze_performance_with_ai(self, performance_data: Dict) -> Dict:
        """
        Use GPT-4o to analyze performance data and suggest improvements
        """
        try:
            # Prepare performance summary
            performance_summary = {
                "accuracy": performance_data.get("accuracy", 0),
                "precision": performance_data.get("precision", 0),
                "recall": performance_data.get("recall", 0),
                "recent_predictions": performance_data.get("recent_predictions", []),
                "feature_importance": performance_data.get("feature_importance", {}),
                "trading_results": performance_data.get("trading_results", {}),
                "market_conditions": performance_data.get("market_conditions", {})
            }
            
            prompt = f"""
            Analyze this ML bot's trading performance data and provide specific improvement recommendations:
            
            Performance Data:
            {json.dumps(performance_summary, indent=2)}
            
            Please analyze and provide:
            1. Performance strengths and weaknesses
            2. Specific parameter adjustments for ML models
            3. Feature engineering suggestions
            4. Risk management improvements
            5. Market condition adaptations
            6. Code-level improvements with specific implementation
            
            Respond in JSON format with structured recommendations.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert quantitative trading analyst and ML engineer. Provide detailed, actionable recommendations for improving trading bot performance."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            analysis = json.loads(response.choices[0].message.content)
            
            # Log the analysis
            self.logger.info(f"AI Performance Analysis: {analysis}")
            self.evolution_log.append({
                "timestamp": datetime.now().isoformat(),
                "type": "performance_analysis",
                "analysis": analysis,
                "input_data": performance_summary
            })
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in AI performance analysis: {e}")
            return {"error": str(e), "recommendations": []}
    
    def auto_fix_errors(self, error_info: Dict) -> Dict:
        """
        Automatically fix code errors using GPT-4o
        """
        try:
            error_details = {
                "error_type": error_info.get("error_type", ""),
                "error_message": error_info.get("error_message", ""),
                "traceback": error_info.get("traceback", ""),
                "code_context": error_info.get("code_context", ""),
                "file_path": error_info.get("file_path", ""),
                "function_name": error_info.get("function_name", "")
            }
            
            prompt = f"""
            Analyze this ML trading bot error and provide a fix:
            
            Error Details:
            {json.dumps(error_details, indent=2)}
            
            Please provide:
            1. Root cause analysis
            2. Specific code fix with exact replacement
            3. Prevention strategies
            4. Testing recommendations
            5. Alternative approaches if needed
            
            Focus on trading-specific ML issues like:
            - Data preprocessing errors
            - Model training failures
            - Feature engineering issues
            - API integration problems
            - Real-time prediction errors
            
            Respond in JSON format with structured fix recommendations.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert Python developer specializing in ML trading systems. Provide precise, working code fixes."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            fix_recommendations = json.loads(response.choices[0].message.content)
            
            # Log the fix
            self.logger.info(f"AI Error Fix: {fix_recommendations}")
            self.error_fixes.append({
                "timestamp": datetime.now().isoformat(),
                "error_info": error_details,
                "fix_recommendations": fix_recommendations,
                "status": "generated"
            })
            
            return fix_recommendations
            
        except Exception as e:
            self.logger.error(f"Error in auto-fix: {e}")
            return {"error": str(e), "fix": None}
    
    def generate_strategy_improvements(self, market_data: Dict, current_strategies: List[Dict]) -> Dict:
        """
        Generate new trading strategies using GPT-4o
        """
        try:
            strategy_context = {
                "current_strategies": current_strategies,
                "market_data": market_data,
                "performance_metrics": self.get_latest_performance_metrics(),
                "market_conditions": self.analyze_market_conditions(market_data)
            }
            
            prompt = f"""
            Generate improved trading strategies for this ML bot based on current market conditions:
            
            Context:
            {json.dumps(strategy_context, indent=2)}
            
            Current focus: Options trading on NIFTY50, BANKNIFTY, FINNIFTY, MIDCPNIFTY, NIFTYNXT50
            Capital: ₹17,000 total
            Risk per trade: 2% (₹340)
            
            Please generate:
            1. Enhanced technical indicators combinations
            2. Market sentiment integration strategies
            3. Risk-adjusted position sizing algorithms
            4. Multi-timeframe analysis approaches
            5. Options Greeks-based entry/exit rules
            6. Volatility-based strategy adaptations
            
            Include specific implementation code and parameters.
            Respond in JSON format with detailed strategies.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert options trading strategist with deep ML knowledge. Generate practical, profitable trading strategies."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            new_strategies = json.loads(response.choices[0].message.content)
            
            # Log the strategies
            self.logger.info(f"AI Generated Strategies: {new_strategies}")
            self.strategy_improvements.append({
                "timestamp": datetime.now().isoformat(),
                "input_context": strategy_context,
                "generated_strategies": new_strategies,
                "status": "generated"
            })
            
            return new_strategies
            
        except Exception as e:
            self.logger.error(f"Error in strategy generation: {e}")
            return {"error": str(e), "strategies": []}
    
    def continuous_evolution_cycle(self, bot_instance) -> Dict:
        """
        Run continuous evolution cycle - analyze, fix, improve
        """
        try:
            evolution_results = {
                "timestamp": datetime.now().isoformat(),
                "performance_analysis": {},
                "error_fixes": [],
                "strategy_improvements": {},
                "implementation_status": {}
            }
            
            # 1. Performance Analysis
            performance_data = self.collect_performance_data(bot_instance)
            if performance_data:
                evolution_results["performance_analysis"] = self.analyze_performance_with_ai(performance_data)
            
            # 2. Check for errors and fix them
            recent_errors = self.collect_recent_errors(bot_instance)
            for error in recent_errors:
                fix_result = self.auto_fix_errors(error)
                evolution_results["error_fixes"].append(fix_result)
                
                # Attempt to implement the fix
                if "code_fix" in fix_result:
                    implementation_result = self.implement_code_fix(fix_result, bot_instance)
                    evolution_results["implementation_status"][error.get("function_name", "unknown")] = implementation_result
            
            # 3. Generate strategy improvements
            market_data = self.collect_market_data(bot_instance)
            current_strategies = self.get_current_strategies(bot_instance)
            if market_data and current_strategies:
                evolution_results["strategy_improvements"] = self.generate_strategy_improvements(market_data, current_strategies)
            
            # 4. Log evolution cycle
            self.evolution_log.append(evolution_results)
            self.logger.info(f"Evolution cycle completed: {evolution_results}")
            
            return evolution_results
            
        except Exception as e:
            self.logger.error(f"Error in evolution cycle: {e}")
            return {"error": str(e), "cycle_status": "failed"}
    
    def collect_performance_data(self, bot_instance) -> Dict:
        """Collect current performance data from bot instance"""
        try:
            return {
                "accuracy": getattr(bot_instance, 'current_accuracy', 0),
                "recent_predictions": getattr(bot_instance, 'recent_predictions', []),
                "feature_importance": getattr(bot_instance, 'feature_importance', {}),
                "trading_results": getattr(bot_instance, 'trading_results', {}),
                "model_performance": getattr(bot_instance, 'model_performance', {})
            }
        except Exception as e:
            self.logger.error(f"Error collecting performance data: {e}")
            return {}
    
    def collect_recent_errors(self, bot_instance) -> List[Dict]:
        """Collect recent errors from bot instance"""
        try:
            return getattr(bot_instance, 'recent_errors', [])
        except Exception as e:
            self.logger.error(f"Error collecting recent errors: {e}")
            return []
    
    def collect_market_data(self, bot_instance) -> Dict:
        """Collect current market data"""
        try:
            return getattr(bot_instance, 'latest_market_data', {})
        except Exception as e:
            self.logger.error(f"Error collecting market data: {e}")
            return {}
    
    def get_current_strategies(self, bot_instance) -> List[Dict]:
        """Get current trading strategies"""
        try:
            return getattr(bot_instance, 'active_strategies', [])
        except Exception as e:
            self.logger.error(f"Error getting current strategies: {e}")
            return []
    
    def implement_code_fix(self, fix_result: Dict, bot_instance) -> Dict:
        """Implement code fix automatically"""
        try:
            if "code_fix" not in fix_result:
                return {"status": "no_fix_available"}
            
            # Extract fix details
            fix_details = fix_result["code_fix"]
            file_path = fix_details.get("file_path")
            function_name = fix_details.get("function_name")
            new_code = fix_details.get("new_code")
            
            if not all([file_path, function_name, new_code]):
                return {"status": "incomplete_fix_details"}
            
            # Log the implementation attempt
            self.logger.info(f"Implementing fix for {function_name} in {file_path}")
            
            # In production, this would implement the actual code changes
            # For now, we log the proposed changes
            implementation_log = {
                "status": "logged",
                "file_path": file_path,
                "function_name": function_name,
                "proposed_changes": new_code,
                "timestamp": datetime.now().isoformat()
            }
            
            return implementation_log
            
        except Exception as e:
            self.logger.error(f"Error implementing code fix: {e}")
            return {"status": "implementation_failed", "error": str(e)}
    
    def get_latest_performance_metrics(self) -> Dict:
        """Get latest performance metrics"""
        if self.performance_history:
            return self.performance_history[-1]
        return {}
    
    def analyze_market_conditions(self, market_data: Dict) -> Dict:
        """Analyze current market conditions"""
        try:
            # Basic market condition analysis
            conditions = {
                "volatility": "normal",
                "trend": "sideways",
                "volume": "average",
                "sentiment": "neutral"
            }
            
            # Add more sophisticated analysis here
            if market_data:
                # Analyze volatility, trend, volume patterns
                pass
            
            return conditions
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {e}")
            return {}
    
    def get_evolution_summary(self) -> Dict:
        """Get summary of evolution activities"""
        return {
            "total_analyses": len(self.evolution_log),
            "error_fixes": len(self.error_fixes),
            "strategy_improvements": len(self.strategy_improvements),
            "last_evolution": self.evolution_log[-1] if self.evolution_log else None,
            "recent_fixes": self.error_fixes[-5:] if self.error_fixes else [],
            "active_improvements": [s for s in self.strategy_improvements if s.get("status") == "active"]
        }
    
    def save_evolution_state(self):
        """Save evolution state to file"""
        try:
            state = {
                "evolution_log": self.evolution_log,
                "error_fixes": self.error_fixes,
                "strategy_improvements": self.strategy_improvements,
                "performance_history": self.performance_history
            }
            
            with open("ml_bot/evolution_state.json", "w") as f:
                json.dump(state, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving evolution state: {e}")
    
    def load_evolution_state(self):
        """Load evolution state from file"""
        try:
            if os.path.exists("ml_bot/evolution_state.json"):
                with open("ml_bot/evolution_state.json", "r") as f:
                    state = json.load(f)
                
                self.evolution_log = state.get("evolution_log", [])
                self.error_fixes = state.get("error_fixes", [])
                self.strategy_improvements = state.get("strategy_improvements", [])
                self.performance_history = state.get("performance_history", [])
                
        except Exception as e:
            self.logger.error(f"Error loading evolution state: {e}")