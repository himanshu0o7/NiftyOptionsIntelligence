"""
Self-Monitoring Module for ML Bot
Automatically detects issues, performance degradation, and optimization opportunities
"""

import os
import json
import time
import threading
import traceback
import psutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
from collections import deque

@dataclass
class PerformanceMetric:
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    latency_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    prediction_confidence: float
    error_count: int

@dataclass
class SystemAlert:
    timestamp: datetime
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    category: str  # PERFORMANCE, ERROR, RESOURCE, STRATEGY
    message: str
    details: Dict
    auto_fix_available: bool

class SelfMonitoringSystem:
    """
    Comprehensive self-monitoring system that tracks:
    1. Model performance metrics
    2. System resource usage
    3. Error patterns and frequency
    4. Prediction accuracy trends
    5. Trading strategy effectiveness
    """

    def __init__(self, monitoring_interval: int = 30):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitoring_thread = None

        # Data storage
        self.performance_history = deque(maxlen=1000)
        self.alerts = deque(maxlen=500)
        self.error_log = deque(maxlen=200)
        self.strategy_performance = {}

        # Thresholds
        self.thresholds = {
            "min_accuracy": 0.65,
            "max_latency_ms": 1000,
            "max_memory_mb": 1024,
            "max_cpu_percent": 80,
            "min_confidence": 0.6,
            "max_error_rate": 0.05
        }

        # Setup logging
        logging.basicConfig(
            filename='ml_bot/monitoring.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def start_monitoring(self):
        """Start continuous monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.logger.info("Self-monitoring system started")

    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Self-monitoring system stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = self._collect_system_metrics()

                # Analyze performance
                self._analyze_performance(metrics)

                # Check for alerts
                alerts = self._check_alert_conditions(metrics)

                # Log significant changes
                self._log_significant_changes(metrics, alerts)

                # Sleep until next interval
                time.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

    def _collect_system_metrics(self) -> PerformanceMetric:
        """Collect current system metrics"""
        try:
            # System resources
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)

            # Get process-specific memory usage
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Create metrics object (some values will be updated by ML bot)
            metrics = PerformanceMetric(
                timestamp=datetime.now(),
                accuracy=0.0,  # Will be updated by ML bot
                precision=0.0,  # Will be updated by ML bot
                recall=0.0,  # Will be updated by ML bot
                latency_ms=0.0,  # Will be updated by ML bot
                memory_usage_mb=process_memory,
                cpu_usage_percent=cpu_percent,
                prediction_confidence=0.0,  # Will be updated by ML bot
                error_count=0  # Will be updated by ML bot
            )

            self.performance_history.append(metrics)
            return metrics

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return PerformanceMetric(
                timestamp=datetime.now(),
                accuracy=0.0, precision=0.0, recall=0.0, latency_ms=0.0,
                memory_usage_mb=0.0, cpu_usage_percent=0.0,
                prediction_confidence=0.0, error_count=0
            )

    def update_ml_metrics(self, accuracy: float, precision: float, recall: float,
                         latency_ms: float, confidence: float, error_count: int):
        """Update ML-specific metrics"""
        if self.performance_history:
            latest_metrics = self.performance_history[-1]
            latest_metrics.accuracy = accuracy
            latest_metrics.precision = precision
            latest_metrics.recall = recall
            latest_metrics.latency_ms = latency_ms
            latest_metrics.prediction_confidence = confidence
            latest_metrics.error_count = error_count

    def _analyze_performance(self, current_metrics: PerformanceMetric):
        """Analyze performance trends"""
        try:
            if len(self.performance_history) < 10:
                return  # Need more data for analysis

            recent_metrics = list(self.performance_history)[-10:]

            # Calculate trends
            accuracies = [m.accuracy for m in recent_metrics if m.accuracy > 0]
            latencies = [m.latency_ms for m in recent_metrics if m.latency_ms > 0]

            if accuracies:
                accuracy_trend = self._calculate_trend(accuracies)
                if accuracy_trend < -0.05:  # Declining accuracy
                    self._create_alert("HIGH", "PERFORMANCE",
                                     f"Model accuracy declining: {accuracy_trend:.3f} trend",
                                     {"current_accuracy": accuracies[-1], "trend": accuracy_trend})

            if latencies:
                avg_latency = np.mean(latencies)
                if avg_latency > self.thresholds["max_latency_ms"]:
                    self._create_alert("MEDIUM", "PERFORMANCE",
                                     f"High prediction latency: {avg_latency:.1f}ms",
                                     {"average_latency": avg_latency, "threshold": self.thresholds["max_latency_ms"]})

        except Exception as e:
            self.logger.error(f"Error in performance analysis: {e}")

    def _check_alert_conditions(self, metrics: PerformanceMetric) -> List[SystemAlert]:
        """Check for alert conditions"""
        alerts = []

        try:
            # Accuracy check
            if metrics.accuracy > 0 and metrics.accuracy < self.thresholds["min_accuracy"]:
                alerts.append(self._create_alert("HIGH", "PERFORMANCE",
                                               f"Low model accuracy: {metrics.accuracy:.3f}",
                                               {"accuracy": metrics.accuracy, "threshold": self.thresholds["min_accuracy"]}))

            # Memory check
            if metrics.memory_usage_mb > self.thresholds["max_memory_mb"]:
                alerts.append(self._create_alert("MEDIUM", "RESOURCE",
                                               f"High memory usage: {metrics.memory_usage_mb:.1f}MB",
                                               {"memory_mb": metrics.memory_usage_mb, "threshold": self.thresholds["max_memory_mb"]}))

            # CPU check
            if metrics.cpu_usage_percent > self.thresholds["max_cpu_percent"]:
                alerts.append(self._create_alert("MEDIUM", "RESOURCE",
                                               f"High CPU usage: {metrics.cpu_usage_percent:.1f}%",
                                               {"cpu_percent": metrics.cpu_usage_percent, "threshold": self.thresholds["max_cpu_percent"]}))

            # Confidence check
            if metrics.prediction_confidence > 0 and metrics.prediction_confidence < self.thresholds["min_confidence"]:
                alerts.append(self._create_alert("MEDIUM", "PERFORMANCE",
                                               f"Low prediction confidence: {metrics.prediction_confidence:.3f}",
                                               {"confidence": metrics.prediction_confidence, "threshold": self.thresholds["min_confidence"]}))

            # Error rate check
            if len(self.performance_history) >= 10:
                recent_errors = sum(m.error_count for m in list(self.performance_history)[-10:])
                error_rate = recent_errors / 10
                if error_rate > self.thresholds["max_error_rate"]:
                    alerts.append(self._create_alert("HIGH", "ERROR",
                                                   f"High error rate: {error_rate:.3f}",
                                                   {"error_rate": error_rate, "threshold": self.thresholds["max_error_rate"]}))

            return alerts

        except Exception as e:
            self.logger.error(f"Error checking alert conditions: {e}")
            return []

    def _create_alert(self, severity: str, category: str, message: str, details: Dict) -> SystemAlert:
        """Create and store alert"""
        alert = SystemAlert(
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            message=message,
            details=details,
            auto_fix_available=self._check_auto_fix_availability(category, details)
        )

        self.alerts.append(alert)
        self.logger.warning(f"Alert: {severity} - {message}")

        return alert

    def _check_auto_fix_availability(self, category: str, details: Dict) -> bool:
        """Check if auto-fix is available for this alert"""
        auto_fix_categories = {
            "PERFORMANCE": ["accuracy", "latency", "confidence"],
            "RESOURCE": ["memory", "cpu"],
            "ERROR": ["error_rate"],
            "STRATEGY": ["performance"]
        }

        return category in auto_fix_categories

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of values"""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return coeffs[0]  # Slope

    def _log_significant_changes(self, metrics: PerformanceMetric, alerts: List[SystemAlert]):
        """Log significant changes"""
        if alerts:
            for alert in alerts:
                self.logger.info(f"Alert generated: {alert.severity} - {alert.message}")

        # Log periodic status
        if len(self.performance_history) % 20 == 0:  # Every 20 measurements
            self.logger.info(f"System status: Memory={metrics.memory_usage_mb:.1f}MB, "
                           f"CPU={metrics.cpu_usage_percent:.1f}%, "
                           f"Accuracy={metrics.accuracy:.3f}")

    def get_health_report(self) -> Dict:
        """Generate comprehensive health report"""
        try:
            if not self.performance_history:
                return {"status": "no_data", "message": "No monitoring data available"}

            recent_metrics = list(self.performance_history)[-10:]
            latest = self.performance_history[-1]

            # Calculate averages
            avg_accuracy = np.mean([m.accuracy for m in recent_metrics if m.accuracy > 0])
            avg_latency = np.mean([m.latency_ms for m in recent_metrics if m.latency_ms > 0])
            avg_memory = np.mean([m.memory_usage_mb for m in recent_metrics])
            avg_cpu = np.mean([m.cpu_usage_percent for m in recent_metrics])

            # Get recent alerts
            recent_alerts = [a for a in self.alerts if a.timestamp > datetime.now() - timedelta(hours=1)]

            # Determine overall health
            health_score = self._calculate_health_score(recent_metrics)

            report = {
                "timestamp": datetime.now().isoformat(),
                "health_score": health_score,
                "status": self._get_health_status(health_score),
                "current_metrics": {
                    "accuracy": latest.accuracy,
                    "latency_ms": latest.latency_ms,
                    "memory_mb": latest.memory_usage_mb,
                    "cpu_percent": latest.cpu_usage_percent,
                    "confidence": latest.prediction_confidence
                },
                "averages": {
                    "accuracy": avg_accuracy if not np.isnan(avg_accuracy) else 0,
                    "latency_ms": avg_latency if not np.isnan(avg_latency) else 0,
                    "memory_mb": avg_memory,
                    "cpu_percent": avg_cpu
                },
                "recent_alerts": len(recent_alerts),
                "alert_breakdown": {
                    "HIGH": len([a for a in recent_alerts if a.severity == "HIGH"]),
                    "MEDIUM": len([a for a in recent_alerts if a.severity == "MEDIUM"]),
                    "LOW": len([a for a in recent_alerts if a.severity == "LOW"])
                },
                "monitoring_duration": len(self.performance_history) * self.monitoring_interval,
                "auto_fixes_available": len([a for a in recent_alerts if a.auto_fix_available])
            }

            return report

        except Exception as e:
            self.logger.error(f"Error generating health report: {e}")
            return {"status": "error", "message": str(e)}

    def _calculate_health_score(self, metrics: List[PerformanceMetric]) -> float:
        """Calculate overall health score (0-100)"""
        try:
            scores = []

            # Accuracy score
            accuracies = [m.accuracy for m in metrics if m.accuracy > 0]
            if accuracies:
                avg_accuracy = np.mean(accuracies)
                accuracy_score = min(100, avg_accuracy * 100)
                scores.append(accuracy_score)

            # Resource score
            avg_memory = np.mean([m.memory_usage_mb for m in metrics])
            avg_cpu = np.mean([m.cpu_usage_percent for m in metrics])

            memory_score = max(0, 100 - (avg_memory / self.thresholds["max_memory_mb"]) * 100)
            cpu_score = max(0, 100 - (avg_cpu / self.thresholds["max_cpu_percent"]) * 100)

            scores.extend([memory_score, cpu_score])

            # Error score
            total_errors = sum(m.error_count for m in metrics)
            error_score = max(0, 100 - (total_errors * 10))  # Penalize errors heavily
            scores.append(error_score)

            return np.mean(scores) if scores else 0

        except Exception as e:
            self.logger.error(f"Error calculating health score: {e}")
            return 0

    def _get_health_status(self, score: float) -> str:
        """Get health status based on score"""
        if score >= 80:
            return "EXCELLENT"
        elif score >= 60:
            return "GOOD"
        elif score >= 40:
            return "FAIR"
        elif score >= 20:
            return "POOR"
        else:
            return "CRITICAL"

    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [a for a in self.alerts if a.timestamp > cutoff]

        return [{
            "timestamp": a.timestamp.isoformat(),
            "severity": a.severity,
            "category": a.category,
            "message": a.message,
            "details": a.details,
            "auto_fix_available": a.auto_fix_available
        } for a in recent]

    def log_error(self, error_type: str, error_message: str, traceback_str: str = ""):
        """Log error for monitoring"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "message": error_message,
            "traceback": traceback_str
        }

        self.error_log.append(error_entry)
        self.logger.error(f"Error logged: {error_type} - {error_message}")

    def get_performance_trends(self) -> Dict:
        """Get performance trends over time"""
        try:
            if len(self.performance_history) < 5:
                return {"message": "Insufficient data for trends"}

            metrics = list(self.performance_history)

            # Calculate trends
            accuracies = [m.accuracy for m in metrics if m.accuracy > 0]
            latencies = [m.latency_ms for m in metrics if m.latency_ms > 0]
            memory_usage = [m.memory_usage_mb for m in metrics]

            trends = {}

            if accuracies:
                trends["accuracy"] = {
                    "current": accuracies[-1],
                    "trend": self._calculate_trend(accuracies),
                    "trend_direction": "improving" if self._calculate_trend(accuracies) > 0 else "declining"
                }

            if latencies:
                trends["latency"] = {
                    "current_ms": latencies[-1],
                    "trend": self._calculate_trend(latencies),
                    "trend_direction": "improving" if self._calculate_trend(latencies) < 0 else "declining"
                }

            if memory_usage:
                trends["memory"] = {
                    "current_mb": memory_usage[-1],
                    "trend": self._calculate_trend(memory_usage),
                    "trend_direction": "stable" if abs(self._calculate_trend(memory_usage)) < 5 else "increasing"
                }

            return trends

        except Exception as e:
            self.logger.error(f"Error calculating performance trends: {e}")
            return {"error": str(e)}