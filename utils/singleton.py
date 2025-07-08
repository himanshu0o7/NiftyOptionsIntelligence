"""
Singleton utility for shared components
"""

class DatabaseSingleton:
    """Singleton for database instance"""
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            from core.database import Database
            cls._instance = Database()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset singleton for testing"""
        cls._instance = None

class LoggerSingleton:
    """Singleton for logger instance"""
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            from utils.logger import Logger
            cls._instance = Logger()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset singleton for testing"""
        cls._instance = None