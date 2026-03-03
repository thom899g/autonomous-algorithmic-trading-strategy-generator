"""
Real-time and historical market data collection with caching and error handling.
"""
import logging
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import ccxt
from dataclasses import dataclass
from .config import get_config
from .firebase_client import FirebaseClient

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Container for market data with validation"""
    symbol: str
    timeframe: str
    data: pd.DataFrame
    timestamp: datetime
    source: str
    
    def validate(self) -> bool:
        """Validate market data structure"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in self.data.columns for col in required_columns):
            logger.error(f"Missing required columns in market data: {self.symbol}")
            return False
        if len(self.data) == 0:
            logger.error(f"Empty data for symbol: {self.symbol}")
            return False
        return True

class DataCollector:
    """Main data collection class with caching and error handling"""
    
    def __init__(self, firebase_client: Optional[FirebaseClient] = None):
        self.config = get_config()
        self.firebase = firebase_client
        self.exchange = self._initialize_exchange()
        self.cache: Dict[str, MarketData] = {}