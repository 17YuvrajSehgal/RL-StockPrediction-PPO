"""
Interactive Brokers TWS API Integration Package.

This package provides a production-ready interface to Interactive Brokers
Trader Workstation (TWS) and IB Gateway for algorithmic trading.

Features:
- Async connection management with auto-reconnect
- Order execution and tracking
- Position and account monitoring
- Real-time market data
- Comprehensive error handling and logging

Example:
    >>> from ibkr import IBKRConnection, IBKRConfig
    >>> 
    >>> config = IBKRConfig.paper_trading()
    >>> async with IBKRConnection(config) as conn:
    ...     account = await conn.get_account_summary()
    ...     print(f"Account: {account}")
"""

from ibkr.config import IBKRConfig, TradingMode
from ibkr.connection import IBKRConnection
from ibkr.trading import (
    OrderManager,
    OrderInfo,
    OrderAction,
    OrderType,
    OrderStatus,
)
from ibkr.positions import (
    PositionManager,
    Position,
    PortfolioSummary,
)
from ibkr.exceptions import (
    IBKRException,
    IBKRConnectionError,
    IBKRAuthenticationError,
    IBKRTimeoutError,
    IBKROrderError,
    IBKRDataError,
    IBKRInsufficientFundsError,
    IBKRInvalidOrderError,
)

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "IBKRConfig",
    "TradingMode",
    # Connection
    "IBKRConnection",
    # Trading
    "OrderManager",
    "OrderInfo",
    "OrderAction",
    "OrderType",
    "OrderStatus",
    # Positions
    "PositionManager",
    "Position",
    "PortfolioSummary",
    # Exceptions
    "IBKRException",
    "IBKRConnectionError",
    "IBKRAuthenticationError",
    "IBKRTimeoutError",
    "IBKROrderError",
    "IBKRDataError",
    "IBKRInsufficientFundsError",
    "IBKRInvalidOrderError",
]
