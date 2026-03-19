"""
Script to close all active open positions in the paper trading account.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the parent directory to sys.path if running as a script directly
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ibkr.config import IBKRConfig
from ibkr.connection import IBKRConnection
from ibkr.positions import PositionManager
from ibkr.trading import OrderManager, OrderAction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def close_all_positions():
    ibkr_config = IBKRConfig.paper_trading()
    
    logger.info("Connecting to IBKR...")
    connection = IBKRConnection(ibkr_config)
    
    try:
        await connection.connect()
        ib = connection.ib
        
        pos_mgr = PositionManager(ib)
        order_mgr = OrderManager(ib)
        
        logger.info("Fetching active positions...")
        # Give IB's reqPositions a bit of time to respond
        await asyncio.sleep(1)
        positions = await pos_mgr.get_positions()
        
        if not positions:
            logger.info("No active open positions found.")
        else:
            logger.info(f"Found {len(positions)} active open positions. Closing them now...")
            
            tasks = []
            for pos in positions:
                if pos.quantity == 0:
                    continue
                    
                action = OrderAction.SELL if pos.quantity > 0 else OrderAction.BUY
                quantity = abs(pos.quantity)
                
                logger.info(f"Placing Market Order to close: {action} {quantity} {pos.symbol}")
                tasks.append(order_mgr.place_market_order(pos.symbol, quantity, action))
                
            await asyncio.gather(*tasks)
            logger.info("All closing orders have been placed.")
            
            # Wait a moment for orders to fill
            await asyncio.sleep(2)
            logger.info("Completed closing positions.")
            
    except Exception as e:
        logger.error(f"Error while closing positions: {e}")
    finally:
        await connection.disconnect()
        logger.info("Disconnected from IBKR.")

if __name__ == "__main__":
    try:
        asyncio.run(close_all_positions())
    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting...")
