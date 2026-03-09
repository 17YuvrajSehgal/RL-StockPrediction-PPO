"""
Example script demonstrating IBKR trading operations.

This script shows how to use the OrderManager and PositionManager
to place orders and track positions.

⚠️ WARNING: This script places REAL orders in paper trading!
Make sure you're connected to paper trading (port 7497).
"""

import asyncio
import logging
from datetime import datetime

from ibkr import (
    IBKRConnection,
    IBKRConfig,
    OrderManager,
    PositionManager,
    OrderAction,
    OrderStatus,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def example_3_monitor_positions(conn: IBKRConnection):
    """
    Example 3: Monitor positions and portfolio.
    
    Demonstrates retrieving positions and calculating portfolio metrics.
    """
    print("\n" + "="*70)
    print("Example 3: Monitor Positions")
    print("="*70)
    
    pos_mgr = PositionManager(conn.ib)
    
    # Get all positions
    print("\nRetrieving positions...")
    positions = await pos_mgr.get_positions()
    
    if positions:
        print(f"\n✓ Found {len(positions)} position(s):")
        print("-"*70)
        
        for pos in positions:
            direction = "LONG" if pos.is_long() else "SHORT"
            print(f"\n{pos.symbol} ({direction}):")
            print(f"  Quantity: {abs(pos.quantity)} shares")
            print(f"  Avg Cost: ${pos.avg_cost:.2f}")
            print(f"  Market Price: ${pos.market_price:.2f}")
            print(f"  Market Value: ${pos.market_value:.2f}")
            print(f"  Unrealized PnL: ${pos.unrealized_pnl:.2f}")
    else:
        print("\nNo positions found")
    
    # Get portfolio summary
    print("\nPortfolio Summary:")
    print("-"*70)
    summary = await pos_mgr.get_portfolio_summary()
    print(f"  Total Positions: {summary.total_positions}")
    print(f"  Long Positions: {summary.long_positions}")
    print(f"  Short Positions: {summary.short_positions}")
    print(f"  Total Market Value: ${summary.total_market_value:.2f}")
    print(f"  Total Unrealized PnL: ${summary.total_unrealized_pnl:.2f}")
    
    print("\n" + "-"*70)


async def example_4_order_callbacks(conn: IBKRConnection):
    """
    Example 4: Subscribe to order updates.
    
    Demonstrates real-time order status monitoring using callbacks.
    """
    print("\n" + "="*70)
    print("Example 4: Order Status Callbacks")
    print("="*70)
    
    order_mgr = OrderManager(conn.ib)
    
    # Define callback
    def on_order_update(order_info):
        print(f"\n📊 Order Update:")
        print(f"  ID: {order_info.order_id}")
        print(f"  Symbol: {order_info.symbol}")
        print(f"  Status: {order_info.status}")
        print(f"  Filled: {order_info.filled_quantity}/{order_info.quantity}")
        
        if order_info.is_filled():
            print(f"  ✓ FILLED @ ${order_info.avg_fill_price:.2f}")
    
    # Subscribe to updates
    order_mgr.subscribe_order_updates(on_order_update)
    print("✓ Subscribed to order updates")
    
    # Place order (with after-hours support)
    print("\nPlacing market order: BUY 5 GOOGL (after-hours enabled)...")
    order = await order_mgr.place_market_order("GOOGL", 5, OrderAction.BUY, outside_rth=True)
    
    # Wait for updates
    print("\nWaiting for order updates...")
    await asyncio.sleep(5)
    
    print("\n" + "-"*70)


async def example_5_complete_trading_workflow(conn: IBKRConnection):
    """
    Example 5: Complete trading workflow.
    
    Demonstrates a complete workflow: place order, monitor fill, check position.
    """
    print("\n" + "="*70)
    print("Example 5: Complete Trading Workflow")
    print("="*70)
    
    order_mgr = OrderManager(conn.ib)
    pos_mgr = PositionManager(conn.ib)
    
    symbol = "TSLA"
    quantity = 5
    
    # Step 1: Check existing position
    print(f"\nStep 1: Check existing {symbol} position...")
    existing_pos = await pos_mgr.get_position(symbol)
    if existing_pos:
        print(f"  Existing position: {existing_pos.quantity} shares")
    else:
        print(f"  No existing {symbol} position")
    
    # Step 2: Place order (with after-hours support)
    print(f"\nStep 2: Place market order: BUY {quantity} {symbol} (after-hours enabled)...")
    order = await order_mgr.place_market_order(symbol, quantity, OrderAction.BUY, outside_rth=True)
    print(f"  Order ID: {order.order_id}")
    
    # Step 3: Wait for fill
    print("\nStep 3: Waiting for order to fill...")
    for i in range(10):
        await asyncio.sleep(1)
        status = await order_mgr.get_order_status(order.order_id)
        
        if status == OrderStatus.FILLED:
            order_info = await order_mgr.get_order(order.order_id)
            print(f"  ✓ Order filled @ ${order_info.avg_fill_price:.2f}")
            break
    
    # Step 4: Check updated position
    print(f"\nStep 4: Check updated {symbol} position...")
    await asyncio.sleep(1)  # Give time for position update
    new_pos = await pos_mgr.get_position(symbol)
    
    if new_pos:
        print(f"  ✓ Position updated:")
        print(f"    Quantity: {new_pos.quantity} shares")
        print(f"    Avg Cost: ${new_pos.avg_cost:.2f}")
        print(f"    Market Value: ${new_pos.market_value:.2f}")
    
    print("\n" + "-"*70)


async def example_6_short_position(conn: IBKRConnection):
    """
    Example 6: Open and close short position.
    
    Demonstrates short selling functionality.
    """
    print("\n" + "="*70)
    print("Example 6: Short Position Trading")
    print("="*70)
    
    order_mgr = OrderManager(conn.ib)
    pos_mgr = PositionManager(conn.ib)
    
    symbol = "NVDA"
    quantity = 5
    
    # Step 1: Open short position (with after-hours support)
    print(f"\nStep 1: Opening short position: SELL {quantity} {symbol} (after-hours enabled)...")
    short_order = await order_mgr.open_short_position(symbol, quantity)
    print(f"  Order ID: {short_order.order_id}")
    print(f"  Action: {short_order.action} (SHORT)")
    
    # Wait for fill
    print("\nStep 2: Waiting for order to fill...")
    for i in range(10):
        await asyncio.sleep(1)
        status = await order_mgr.get_order_status(short_order.order_id)
        
        if status == OrderStatus.FILLED:
            order_info = await order_mgr.get_order(short_order.order_id)
            print(f"  ✓ Short position opened @ ${order_info.avg_fill_price:.2f}")
            break
    
    # Check position
    await asyncio.sleep(1)
    position = await pos_mgr.get_position(symbol)
    if position:
        print(f"\nStep 3: Short position confirmed:")
        print(f"  Quantity: {position.quantity} (negative = short)")
        print(f"  Avg Cost: ${position.avg_cost:.2f}")
    
    # Close short position (with after-hours support)
    print(f"\nStep 4: Closing short position: BUY {quantity} {symbol} (after-hours enabled)...")
    close_order = await order_mgr.close_short_position(symbol, quantity)
    
    # Wait for fill
    for i in range(10):
        await asyncio.sleep(1)
        status = await order_mgr.get_order_status(close_order.order_id)
        
        if status == OrderStatus.FILLED:
            order_info = await order_mgr.get_order(close_order.order_id)
            print(f"  ✓ Short position closed @ ${order_info.avg_fill_price:.2f}")
            break
    
    print("\n" + "-"*70)


async def example_7_comprehensive_test(conn: IBKRConnection):
    """
    Example 7: Comprehensive feature test.
    
    Tests all major features in one workflow.
    """
    print("\n" + "="*70)
    print("Example 7: Comprehensive Feature Test")
    print("="*70)
    
    order_mgr = OrderManager(conn.ib)
    pos_mgr = PositionManager(conn.ib)
    
    print("\n📊 Testing All Features:")
    print("-"*70)
    
    # 1. Market order (after-hours)
    print("\n1. Market Order Test (after-hours enabled)...")
    market_order = await order_mgr.place_market_order("AAPL", 5, "BUY", outside_rth=True)
    await asyncio.sleep(2)
    print(f"   ✓ Market order: {market_order.order_id}")
    
    # 2. Limit order (after-hours)
    print("\n2. Limit Order Test (after-hours enabled)...")
    limit_order = await order_mgr.place_limit_order("MSFT", 5, "BUY", 1.00, outside_rth=True)
    await asyncio.sleep(1)
    print(f"   ✓ Limit order: {limit_order.order_id}")
    
    # 3. Cancel order
    print("\n3. Order Cancellation Test...")
    await order_mgr.cancel_order(limit_order.order_id)
    await asyncio.sleep(1)
    status = await order_mgr.get_order_status(limit_order.order_id)
    print(f"   ✓ Order cancelled: {status}")
    
    # 4. Long position (after-hours)
    print("\n4. Long Position Test (after-hours enabled)...")
    long_order = await order_mgr.open_long_position("GOOGL", 2)
    await asyncio.sleep(2)
    print(f"   ✓ Long position opened: {long_order.order_id}")
    
    # 5. Short position (after-hours)
    print("\n5. Short Position Test (after-hours enabled)...")
    short_order = await order_mgr.open_short_position("TSLA", 2)
    await asyncio.sleep(2)
    print(f"   ✓ Short position opened: {short_order.order_id}")
    
    # 6. Position tracking
    print("\n6. Position Tracking Test...")
    positions = await pos_mgr.get_positions()
    print(f"   ✓ Found {len(positions)} positions:")
    for pos in positions[:5]:  # Show first 5
        direction = "LONG" if pos.is_long() else "SHORT"
        print(f"     - {pos.symbol}: {direction} {abs(pos.quantity)}")
    
    # 7. Portfolio summary
    print("\n7. Portfolio Summary Test...")
    summary = await pos_mgr.get_portfolio_summary()
    print(f"   ✓ Total positions: {summary.total_positions}")
    print(f"   ✓ Long positions: {summary.long_positions}")
    print(f"   ✓ Short positions: {summary.short_positions}")
    print(f"   ✓ Market value: ${summary.total_market_value:.2f}")
    
    # 8. Order history
    print("\n8. Order History Test...")
    all_orders = await order_mgr.get_all_orders()
    open_orders = await order_mgr.get_open_orders()
    print(f"   ✓ Total orders: {len(all_orders)}")
    print(f"   ✓ Open orders: {len(open_orders)}")
    
    print("\n" + "="*70)
    print("✓ All Features Tested Successfully!")
    print("="*70)


async def main():
    """
    Run all trading examples with a single shared connection.
    
    ⚠️ WARNING: These examples place REAL orders in paper trading!
    """
    # Setup logging
    from ibkr.logging_config import setup_logging
    setup_logging(log_level=logging.INFO, console_level=logging.INFO)
    
    print("\n" + "="*70)
    print("IBKR Trading Operations - Comprehensive Test Suite")
    print("="*70)
    print(f"Time: {datetime.now()}")
    print("\n⚠️  WARNING: These examples will place REAL orders!")
    print("Make sure you're connected to PAPER TRADING (port 7497)")
    print("\n📁 Logs are being saved to: ibkr_logs/")
    print("✅ After-hours trading enabled for all orders")
    print("="*70)
    
    config = IBKRConfig.paper_trading()
    
    try:
        # Create single connection for all examples
        print("\n🔌 Establishing connection to TWS/Gateway...")
        async with IBKRConnection(config) as conn:
            print("✓ Connected! Running all examples with shared connection...\n")
            
            # Run all examples with shared connection
            await example_3_monitor_positions(conn)
            await example_4_order_callbacks(conn)
            await example_5_complete_trading_workflow(conn)
            await example_6_short_position(conn)
            await example_7_comprehensive_test(conn)
        
        print("\n" + "="*70)
        print("✓ All Examples Completed Successfully!")
        print("="*70)
        print("\n📁 Check ibkr_logs/ directory for detailed logs")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Verify TWS/Gateway is running on port 7497")
        print("  2. Check that you have sufficient paper trading funds")
        print("  3. Verify market is open (or use paper trading)")
        print("  4. Check ibkr_logs/ for detailed error logs")


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
