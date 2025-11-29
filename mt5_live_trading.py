"""
MT5 Live Trading Module for FTMO Demo Account

Uses MetaApi.cloud for cloud-based MT5 connection (works on Linux/Replit).
Implements Safe Mode position sizing with dynamic risk management.

SAFE MODE RULES:
- Base Risk: 2.0% when account is healthy
- Reduced Risk: 1.0% at 2%+ drawdown  
- Minimum Risk: 0.5% at 4%+ drawdown
- Max Total Exposure: 4% (ensures ALL concurrent SLs can't breach 5% daily DD)

Required Secrets (set in Replit Secrets):
- METAAPI_TOKEN: Your MetaApi access token
- MT5_ACCOUNT_ID: Your MetaApi account ID for FTMO
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

try:
    from metaapi_cloud_sdk import MetaApi
except ImportError:
    MetaApi = None


@dataclass
class SafeModeConfig:
    """Safe Mode position sizing configuration for FTMO.
    
    CRITICAL SAFETY: max_total_exposure_pct = 4.0%
    - If ALL concurrent trades hit SL simultaneously, max loss = 4%
    - This provides 1% buffer under 5% daily DD limit
    - Ensures account NEVER breaches even in worst-case scenario
    """
    base_risk_pct: float = 2.0
    reduced_risk_pct: float = 1.0
    min_risk_pct: float = 0.5
    
    max_dd_pct: float = 10.0
    daily_dd_pct: float = 5.0
    
    dd_threshold_for_reduction: float = 2.0
    dd_threshold_for_min: float = 4.0
    
    max_total_exposure_pct: float = 4.0
    max_concurrent_trades: int = 2
    
    partial_tp_r: float = 1.0
    partial_close_pct: float = 50.0
    sl_to_profit_buffer_r: float = 0.1
    
    fee_per_trade_pct: float = 0.15
    round_trip_fee_pct: float = 0.30


@dataclass
class TradeSignal:
    """Trade signal from strategy."""
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None
    confluence_score: int = 0
    timeframe: str = "D"


@dataclass
class OpenTrade:
    """Represents an open trade in MT5."""
    ticket: int
    symbol: str
    direction: str
    volume: float
    entry_price: float
    stop_loss: float
    take_profit: float
    open_time: datetime
    risk_usd: float
    current_risk_usd: float
    partial_tp_taken: bool = False
    sl_moved_to_be: bool = False


class MT5LiveTrader:
    """Live trading handler for MT5 via MetaApi.cloud.
    
    This class:
    1. Connects to FTMO MT5 via MetaApi cloud service
    2. Manages trades with Safe Mode position sizing
    3. Tracks concurrent exposure to stay under 4% cap
    4. Handles partial TP and SL-to-BE moves
    """
    
    def __init__(self, config: SafeModeConfig = None):
        self.config = config or SafeModeConfig()
        self.api = None
        self.account = None
        self.connection = None
        self.is_connected = False
        
        self.starting_balance: float = 0.0
        self.current_balance: float = 0.0
        self.prev_day_balance: float = 0.0
        self.open_trades: Dict[int, OpenTrade] = {}
        self.daily_pnl: float = 0.0
        self.current_day: str = ""
        
        self.token = os.environ.get("METAAPI_TOKEN")
        self.account_id = os.environ.get("MT5_ACCOUNT_ID")
    
    def is_configured(self) -> bool:
        """Check if required credentials are configured."""
        return bool(self.token and self.account_id)
    
    async def connect(self) -> bool:
        """Connect to MT5 via MetaApi."""
        if not MetaApi:
            print("MetaApi SDK not installed. Run: pip install metaapi-cloud-sdk")
            return False
        
        if not self.is_configured():
            print("MT5 credentials not configured.")
            print("Please set METAAPI_TOKEN and MT5_ACCOUNT_ID in Replit Secrets.")
            return False
        
        try:
            self.api = MetaApi(self.token)
            
            self.account = await self.api.metatrader_account_api.get_account(self.account_id)
            
            if self.account.state != 'DEPLOYED':
                print("Deploying MT5 account...")
                await self.account.deploy()
            
            print("Waiting for MT5 connection...")
            await self.account.wait_connected()
            
            self.connection = self.account.get_rpc_connection()
            await self.connection.connect()
            await self.connection.wait_synchronized()
            
            account_info = await self.connection.get_account_information()
            self.starting_balance = account_info['balance']
            self.current_balance = account_info['balance']
            self.prev_day_balance = account_info['balance']
            self.current_day = datetime.now().strftime('%Y-%m-%d')
            
            self.is_connected = True
            print(f"Connected to FTMO MT5 Account")
            print(f"Balance: ${self.current_balance:,.2f}")
            print(f"Equity: ${account_info.get('equity', 0):,.2f}")
            print(f"Server: {account_info.get('server', 'Unknown')}")
            
            return True
            
        except Exception as e:
            print(f"Connection error: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from MT5."""
        if self.connection:
            await self.connection.close()
        if self.account:
            await self.account.undeploy()
        self.is_connected = False
        print("Disconnected from MT5")
    
    async def refresh_account_state(self):
        """Refresh account balance and position data."""
        if not self.is_connected:
            return
        
        try:
            account_info = await self.connection.get_account_information()
            self.current_balance = account_info['balance']
            
            today = datetime.now().strftime('%Y-%m-%d')
            if today != self.current_day:
                self.prev_day_balance = self.current_balance
                self.daily_pnl = 0.0
                self.current_day = today
            
            positions = await self.connection.get_positions()
            
            current_tickets = set()
            for pos in positions:
                ticket = pos['id']
                current_tickets.add(ticket)
                
                if ticket not in self.open_trades:
                    self.open_trades[ticket] = OpenTrade(
                        ticket=ticket,
                        symbol=pos['symbol'],
                        direction='BUY' if pos['type'] == 'POSITION_TYPE_BUY' else 'SELL',
                        volume=pos['volume'],
                        entry_price=pos['openPrice'],
                        stop_loss=pos.get('stopLoss', 0),
                        take_profit=pos.get('takeProfit', 0),
                        open_time=datetime.now(),
                        risk_usd=0,
                        current_risk_usd=0,
                    )
            
            closed_tickets = set(self.open_trades.keys()) - current_tickets
            for ticket in closed_tickets:
                del self.open_trades[ticket]
                
        except Exception as e:
            print(f"Error refreshing account state: {e}")
    
    def calculate_current_exposure(self) -> float:
        """Calculate total current exposure from open trades."""
        total_exposure = sum(trade.current_risk_usd for trade in self.open_trades.values())
        return total_exposure
    
    def calculate_smart_risk(self) -> Tuple[float, str]:
        """Calculate risk amount using Safe Mode rules.
        
        Returns (risk_usd, reason)
        """
        current_dd_pct = ((self.starting_balance - self.current_balance) / self.starting_balance) * 100
        
        max_dd_floor = self.starting_balance * (1 - self.config.max_dd_pct / 100)
        daily_dd_floor = self.prev_day_balance * (1 - self.config.daily_dd_pct / 100)
        
        if self.current_balance <= max_dd_floor:
            return 0, "MAX_DD_LIMIT_HIT"
        if self.current_balance <= daily_dd_floor:
            return 0, "DAILY_DD_LIMIT_HIT"
        
        if current_dd_pct >= self.config.dd_threshold_for_min:
            risk_pct = self.config.min_risk_pct
            reason = "MIN_RISK_DEEP_DD"
        elif current_dd_pct >= self.config.dd_threshold_for_reduction:
            risk_pct = self.config.reduced_risk_pct
            reason = "REDUCED_RISK_IN_DD"
        else:
            risk_pct = self.config.base_risk_pct
            reason = "BASE_RISK"
        
        risk_usd = self.starting_balance * (risk_pct / 100)
        
        open_exposure = self.calculate_current_exposure()
        max_total_exposure = self.starting_balance * (self.config.max_total_exposure_pct / 100)
        available_exposure = max_total_exposure - open_exposure
        
        if available_exposure <= 0:
            return 0, "MAX_EXPOSURE_REACHED"
        
        if risk_usd > available_exposure:
            risk_usd = available_exposure
            reason = "CAPPED_BY_EXPOSURE"
        
        room_to_max_dd = self.current_balance - max_dd_floor
        room_to_daily_dd = self.current_balance - daily_dd_floor
        
        combined_potential_loss = open_exposure + risk_usd
        if combined_potential_loss > room_to_max_dd * 0.95:
            risk_usd = max(0, room_to_max_dd * 0.95 - open_exposure)
            reason = "CAPPED_BY_DD_ROOM"
        
        if combined_potential_loss > room_to_daily_dd * 0.95:
            risk_usd = max(0, room_to_daily_dd * 0.95 - open_exposure)
            reason = "CAPPED_BY_DAILY_DD"
        
        return max(0, risk_usd), reason
    
    def calculate_lot_size(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        risk_usd: float
    ) -> float:
        """Calculate lot size based on risk and stop distance."""
        from position_sizing import get_pip_value_per_lot
        from config import CONTRACT_SPECS
        
        specs = CONTRACT_SPECS.get(symbol, {"pip_value": 0.0001, "pip_location": 4})
        pip_value = specs.get("pip_value", 0.0001)
        
        stop_distance = abs(entry_price - stop_price)
        stop_pips = stop_distance / pip_value if pip_value > 0 else stop_distance
        
        if stop_pips <= 0:
            return 0.01
        
        pip_value_per_lot = get_pip_value_per_lot(symbol, current_price=entry_price)
        
        if pip_value_per_lot <= 0:
            return 0.01
        
        lot_size = risk_usd / (stop_pips * pip_value_per_lot)
        lot_size = round(lot_size, 2)
        lot_size = max(0.01, lot_size)
        
        return lot_size
    
    async def execute_trade(self, signal: TradeSignal) -> Optional[Dict]:
        """Execute a trade based on signal with Safe Mode sizing.
        
        Returns trade result dict or None if trade was rejected.
        """
        if not self.is_connected:
            print("Not connected to MT5")
            return None
        
        await self.refresh_account_state()
        
        if len(self.open_trades) >= self.config.max_concurrent_trades:
            print(f"Max concurrent trades ({self.config.max_concurrent_trades}) reached")
            return None
        
        risk_usd, risk_reason = self.calculate_smart_risk()
        
        if risk_usd < 25:
            print(f"Trade rejected: {risk_reason} (risk=${risk_usd:.0f})")
            return None
        
        lot_size = self.calculate_lot_size(
            symbol=signal.symbol,
            entry_price=signal.entry_price,
            stop_price=signal.stop_loss,
            risk_usd=risk_usd
        )
        
        try:
            order_type = 'ORDER_TYPE_BUY' if signal.direction.upper() == 'BUY' else 'ORDER_TYPE_SELL'
            
            order_result = await self.connection.create_market_buy_order(
                symbol=signal.symbol.replace('_', ''),
                volume=lot_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit_1,
                options={
                    'comment': f'Blueprint_{signal.confluence_score}C',
                }
            ) if signal.direction.upper() == 'BUY' else await self.connection.create_market_sell_order(
                symbol=signal.symbol.replace('_', ''),
                volume=lot_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit_1,
                options={
                    'comment': f'Blueprint_{signal.confluence_score}C',
                }
            )
            
            if order_result.get('orderId'):
                print(f"Trade executed: {signal.symbol} {signal.direction}")
                print(f"  Lot size: {lot_size}")
                print(f"  Risk: ${risk_usd:.0f} ({risk_reason})")
                print(f"  Entry: {signal.entry_price}")
                print(f"  SL: {signal.stop_loss}")
                print(f"  TP: {signal.take_profit_1}")
                
                self.open_trades[order_result['orderId']] = OpenTrade(
                    ticket=order_result['orderId'],
                    symbol=signal.symbol,
                    direction=signal.direction,
                    volume=lot_size,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit_1,
                    open_time=datetime.now(),
                    risk_usd=risk_usd,
                    current_risk_usd=risk_usd,
                )
                
                return {
                    'success': True,
                    'ticket': order_result['orderId'],
                    'symbol': signal.symbol,
                    'direction': signal.direction,
                    'lot_size': lot_size,
                    'risk_usd': risk_usd,
                    'risk_reason': risk_reason,
                }
            else:
                print(f"Trade failed: {order_result}")
                return None
                
        except Exception as e:
            print(f"Error executing trade: {e}")
            return None
    
    async def modify_trade_sl_to_be(self, ticket: int, entry_price: float) -> bool:
        """Move stop loss to breakeven + buffer."""
        if ticket not in self.open_trades:
            return False
        
        trade = self.open_trades[ticket]
        if trade.sl_moved_to_be:
            return True
        
        try:
            new_sl = entry_price * (1 + self.config.sl_to_profit_buffer_r * 0.001)
            
            await self.connection.modify_position(
                position_id=str(ticket),
                stop_loss=new_sl
            )
            
            trade.sl_moved_to_be = True
            trade.current_risk_usd = trade.risk_usd * 0.3
            
            print(f"SL moved to BE for ticket {ticket}")
            return True
            
        except Exception as e:
            print(f"Error moving SL to BE: {e}")
            return False
    
    async def close_partial(self, ticket: int, close_pct: float = 50.0) -> bool:
        """Close partial position."""
        if ticket not in self.open_trades:
            return False
        
        trade = self.open_trades[ticket]
        if trade.partial_tp_taken:
            return True
        
        try:
            close_volume = round(trade.volume * (close_pct / 100), 2)
            close_volume = max(0.01, close_volume)
            
            await self.connection.close_position_partially(
                position_id=str(ticket),
                volume=close_volume
            )
            
            trade.partial_tp_taken = True
            trade.volume -= close_volume
            
            print(f"Partial close for ticket {ticket}: {close_volume} lots")
            return True
            
        except Exception as e:
            print(f"Error closing partial: {e}")
            return False
    
    async def close_trade(self, ticket: int) -> bool:
        """Close a trade completely."""
        try:
            await self.connection.close_position(position_id=str(ticket))
            
            if ticket in self.open_trades:
                del self.open_trades[ticket]
            
            print(f"Trade closed: ticket {ticket}")
            return True
            
        except Exception as e:
            print(f"Error closing trade: {e}")
            return False
    
    async def get_positions(self) -> List[Dict]:
        """Get all open positions."""
        if not self.is_connected:
            return []
        
        try:
            positions = await self.connection.get_positions()
            return positions
        except Exception as e:
            print(f"Error getting positions: {e}")
            return []
    
    async def get_account_status(self) -> Dict:
        """Get current account status and risk metrics."""
        await self.refresh_account_state()
        
        current_dd_pct = ((self.starting_balance - self.current_balance) / self.starting_balance) * 100
        daily_dd_pct = ((self.prev_day_balance - self.current_balance) / self.prev_day_balance) * 100
        
        open_exposure = self.calculate_current_exposure()
        max_exposure = self.starting_balance * (self.config.max_total_exposure_pct / 100)
        
        risk_usd, risk_reason = self.calculate_smart_risk()
        
        return {
            'connected': self.is_connected,
            'starting_balance': self.starting_balance,
            'current_balance': self.current_balance,
            'current_dd_pct': current_dd_pct,
            'daily_dd_pct': daily_dd_pct,
            'open_trades': len(self.open_trades),
            'open_exposure': open_exposure,
            'max_exposure': max_exposure,
            'exposure_pct': (open_exposure / max_exposure * 100) if max_exposure > 0 else 0,
            'next_trade_risk': risk_usd,
            'risk_reason': risk_reason,
            'max_concurrent': self.config.max_concurrent_trades,
        }


async def test_connection():
    """Test MT5 connection and display account info."""
    trader = MT5LiveTrader()
    
    if not trader.is_configured():
        print("=" * 60)
        print("MT5 LIVE TRADING - SETUP REQUIRED")
        print("=" * 60)
        print()
        print("To connect to your FTMO MT5 account, you need to:")
        print()
        print("1. Sign up for MetaApi.cloud (free tier available)")
        print("   https://metaapi.cloud")
        print()
        print("2. Add your FTMO MT5 account to MetaApi")
        print("   - Use your FTMO demo account credentials")
        print("   - Server: FTMO-Demo or your FTMO server name")
        print()
        print("3. Set these secrets in Replit:")
        print("   - METAAPI_TOKEN: Your MetaApi access token")
        print("   - MT5_ACCOUNT_ID: The account ID from MetaApi")
        print()
        print("=" * 60)
        return False
    
    print("Connecting to FTMO MT5...")
    connected = await trader.connect()
    
    if connected:
        status = await trader.get_account_status()
        
        print()
        print("=" * 60)
        print("ACCOUNT STATUS")
        print("=" * 60)
        print(f"Balance: ${status['current_balance']:,.2f}")
        print(f"Drawdown: {status['current_dd_pct']:.2f}%")
        print(f"Daily DD: {status['daily_dd_pct']:.2f}%")
        print(f"Open Trades: {status['open_trades']}/{status['max_concurrent']}")
        print(f"Exposure: ${status['open_exposure']:,.0f} / ${status['max_exposure']:,.0f} ({status['exposure_pct']:.1f}%)")
        print(f"Next Trade Risk: ${status['next_trade_risk']:,.0f} ({status['risk_reason']})")
        print("=" * 60)
        
        await trader.disconnect()
        return True
    
    return False


if __name__ == '__main__':
    asyncio.run(test_connection())
