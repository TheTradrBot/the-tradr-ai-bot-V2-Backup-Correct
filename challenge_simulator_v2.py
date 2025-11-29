"""
5%ers Challenge Simulator V2 - Dynamic Position Sizing

ENHANCED FEATURES:
1. Dynamic risk per trade based on exposure limits
2. Concurrent trade tracking (multiple open positions)
3. TP1 partial profit + move SL to profit (breakeven+)
4. Exposure management to prevent breach on multiple SL hits

SAFETY RULES:
- Max total exposure: 8% (leaves 2% buffer to max DD)
- Max daily exposure: 4% (leaves 1% buffer to daily DD)
- When TP1 hit: take 50% profit, move SL to entry+0.1R (locked profit)
- Base risk: 2.5-4% per trade (dynamic based on available room)
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field
import math


@dataclass
class OpenTrade:
    """Represents an open trade with risk tracking."""
    trade_id: str
    entry_time: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_final: float
    initial_risk_usd: float
    current_risk_usd: float
    position_size: float = 1.0
    partial_taken: bool = False
    sl_at_profit: bool = False
    locked_profit_usd: float = 0.0


class DynamicChallengeSimulator:
    """
    Enhanced 5%ers challenge simulator with dynamic position sizing.
    
    Key features:
    - Tracks multiple concurrent open trades
    - Adjusts position size based on available exposure room
    - Implements partial TP + breakeven management
    - Prevents breach by limiting total exposure
    """
    
    def __init__(
        self,
        starting_balance: float = 10000,
        max_drawdown_pct: float = 10.0,
        daily_drawdown_pct: float = 5.0,
        max_exposure_pct: float = 8.0,
        max_daily_exposure_pct: float = 4.0,
        base_risk_pct: float = 2.5,
        max_risk_pct: float = 4.0,
        partial_tp_r: float = 1.0,
        partial_close_pct: float = 50.0,
        be_buffer_r: float = 0.1,
        step1_target_pct: float = 8.0,
        step2_target_pct: float = 5.0,
        min_profitable_days: int = 3,
        max_trades_per_day: int = 12,
    ):
        self.starting_balance = starting_balance
        self.max_drawdown_pct = max_drawdown_pct
        self.daily_drawdown_pct = daily_drawdown_pct
        self.max_exposure_pct = max_exposure_pct
        self.max_daily_exposure_pct = max_daily_exposure_pct
        self.base_risk_pct = base_risk_pct
        self.max_risk_pct = max_risk_pct
        self.partial_tp_r = partial_tp_r
        self.partial_close_pct = partial_close_pct
        self.be_buffer_r = be_buffer_r
        self.step1_target_pct = step1_target_pct
        self.step2_target_pct = step2_target_pct
        self.min_profitable_days = min_profitable_days
        self.max_trades_per_day = max_trades_per_day
        
        self.reset()
    
    def reset(self):
        """Reset simulator state."""
        self.balance = self.starting_balance
        self.peak_balance = self.starting_balance
        self.prev_day_balance = self.starting_balance
        self.current_day = None
        
        self.open_trades: Dict[str, OpenTrade] = {}
        self.total_open_risk = 0.0
        self.daily_open_risk = 0.0
        
        self.step1_passed = False
        self.step1_balance = None
        self.step2_target = None
        self.step2_passed = False
        self.blown = False
        self.blown_reason = None
        
        self.daily_pnl = defaultdict(float)
        self.daily_trades = defaultdict(int)
        self.profitable_days = set()
        
        self.trade_log = []
        self.exposure_log = []
    
    def get_available_exposure(self) -> Tuple[float, float]:
        """Calculate available exposure for new trades."""
        max_dd_floor = self.starting_balance * (1 - self.max_drawdown_pct / 100)
        daily_dd_floor = self.prev_day_balance * (1 - self.daily_drawdown_pct / 100)
        
        room_to_max_dd = self.balance - max_dd_floor
        room_to_daily_dd = self.balance - daily_dd_floor
        
        max_allowed_exposure = self.balance * (self.max_exposure_pct / 100)
        max_daily_exposure = self.balance * (self.max_daily_exposure_pct / 100)
        
        available_exposure = min(
            max_allowed_exposure - self.total_open_risk,
            max_daily_exposure - self.daily_open_risk,
            room_to_max_dd - self.total_open_risk,
            room_to_daily_dd - self.daily_open_risk,
        )
        
        return max(0, available_exposure), room_to_max_dd
    
    def calculate_position_risk(self, base_risk_requested: float = None) -> float:
        """
        Calculate actual risk for a new trade based on exposure limits.
        
        Returns the maximum safe risk amount in USD.
        """
        if base_risk_requested is None:
            base_risk_requested = self.balance * (self.base_risk_pct / 100)
        
        max_risk = self.balance * (self.max_risk_pct / 100)
        available_exposure, room_to_dd = self.get_available_exposure()
        
        if available_exposure <= 0:
            return 0
        
        safe_risk = min(base_risk_requested, available_exposure, max_risk)
        
        return max(0, safe_risk)
    
    def open_trade(
        self,
        trade_id: str,
        entry_time: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_final: float,
    ) -> Optional[OpenTrade]:
        """
        Open a new trade with dynamic position sizing.
        
        Returns the OpenTrade object if successful, None if cannot open.
        """
        risk_usd = self.calculate_position_risk()
        
        if risk_usd < 50:
            return None
        
        trade = OpenTrade(
            trade_id=trade_id,
            entry_time=entry_time,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_final=take_profit_final,
            initial_risk_usd=risk_usd,
            current_risk_usd=risk_usd,
        )
        
        self.open_trades[trade_id] = trade
        self.total_open_risk += risk_usd
        self.daily_open_risk += risk_usd
        
        self.exposure_log.append({
            'time': entry_time,
            'action': 'OPEN',
            'trade_id': trade_id,
            'risk_added': risk_usd,
            'total_exposure': self.total_open_risk,
            'balance': self.balance,
        })
        
        return trade
    
    def process_partial_tp(self, trade: OpenTrade, current_price: float) -> float:
        """
        Process partial TP hit - take profit on 50% and move SL to profit.
        
        Returns the profit realized from partial close.
        """
        if trade.partial_taken:
            return 0.0
        
        sl_distance = abs(trade.entry_price - trade.stop_loss)
        if sl_distance == 0:
            return 0.0
        
        if trade.direction == 'long':
            current_r = (current_price - trade.entry_price) / sl_distance
        else:
            current_r = (trade.entry_price - current_price) / sl_distance
        
        if current_r >= self.partial_tp_r:
            partial_profit = (self.partial_close_pct / 100) * trade.current_risk_usd * self.partial_tp_r
            
            remaining_risk_pct = 1 - (self.partial_close_pct / 100)
            old_risk = trade.current_risk_usd
            trade.current_risk_usd *= remaining_risk_pct
            
            if trade.direction == 'long':
                trade.stop_loss = trade.entry_price + (sl_distance * self.be_buffer_r)
            else:
                trade.stop_loss = trade.entry_price - (sl_distance * self.be_buffer_r)
            
            trade.partial_taken = True
            trade.sl_at_profit = True
            trade.locked_profit_usd = trade.current_risk_usd * self.be_buffer_r
            
            risk_reduction = old_risk - trade.current_risk_usd
            self.total_open_risk -= risk_reduction
            self.daily_open_risk = max(0, self.daily_open_risk - risk_reduction)
            
            self.balance += partial_profit
            self.peak_balance = max(self.peak_balance, self.balance)
            
            return partial_profit
        
        return 0.0
    
    def close_trade(
        self,
        trade_id: str,
        exit_time: str,
        exit_price: float,
        hit_tp: bool = False,
        hit_sl: bool = False,
    ) -> Dict:
        """
        Close a trade and calculate final P&L.
        
        Returns trade result dictionary.
        """
        if trade_id not in self.open_trades:
            return {'error': 'Trade not found'}
        
        trade = self.open_trades[trade_id]
        sl_distance = abs(trade.entry_price - trade.stop_loss)
        
        if trade.direction == 'long':
            pnl_pips = exit_price - trade.entry_price
        else:
            pnl_pips = trade.entry_price - exit_price
        
        if sl_distance > 0:
            pnl_r = pnl_pips / (abs(trade.entry_price - trade.stop_loss) if not trade.sl_at_profit else sl_distance)
        else:
            pnl_r = 0
        
        if trade.partial_taken:
            remaining_pnl = pnl_r * trade.current_risk_usd
        else:
            remaining_pnl = pnl_r * trade.current_risk_usd
        
        self.balance += remaining_pnl
        self.peak_balance = max(self.peak_balance, self.balance)
        
        self.total_open_risk -= trade.current_risk_usd
        self.daily_open_risk = max(0, self.daily_open_risk - trade.current_risk_usd)
        
        del self.open_trades[trade_id]
        
        total_pnl = remaining_pnl
        total_r = pnl_r
        if trade.partial_taken:
            total_pnl += (trade.initial_risk_usd * 0.5 * self.partial_tp_r)
            total_r = (total_pnl / trade.initial_risk_usd) if trade.initial_risk_usd > 0 else 0
        
        result = 'WIN' if total_pnl > 0 else ('BE' if abs(total_pnl) < 10 else 'LOSS')
        
        trade_result = {
            'trade_id': trade_id,
            'entry_time': trade.entry_time,
            'exit_time': exit_time,
            'direction': trade.direction,
            'entry_price': trade.entry_price,
            'exit_price': exit_price,
            'initial_risk': trade.initial_risk_usd,
            'pnl_usd': total_pnl,
            'pnl_r': total_r,
            'result': result,
            'partial_taken': trade.partial_taken,
            'balance_after': self.balance,
        }
        
        self.trade_log.append(trade_result)
        
        trade_day = exit_time[:10] if len(exit_time) >= 10 else trade.entry_time[:10]
        self.daily_pnl[trade_day] += total_pnl
        self.daily_trades[trade_day] += 1
        
        if self.daily_pnl[trade_day] > 0:
            self.profitable_days.add(trade_day)
        
        self.exposure_log.append({
            'time': exit_time,
            'action': 'CLOSE',
            'trade_id': trade_id,
            'pnl': total_pnl,
            'total_exposure': self.total_open_risk,
            'balance': self.balance,
        })
        
        return trade_result
    
    def check_breach(self) -> bool:
        """Check if account has breached drawdown limits."""
        max_dd_floor = self.starting_balance * (1 - self.max_drawdown_pct / 100)
        daily_dd_floor = self.prev_day_balance * (1 - self.daily_drawdown_pct / 100)
        
        if self.balance < max_dd_floor:
            self.blown = True
            self.blown_reason = f"Max DD breached: ${self.balance:.0f} < ${max_dd_floor:.0f}"
            return True
        
        if self.balance < daily_dd_floor:
            self.blown = True
            self.blown_reason = f"Daily DD breached: ${self.balance:.0f} < ${daily_dd_floor:.0f}"
            return True
        
        return False
    
    def check_targets(self):
        """Check if step targets are hit."""
        step1_target = self.starting_balance * (1 + self.step1_target_pct / 100)
        
        if not self.step1_passed and self.balance >= step1_target:
            self.step1_passed = True
            self.step1_balance = self.balance
            self.step2_target = self.balance * (1 + self.step2_target_pct / 100)
        
        if self.step1_passed and not self.step2_passed and self.step2_target:
            if self.balance >= self.step2_target:
                self.step2_passed = True
    
    def new_day(self, day: str):
        """Process new trading day."""
        if self.current_day and self.current_day != day:
            if self.daily_pnl[self.current_day] > 0:
                self.profitable_days.add(self.current_day)
            self.prev_day_balance = self.balance
            self.daily_open_risk = sum(t.current_risk_usd for t in self.open_trades.values())
        
        self.current_day = day
    
    def get_status(self) -> Dict:
        """Get current challenge status."""
        return {
            'balance': self.balance,
            'peak_balance': self.peak_balance,
            'open_trades': len(self.open_trades),
            'total_exposure': self.total_open_risk,
            'step1_passed': self.step1_passed,
            'step2_passed': self.step2_passed,
            'blown': self.blown,
            'blown_reason': self.blown_reason,
            'profitable_days': len(self.profitable_days),
        }
    
    def get_final_result(self) -> Dict:
        """Get final challenge result."""
        if self.current_day and self.daily_pnl[self.current_day] > 0:
            self.profitable_days.add(self.current_day)
        
        challenge_passed = (
            self.step1_passed and
            self.step2_passed and
            len(self.profitable_days) >= self.min_profitable_days and
            not self.blown
        )
        
        total_trades = len(self.trade_log)
        wins = sum(1 for t in self.trade_log if t['result'] == 'WIN')
        partials = sum(1 for t in self.trade_log if t.get('partial_taken', False))
        
        return {
            'passed': challenge_passed,
            'step1_passed': self.step1_passed,
            'step2_passed': self.step2_passed,
            'blown': self.blown,
            'blown_reason': self.blown_reason,
            'final_balance': self.balance,
            'starting_balance': self.starting_balance,
            'total_pnl': self.balance - self.starting_balance,
            'total_return_pct': ((self.balance - self.starting_balance) / self.starting_balance) * 100,
            'total_trades': total_trades,
            'wins': wins,
            'win_rate': (wins / total_trades * 100) if total_trades > 0 else 0,
            'partials_taken': partials,
            'profitable_days': len(self.profitable_days),
            'min_profitable_days_required': self.min_profitable_days,
            'trade_log': self.trade_log,
            'exposure_log': self.exposure_log,
        }


def simulate_with_dynamic_sizing(
    trades: List[Dict],
    base_risk_pct: float = 2.5,
    max_risk_pct: float = 4.0,
    max_exposure_pct: float = 8.0,
    partial_tp_r: float = 1.0,
) -> Dict:
    """
    Run challenge simulation with dynamic position sizing.
    
    FIXED: Uses starting balance for risk calculation to prevent exponential compounding.
    """
    starting_balance = 10000
    balance = starting_balance
    peak_balance = starting_balance
    prev_day_balance = starting_balance
    current_day = None
    
    max_dd_floor = starting_balance * 0.90
    daily_dd_pct = 0.05
    
    step1_target = starting_balance * 1.08
    step1_passed = False
    step2_target = None
    step2_passed = False
    blown = False
    blown_reason = None
    
    daily_pnl = defaultdict(float)
    daily_trades = defaultdict(int)
    profitable_days = set()
    trade_log = []
    
    total_open_exposure = 0.0
    
    sorted_trades = sorted(trades, key=lambda x: x.get('entry_time', ''))
    
    for i, trade in enumerate(sorted_trades):
        if blown:
            break
        
        entry_time = trade.get('entry_time', '')
        if len(entry_time) >= 10:
            trade_day = entry_time[:10]
        else:
            continue
        
        if trade_day != current_day:
            if current_day and daily_pnl[current_day] > 0:
                profitable_days.add(current_day)
            current_day = trade_day
            prev_day_balance = balance
            total_open_exposure = 0.0
        
        daily_dd_floor = prev_day_balance * (1 - daily_dd_pct)
        
        base_risk = starting_balance * (base_risk_pct / 100)
        max_risk = starting_balance * (max_risk_pct / 100)
        
        risk_usd = min(base_risk, max_risk)
        
        if balance - risk_usd < max_dd_floor:
            risk_usd = max(0, balance - max_dd_floor - 50)
        
        if balance - risk_usd < daily_dd_floor:
            risk_usd = max(0, balance - daily_dd_floor - 50)
        
        if risk_usd < 50:
            continue
        
        pnl_r = trade.get('r_multiple', trade.get('pnl_r', 0))
        result = trade.get('result', 'UNKNOWN')
        
        pnl_usd = risk_usd * pnl_r
        
        new_balance = balance + pnl_usd
        
        if new_balance < max_dd_floor:
            blown = True
            blown_reason = f"Max DD: ${new_balance:.0f} < ${max_dd_floor:.0f}"
            balance = new_balance
            break
        
        if new_balance < daily_dd_floor:
            blown = True
            blown_reason = f"Daily DD on {trade_day}"
            balance = new_balance
            break
        
        balance = new_balance
        peak_balance = max(peak_balance, balance)
        daily_pnl[trade_day] += pnl_usd
        daily_trades[trade_day] += 1
        
        if not step1_passed and balance >= step1_target:
            step1_passed = True
            step2_target = balance * 1.05
        
        if step1_passed and step2_target and balance >= step2_target:
            step2_passed = True
        
        trade_log.append({
            'time': entry_time,
            'risk': risk_usd,
            'pnl_r': pnl_r,
            'pnl_usd': pnl_usd,
            'balance': balance,
            'result': result,
        })
    
    if current_day and daily_pnl[current_day] > 0:
        profitable_days.add(current_day)
    
    total_trades = len(trade_log)
    wins = sum(1 for t in trade_log if t['result'] in ['WIN', 'PARTIAL_WIN'])
    partials = sum(1 for t in trade_log if t['result'] == 'PARTIAL_WIN')
    
    challenge_passed = (
        step1_passed and
        step2_passed and
        len(profitable_days) >= 3 and
        not blown
    )
    
    return {
        'passed': challenge_passed,
        'step1_passed': step1_passed,
        'step2_passed': step2_passed,
        'blown': blown,
        'blown_reason': blown_reason,
        'final_balance': balance,
        'starting_balance': starting_balance,
        'total_pnl': balance - starting_balance,
        'total_return_pct': ((balance - starting_balance) / starting_balance) * 100,
        'total_trades': total_trades,
        'wins': wins,
        'win_rate': (wins / total_trades * 100) if total_trades > 0 else 0,
        'partials_taken': partials,
        'profitable_days': len(profitable_days),
        'min_profitable_days_required': 3,
        'trade_log': trade_log,
    }


def test_optimal_risk_levels():
    """Test different risk levels to find optimal settings."""
    print("Testing Optimal Risk Levels for 5%ers Challenge")
    print("=" * 70)
    
    sample_trades = []
    for month in range(1, 13):
        for day in range(1, 20):
            pnl_r = 2.5 if (day % 4 == 0) else -1.0
            sample_trades.append({
                'entry_time': f"2024-{month:02d}-{day:02d}T10:00:00",
                'exit_time': f"2024-{month:02d}-{day+2:02d}T10:00:00" if day < 28 else f"2024-{month:02d}-28T10:00:00",
                'direction': 'long',
                'entry': 1.1000,
                'stop_loss': 1.0950,
                'take_profit': 1.1125,
                'highest_r': max(pnl_r, 1.5),
                'r_multiple': pnl_r,
                'result': 'WIN' if pnl_r > 0 else 'LOSS',
            })
    
    for base_risk in [2.5, 3.0, 3.5, 4.0]:
        for max_risk in [4.0, 5.0, 6.0]:
            for max_exposure in [6.0, 8.0, 10.0]:
                if max_risk < base_risk:
                    continue
                
                result = simulate_with_dynamic_sizing(
                    sample_trades,
                    base_risk_pct=base_risk,
                    max_risk_pct=max_risk,
                    max_exposure_pct=max_exposure,
                    partial_tp_r=1.0,
                )
                
                if not result['blown']:
                    print(f"Base {base_risk}% | Max {max_risk}% | Exp {max_exposure}% → "
                          f"P/L: ${result['total_pnl']:,.0f} | "
                          f"S1: {'Y' if result['step1_passed'] else 'N'} | "
                          f"S2: {'Y' if result['step2_passed'] else 'N'} | "
                          f"Days: {result['profitable_days']}")


if __name__ == '__main__':
    test_optimal_risk_levels()
