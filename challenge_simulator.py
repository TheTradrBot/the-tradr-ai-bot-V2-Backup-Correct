"""
5%ers 10K Challenge Simulator
Simulates challenge with proper position sizing, drawdown rules, and step tracking.

CHALLENGE RULES:
- Account: $10,000
- Step 1: 8% profit target ($800)
- Step 2: 5% profit target (from new balance)
- Max Drawdown: 10% absolute (cannot go below $9,000)
- Daily Drawdown: 5% of previous day's closing balance
- Min 3 profitable trading days required
- Risk per trade: 2.5% = $250

STRATEGY FOCUS:
- High R:R trades (3R-6R) with ~30-40% win rate
- This beats high win rate with low R:R for challenge pass rate
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import math

CHALLENGE_RULES = {
    'starting_balance': 10000,
    'step1_target_pct': 8.0,
    'step2_target_pct': 5.0,
    'max_drawdown_pct': 10.0,
    'daily_drawdown_pct': 5.0,
    'risk_per_trade_usd': 250,
    'min_profitable_days': 3,
    'max_trades_per_day': 12,
}


def simulate_challenge(trades: List[Dict], config: Dict = None) -> Dict:
    """
    Simulate 5%ers challenge with proper position sizing.
    
    Each trade should have:
    - entry_time: datetime string
    - direction: 'long' or 'short'
    - pnl_r: P&L in R-multiples (e.g., -1.0 for full loss, 2.0 for 2R win)
    
    Returns comprehensive challenge result.
    """
    if config is None:
        config = CHALLENGE_RULES
    
    starting_balance = config['starting_balance']
    risk_usd = config['risk_per_trade_usd']
    step1_target = starting_balance * (1 + config['step1_target_pct'] / 100)
    max_dd_floor = starting_balance * (1 - config['max_drawdown_pct'] / 100)
    daily_dd_pct = config['daily_drawdown_pct'] / 100
    
    balance = starting_balance
    peak_balance = starting_balance
    prev_day_balance = starting_balance
    
    step1_passed = False
    step1_balance = None
    step2_target = None
    step2_passed = False
    blown = False
    blown_reason = None
    
    daily_pnl = defaultdict(float)
    daily_trades = defaultdict(int)
    profitable_days = set()
    
    trade_log = []
    current_day = None
    daily_dd_floor = prev_day_balance * (1 - daily_dd_pct)
    
    for trade in trades:
        if blown:
            break
        
        entry_time = trade.get('entry_time', '')
        if isinstance(entry_time, str) and len(entry_time) >= 10:
            trade_day = entry_time[:10]
        else:
            continue
        
        if trade_day != current_day:
            if current_day is not None:
                if daily_pnl[current_day] > 0:
                    profitable_days.add(current_day)
            current_day = trade_day
            prev_day_balance = balance
            daily_dd_floor = prev_day_balance * (1 - daily_dd_pct)
        
        if daily_trades[trade_day] >= config['max_trades_per_day']:
            continue
        
        pnl_r = trade.get('pnl_r', 0)
        pnl_usd = pnl_r * risk_usd
        
        new_balance = balance + pnl_usd
        
        if new_balance < max_dd_floor:
            blown = True
            blown_reason = f"Max drawdown breached (balance ${new_balance:.0f} < ${max_dd_floor:.0f})"
            balance = new_balance
            break
        
        if new_balance < daily_dd_floor:
            blown = True
            blown_reason = f"Daily drawdown breached on {trade_day}"
            balance = new_balance
            break
        
        balance = new_balance
        peak_balance = max(peak_balance, balance)
        daily_pnl[trade_day] += pnl_usd
        daily_trades[trade_day] += 1
        
        if not step1_passed and balance >= step1_target:
            step1_passed = True
            step1_balance = balance
            step2_target = balance * (1 + config['step2_target_pct'] / 100)
        
        if step1_passed and not step2_passed and step2_target and balance >= step2_target:
            step2_passed = True
        
        trade_log.append({
            'time': entry_time,
            'pnl_r': pnl_r,
            'pnl_usd': pnl_usd,
            'balance': balance,
            'step1_passed': step1_passed,
            'step2_passed': step2_passed,
        })
    
    if current_day and daily_pnl[current_day] > 0:
        profitable_days.add(current_day)
    
    total_trades = len(trade_log)
    wins = sum(1 for t in trade_log if t['pnl_r'] > 0)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    total_pnl = balance - starting_balance
    total_return_pct = (total_pnl / starting_balance) * 100
    max_drawdown_seen = ((peak_balance - min(t['balance'] for t in trade_log)) / peak_balance * 100) if trade_log else 0
    
    challenge_passed = (
        step1_passed and 
        step2_passed and 
        len(profitable_days) >= config['min_profitable_days'] and 
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
        'total_pnl': total_pnl,
        'total_return_pct': total_return_pct,
        'total_trades': total_trades,
        'wins': wins,
        'win_rate': win_rate,
        'profitable_days': len(profitable_days),
        'min_profitable_days_required': config['min_profitable_days'],
        'max_drawdown_seen': max_drawdown_seen,
        'trade_log': trade_log,
    }


def convert_backtest_to_challenge_trades(backtest_result: Dict) -> List[Dict]:
    """
    Convert backtest trades to challenge format.
    
    Expects backtest trades with:
    - entry_time
    - pnl_pct (percentage P&L)
    - direction
    
    Converts pnl_pct to R-multiples based on SL distance.
    """
    challenge_trades = []
    
    for trade in backtest_result.get('trades', []):
        entry = trade.get('entry', 0)
        sl = trade.get('sl', 0)
        pnl_pct = trade.get('pnl_pct', 0)
        
        if entry == 0 or sl == 0:
            continue
        
        sl_distance_pct = abs(entry - sl) / entry
        if sl_distance_pct == 0:
            continue
        
        pnl_r = pnl_pct / sl_distance_pct
        
        challenge_trades.append({
            'entry_time': trade.get('entry_time', ''),
            'exit_time': trade.get('exit_time', ''),
            'direction': trade.get('direction', 'long'),
            'pnl_r': pnl_r,
            'original_pnl_pct': pnl_pct,
        })
    
    return challenge_trades


def simulate_monthly_challenge(trades: List[Dict], year: int, month: int, config: Dict = None) -> Dict:
    """Simulate challenge for a specific month."""
    month_str = f"{year}-{month:02d}"
    month_trades = [t for t in trades if t.get('entry_time', '')[:7] == month_str]
    
    if not month_trades:
        return {'error': 'No trades for this month'}
    
    return simulate_challenge(month_trades, config)


def run_multi_month_backtest(backtest_result: Dict, year: int) -> Dict:
    """Run challenge simulation for each month and return summary."""
    challenge_trades = convert_backtest_to_challenge_trades(backtest_result)
    
    monthly_results = {}
    for month in range(1, 13):
        result = simulate_monthly_challenge(challenge_trades, year, month)
        if 'error' not in result:
            monthly_results[f"{year}-{month:02d}"] = result
    
    passed_months = sum(1 for r in monthly_results.values() if r.get('passed', False))
    step1_months = sum(1 for r in monthly_results.values() if r.get('step1_passed', False))
    
    return {
        'year': year,
        'total_months': len(monthly_results),
        'passed_months': passed_months,
        'step1_months': step1_months,
        'monthly_results': monthly_results,
    }


def calculate_required_winrate(target_rr: float, monthly_trades: int = 30) -> Dict:
    """
    Calculate required win rate to pass 5%ers challenge given R:R.
    
    Challenge requires:
    - Step 1: 8% = $800 profit
    - Step 2: 5% = ~$500 profit (from Step 1 balance)
    - Total: ~$1,300 profit needed
    
    With $250 risk per trade, need +5.2R total.
    """
    risk_usd = 250
    step1_target = 800
    step2_target = 500
    total_target = step1_target + step2_target
    total_r_needed = total_target / risk_usd
    
    for wr in range(10, 101, 5):
        win_rate = wr / 100
        expected_trades = monthly_trades
        expected_wins = expected_trades * win_rate
        expected_losses = expected_trades * (1 - win_rate)
        
        expected_pnl = (expected_wins * target_rr * risk_usd) - (expected_losses * risk_usd)
        
        if expected_pnl >= total_target:
            return {
                'target_rr': target_rr,
                'required_winrate': wr,
                'expected_pnl': expected_pnl,
                'expected_trades': expected_trades,
                'passes_challenge': True,
            }
    
    return {
        'target_rr': target_rr,
        'required_winrate': 100,
        'expected_pnl': 0,
        'passes_challenge': False,
    }


if __name__ == '__main__':
    print("5%ERS CHALLENGE - REQUIRED WIN RATES BY R:R")
    print("="*60)
    print(f"Risk per trade: ${CHALLENGE_RULES['risk_per_trade_usd']}")
    print(f"Target: Pass Step 1 (8%) + Step 2 (5%) = ~$1,300 profit")
    print("="*60)
    print(f"{'R:R':<8} {'Min Win Rate':<15} {'Expected P&L':<15} {'Passes?'}")
    print("-"*60)
    
    for rr in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]:
        result = calculate_required_winrate(rr, monthly_trades=30)
        print(f"{rr:<8} {result['required_winrate']:>5}%{'':<9} ${result['expected_pnl']:>10.0f}{'':5} {'YES' if result['passes_challenge'] else 'NO'}")
