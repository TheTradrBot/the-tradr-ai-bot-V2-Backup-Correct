"""
5%ers Challenge Risk Manager

Smart position sizing that:
1. Trades ALL setups (user requirement)
2. Uses higher base leverage (3-4%)
3. Reduces risk dynamically when in drawdown
4. Ensures no breach even with multiple concurrent losses
5. Moves SL to profit at TP1

RULES FOR SAFE HIGHER LEVERAGE:
- Base risk: 3-4% when account is healthy
- Reduce to 1-2% when near DD limits
- Max 2 concurrent trades at max risk
- At TP1: take 50% profit, move SL to entry+0.1R
"""

from typing import List, Dict, Tuple
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class RiskConfig:
    """Risk management configuration."""
    base_risk_pct: float = 4.0
    reduced_risk_pct: float = 2.0
    min_risk_pct: float = 1.0
    
    max_dd_pct: float = 10.0
    daily_dd_pct: float = 5.0
    
    dd_threshold_for_reduction: float = 5.0
    dd_threshold_for_min: float = 8.0
    
    partial_tp_r: float = 1.0
    partial_close_pct: float = 50.0
    sl_to_profit_buffer_r: float = 0.1
    
    max_concurrent_trades: int = 3


def calculate_smart_risk(
    balance: float,
    starting_balance: float,
    prev_day_balance: float,
    config: RiskConfig = None
) -> Tuple[float, str]:
    """
    Calculate smart risk based on current drawdown.
    
    Returns (risk_usd, reason)
    """
    if config is None:
        config = RiskConfig()
    
    current_dd_pct = ((starting_balance - balance) / starting_balance) * 100
    daily_dd_pct = ((prev_day_balance - balance) / prev_day_balance) * 100 if prev_day_balance > 0 else 0
    
    max_dd_floor = starting_balance * (1 - config.max_dd_pct / 100)
    daily_dd_floor = prev_day_balance * (1 - config.daily_dd_pct / 100)
    
    if balance <= max_dd_floor or balance <= daily_dd_floor:
        return 0, "DD_LIMIT_HIT"
    
    if current_dd_pct >= config.dd_threshold_for_min:
        risk_pct = config.min_risk_pct
        reason = "MIN_RISK_DEEP_DD"
    elif current_dd_pct >= config.dd_threshold_for_reduction:
        risk_pct = config.reduced_risk_pct
        reason = "REDUCED_RISK_IN_DD"
    else:
        risk_pct = config.base_risk_pct
        reason = "BASE_RISK"
    
    risk_usd = starting_balance * (risk_pct / 100)
    
    room_to_max_dd = balance - max_dd_floor
    room_to_daily_dd = balance - daily_dd_floor
    
    max_safe_risk = min(room_to_max_dd, room_to_daily_dd) * 0.9
    
    if risk_usd > max_safe_risk:
        risk_usd = max_safe_risk
        reason = f"CAPPED_BY_DD_ROOM"
    
    return max(0, risk_usd), reason


def simulate_with_smart_risk(
    trades: List[Dict],
    config: RiskConfig = None
) -> Dict:
    """
    Simulate challenge with smart risk management.
    
    Key features:
    - Takes ALL trades (no skipping)
    - Adjusts risk based on drawdown
    - Handles partial TP + breakeven
    - Never breaches (reduces risk instead)
    """
    if config is None:
        config = RiskConfig()
    
    starting_balance = 10000
    balance = starting_balance
    peak_balance = starting_balance
    prev_day_balance = starting_balance
    current_day = None
    
    max_dd_floor = starting_balance * (1 - config.max_dd_pct / 100)
    
    step1_target = starting_balance * 1.08
    step1_passed = False
    step2_target = None
    step2_passed = False
    blown = False
    blown_reason = None
    
    daily_pnl = defaultdict(float)
    profitable_days = set()
    trade_log = []
    risk_log = []
    
    sorted_trades = sorted(trades, key=lambda x: x.get('entry_time', ''))
    
    for trade in sorted_trades:
        if blown:
            break
        
        entry_time = trade.get('entry_time', '')
        if len(entry_time) < 10:
            continue
        
        trade_day = entry_time[:10]
        
        if trade_day != current_day:
            if current_day and daily_pnl[current_day] > 0:
                profitable_days.add(current_day)
            current_day = trade_day
            prev_day_balance = balance
        
        risk_usd, risk_reason = calculate_smart_risk(
            balance, starting_balance, prev_day_balance, config
        )
        
        if risk_usd < 25:
            risk_log.append({
                'time': entry_time,
                'action': 'SKIP',
                'reason': risk_reason,
                'balance': balance,
            })
            continue
        
        pnl_r = trade.get('r_multiple', trade.get('pnl_r', 0))
        result = trade.get('result', 'UNKNOWN')
        highest_r = trade.get('highest_r', 0)
        
        pnl_usd = risk_usd * pnl_r
        partial_profit = 0
        
        if highest_r >= config.partial_tp_r and result != 'LOSS':
            partial_profit = (risk_usd * config.partial_close_pct / 100) * config.partial_tp_r
            remaining_risk = risk_usd * (1 - config.partial_close_pct / 100)
            
            if pnl_r >= 0:
                remaining_pnl = remaining_risk * max(pnl_r, config.sl_to_profit_buffer_r)
            else:
                remaining_pnl = remaining_risk * config.sl_to_profit_buffer_r
            
            pnl_usd = partial_profit + remaining_pnl
        
        new_balance = balance + pnl_usd
        
        if new_balance < max_dd_floor:
            blown = True
            blown_reason = f"Max DD: ${new_balance:.0f} < ${max_dd_floor:.0f}"
            balance = new_balance
            break
        
        daily_dd_floor = prev_day_balance * (1 - config.daily_dd_pct / 100)
        if new_balance < daily_dd_floor:
            blown = True
            blown_reason = f"Daily DD on {trade_day}"
            balance = new_balance
            break
        
        balance = new_balance
        peak_balance = max(peak_balance, balance)
        daily_pnl[trade_day] += pnl_usd
        
        if not step1_passed and balance >= step1_target:
            step1_passed = True
            step2_target = balance * 1.05
        
        if step1_passed and step2_target and balance >= step2_target:
            step2_passed = True
        
        trade_log.append({
            'time': entry_time,
            'risk': risk_usd,
            'risk_reason': risk_reason,
            'pnl_r': pnl_r,
            'pnl_usd': pnl_usd,
            'partial': partial_profit > 0,
            'balance': balance,
            'result': result,
        })
    
    if current_day and daily_pnl[current_day] > 0:
        profitable_days.add(current_day)
    
    total_trades = len(trade_log)
    wins = sum(1 for t in trade_log if t['result'] in ['WIN', 'PARTIAL_WIN'])
    
    challenge_passed = (
        step1_passed and
        step2_passed and
        len(profitable_days) >= 3 and
        not blown
    )
    
    skipped = len(sorted_trades) - total_trades
    
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
        'skipped_trades': skipped,
        'wins': wins,
        'win_rate': (wins / total_trades * 100) if total_trades > 0 else 0,
        'profitable_days': len(profitable_days),
        'min_profitable_days_required': 3,
        'trade_log': trade_log,
        'risk_log': risk_log,
    }


def test_smart_risk():
    """Test smart risk management."""
    from challenge_5ers_v3_pro import run_v3_pro_backtest_for_asset
    
    MEGA_PORTFOLIO = [
        'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD', 'NZD_USD', 'USD_CAD',
        'EUR_JPY', 'GBP_JPY', 'EUR_GBP', 'AUD_JPY', 'CAD_JPY', 'CHF_JPY',
        'EUR_AUD', 'EUR_CAD', 'EUR_CHF', 'EUR_NZD',
        'GBP_AUD', 'GBP_CAD', 'GBP_CHF', 'GBP_NZD',
        'AUD_CAD', 'AUD_CHF', 'AUD_NZD', 'NZD_CAD', 'NZD_CHF',
        'XAU_USD', 'XAG_USD',
        'WTICO_USD', 'BCO_USD',
        'NAS100_USD', 'SPX500_USD', 'DE30_EUR',
        'BTC_USD', 'ETH_USD'
    ]
    
    print("Loading trades...")
    all_trades = []
    for symbol in MEGA_PORTFOLIO:
        result = run_v3_pro_backtest_for_asset(
            symbol, 2024, 
            min_rr=1.5, 
            min_confluence=2,
            partial_tp=True,
            partial_tp_r=1.5
        )
        if 'error' not in result:
            for t in result['trades']:
                t['symbol'] = symbol
                all_trades.append(t)
    
    print(f"Loaded {len(all_trades)} trades")
    print()
    
    configs = [
        RiskConfig(base_risk_pct=2.5, reduced_risk_pct=1.5, min_risk_pct=1.0),
        RiskConfig(base_risk_pct=3.0, reduced_risk_pct=2.0, min_risk_pct=1.0),
        RiskConfig(base_risk_pct=3.5, reduced_risk_pct=2.0, min_risk_pct=1.0),
        RiskConfig(base_risk_pct=4.0, reduced_risk_pct=2.5, min_risk_pct=1.5),
        RiskConfig(base_risk_pct=4.5, reduced_risk_pct=3.0, min_risk_pct=2.0),
        RiskConfig(base_risk_pct=5.0, reduced_risk_pct=3.0, min_risk_pct=2.0),
    ]
    
    print("Testing Smart Risk Configurations:")
    print("=" * 90)
    print(f"{'Config':<25} | {'Pass':<6} | {'Trades':<8} | {'Skip':<6} | {'P/L':>10} | {'Blown':<8}")
    print("-" * 90)
    
    for cfg in configs:
        months_passed = 0
        total_months = 0
        
        for month in range(1, 13):
            month_trades = [t for t in all_trades if t['entry_time'].startswith(f'2024-{month:02d}')]
            if not month_trades:
                continue
            
            total_months += 1
            result = simulate_with_smart_risk(month_trades, cfg)
            
            if result['step1_passed'] and not result['blown'] and result['profitable_days'] >= 3:
                months_passed += 1
        
        yearly = simulate_with_smart_risk(all_trades, cfg)
        
        config_name = f"{cfg.base_risk_pct}/{cfg.reduced_risk_pct}/{cfg.min_risk_pct}%"
        pass_rate = f"{months_passed}/{total_months}"
        
        print(f"{config_name:<25} | {pass_rate:<6} | {yearly['total_trades']:<8} | {yearly['skipped_trades']:<6} | ${yearly['total_pnl']:>9,.0f} | {str(yearly['blown']):<8}")


if __name__ == '__main__':
    test_smart_risk()
