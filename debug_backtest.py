"""Debug backtest results for V3 strategy."""

from challenge_5ers import run_challenge_backtest

print('Detailed analysis for GBP_JPY January 2024:')
print('=' * 60)

result = run_challenge_backtest('GBP_JPY', 2024, 1)

if 'error' in result:
    print(f'Error: {result["error"]}')
else:
    trades = result.get('trades', [])
    print(f'Total trades: {len(trades)}')
    
    if trades:
        wins = [t for t in trades if t.get('r_result', 0) > 0]
        losses = [t for t in trades if t.get('r_result', 0) < 0]
        
        print(f'Wins: {len(wins)}')
        print(f'Losses: {len(losses)}')
        
        if trades:
            avg_r = sum(t.get('r_result', 0) for t in trades) / len(trades)
            win_rate = len(wins) / len(trades) * 100
            print(f'Win rate: {win_rate:.1f}%')
            print(f'Avg R result: {avg_r:.2f}')
        
        print()
        print('Sample trades:')
        for i, t in enumerate(trades[:10]):
            entry_time = t.get('entry_time', 'N/A')
            direction = t.get('direction', 'N/A')
            entry = t.get('entry_price', 0)
            sl = t.get('stop_loss', 0)
            tp = t.get('take_profit', 0)
            r_result = t.get('r_result', 0)
            exit_type = t.get('exit_type', 'N/A')
            
            entry_str = str(entry_time)[:10] if entry_time else 'N/A'
            print(f'  {i+1}. {entry_str} | {direction:5} | Entry: {entry:.5f} | SL: {sl:.5f} | TP: {tp:.5f} | Exit: {exit_type} | R: {r_result:.2f}')
    else:
        print('No trades generated!')
    
    challenge = result.get('challenge', {})
    print()
    print('Challenge results:')
    print(f'  Total P/L: ${challenge.get("total_pnl", 0):.0f}')
    print(f'  Step 1 passed: {challenge.get("step1_passed", False)}')
    print(f'  Step 2 passed: {challenge.get("step2_passed", False)}')
    print(f'  Max drawdown: {challenge.get("max_drawdown", 0):.1f}%')
