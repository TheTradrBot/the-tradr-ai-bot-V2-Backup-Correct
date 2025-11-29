"""
5%ers 10K High Stakes Challenge Strategy
=========================================
Account: $10,000
Target: Pass Step 1 (8%) + Step 2 (5%) within 2 weeks

CHALLENGE RULES:
- Step 1 Profit Target: 8% ($800)
- Step 2 Profit Target: 5% ($500)
- Max Drawdown: 10% absolute (can't go below $9,000)
- Daily Drawdown: 5% of previous day's closing equity/balance (higher)
- Minimum 3 profitable trading days required
- Stop loss required within 3 minutes
- Leverage: Up to 1:100

RISK MANAGEMENT:
- Risk per trade: 1-2% ($100-$200)
- Max trades per day: 3 (to control daily DD)
- Prefer 6:1 R:R for faster target achievement
"""

import os
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import discord

OANDA_API_KEY = os.environ.get('OANDA_API_KEY', '')
BASE_URL = 'https://api-fxpractice.oanda.com'
HEADERS = {'Authorization': f'Bearer {OANDA_API_KEY}', 'Content-Type': 'application/json'}

CHALLENGE_CONFIG = {
    'account_size': 10000,
    'step1_target_pct': 8.0,
    'step2_target_pct': 5.0,
    'max_drawdown_pct': 10.0,
    'daily_drawdown_pct': 5.0,
    'risk_per_trade_pct': 1.5,
    'max_trades_per_day': 10,
    'min_profitable_days': 3,
    'leverage': 100,
}

ASSET_CONFIGS_HIGH_RR = {
    'EUR_USD': {'tp_rr': 6.0, 'atr_mult': 0.4, 'ema': 100, 'pip_value': 10.0, 'pip_size': 0.0001},
    'GBP_USD': {'tp_rr': 6.0, 'atr_mult': 0.4, 'ema': 100, 'pip_value': 10.0, 'pip_size': 0.0001},
    'USD_JPY': {'tp_rr': 6.0, 'atr_mult': 0.4, 'ema': 100, 'pip_value': 9.0, 'pip_size': 0.01},
    'USD_CHF': {'tp_rr': 4.0, 'atr_mult': 0.5, 'ema': 100, 'pip_value': 11.0, 'pip_size': 0.0001},
    'AUD_USD': {'tp_rr': 4.0, 'atr_mult': 0.5, 'ema': 100, 'pip_value': 10.0, 'pip_size': 0.0001},
    'NZD_USD': {'tp_rr': 6.0, 'atr_mult': 0.4, 'ema': 100, 'pip_value': 10.0, 'pip_size': 0.0001},
    'USD_CAD': {'tp_rr': 6.0, 'atr_mult': 0.4, 'ema': 100, 'pip_value': 7.5, 'pip_size': 0.0001},
    'EUR_GBP': {'tp_rr': 5.0, 'atr_mult': 0.5, 'ema': 100, 'pip_value': 12.5, 'pip_size': 0.0001},
    'EUR_JPY': {'tp_rr': 5.0, 'atr_mult': 0.5, 'ema': 100, 'pip_value': 9.0, 'pip_size': 0.01},
    'GBP_JPY': {'tp_rr': 4.0, 'atr_mult': 0.5, 'ema': 100, 'pip_value': 9.0, 'pip_size': 0.01},
    'XAU_USD': {'tp_rr': 5.0, 'atr_mult': 0.5, 'ema': 100, 'pip_value': 1.0, 'pip_size': 0.01},
    'BTC_USD': {'tp_rr': 6.0, 'atr_mult': 0.4, 'ema': 100, 'pip_value': 1.0, 'pip_size': 1.0},
    'ETH_USD': {'tp_rr': 6.0, 'atr_mult': 0.4, 'ema': 100, 'pip_value': 1.0, 'pip_size': 0.01},
    'SPX500_USD': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 100, 'pip_value': 1.0, 'pip_size': 0.1},
    'NAS100_USD': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 100, 'pip_value': 1.0, 'pip_size': 0.1},
}

ASSET_CONFIGS_HIGH_WINRATE = {
    'EUR_USD': {'tp_rr': 1.5, 'atr_mult': 1.0, 'ema': 50, 'pip_value': 10.0, 'pip_size': 0.0001, 'rsi_ob': 70, 'rsi_os': 30},
    'GBP_USD': {'tp_rr': 1.5, 'atr_mult': 1.0, 'ema': 50, 'pip_value': 10.0, 'pip_size': 0.0001, 'rsi_ob': 70, 'rsi_os': 30},
    'USD_JPY': {'tp_rr': 1.5, 'atr_mult': 1.0, 'ema': 50, 'pip_value': 9.0, 'pip_size': 0.01, 'rsi_ob': 70, 'rsi_os': 30},
    'USD_CHF': {'tp_rr': 1.5, 'atr_mult': 1.0, 'ema': 50, 'pip_value': 11.0, 'pip_size': 0.0001, 'rsi_ob': 70, 'rsi_os': 30},
    'AUD_USD': {'tp_rr': 1.5, 'atr_mult': 1.0, 'ema': 50, 'pip_value': 10.0, 'pip_size': 0.0001, 'rsi_ob': 70, 'rsi_os': 30},
    'NZD_USD': {'tp_rr': 1.5, 'atr_mult': 1.0, 'ema': 50, 'pip_value': 10.0, 'pip_size': 0.0001, 'rsi_ob': 70, 'rsi_os': 30},
    'USD_CAD': {'tp_rr': 1.5, 'atr_mult': 1.0, 'ema': 50, 'pip_value': 7.5, 'pip_size': 0.0001, 'rsi_ob': 70, 'rsi_os': 30},
    'EUR_GBP': {'tp_rr': 1.5, 'atr_mult': 1.0, 'ema': 50, 'pip_value': 12.5, 'pip_size': 0.0001, 'rsi_ob': 70, 'rsi_os': 30},
    'EUR_JPY': {'tp_rr': 1.5, 'atr_mult': 1.0, 'ema': 50, 'pip_value': 9.0, 'pip_size': 0.01, 'rsi_ob': 70, 'rsi_os': 30},
    'GBP_JPY': {'tp_rr': 1.5, 'atr_mult': 1.0, 'ema': 50, 'pip_value': 9.0, 'pip_size': 0.01, 'rsi_ob': 70, 'rsi_os': 30},
    'XAU_USD': {'tp_rr': 1.5, 'atr_mult': 1.0, 'ema': 50, 'pip_value': 1.0, 'pip_size': 0.01, 'rsi_ob': 70, 'rsi_os': 30},
    'BTC_USD': {'tp_rr': 1.5, 'atr_mult': 1.0, 'ema': 50, 'pip_value': 1.0, 'pip_size': 1.0, 'rsi_ob': 70, 'rsi_os': 30},
    'ETH_USD': {'tp_rr': 1.5, 'atr_mult': 1.0, 'ema': 50, 'pip_value': 1.0, 'pip_size': 0.01, 'rsi_ob': 70, 'rsi_os': 30},
    'SPX500_USD': {'tp_rr': 1.5, 'atr_mult': 1.0, 'ema': 50, 'pip_value': 1.0, 'pip_size': 0.1, 'rsi_ob': 70, 'rsi_os': 30},
    'NAS100_USD': {'tp_rr': 1.5, 'atr_mult': 1.0, 'ema': 50, 'pip_value': 1.0, 'pip_size': 0.1, 'rsi_ob': 70, 'rsi_os': 30},
}

ASSET_CONFIGS_AGGRESSIVE = {
    'EUR_USD': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 10.0, 'pip_size': 0.0001, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'GBP_USD': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 10.0, 'pip_size': 0.0001, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'USD_JPY': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 9.0, 'pip_size': 0.01, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'USD_CHF': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 11.0, 'pip_size': 0.0001, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'AUD_USD': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 10.0, 'pip_size': 0.0001, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'NZD_USD': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 10.0, 'pip_size': 0.0001, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'USD_CAD': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 7.5, 'pip_size': 0.0001, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'EUR_GBP': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 12.5, 'pip_size': 0.0001, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'EUR_JPY': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 9.0, 'pip_size': 0.01, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'GBP_JPY': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 9.0, 'pip_size': 0.01, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'EUR_CHF': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 11.0, 'pip_size': 0.0001, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'EUR_AUD': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 10.0, 'pip_size': 0.0001, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'EUR_CAD': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 7.5, 'pip_size': 0.0001, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'GBP_CHF': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 11.0, 'pip_size': 0.0001, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'GBP_AUD': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 10.0, 'pip_size': 0.0001, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'GBP_CAD': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 7.5, 'pip_size': 0.0001, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'AUD_JPY': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 9.0, 'pip_size': 0.01, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'AUD_NZD': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 10.0, 'pip_size': 0.0001, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'AUD_CAD': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 7.5, 'pip_size': 0.0001, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'NZD_JPY': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 9.0, 'pip_size': 0.01, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'CAD_JPY': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 9.0, 'pip_size': 0.01, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'XAU_USD': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 1.0, 'pip_size': 0.01, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'XAG_USD': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 5.0, 'pip_size': 0.001, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'BTC_USD': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 1.0, 'pip_size': 1.0, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'ETH_USD': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 1.0, 'pip_size': 0.01, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'SPX500_USD': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 1.0, 'pip_size': 0.1, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'NAS100_USD': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 1.0, 'pip_size': 0.1, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'US30_USD': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 1.0, 'pip_size': 1.0, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
    'WTICO_USD': {'tp_rr': 5.0, 'atr_mult': 0.4, 'ema': 50, 'pip_value': 10.0, 'pip_size': 0.01, 'rsi_ob': 55, 'rsi_os': 45, 'min_confluence': 0},
}

ASSET_CONFIGS = ASSET_CONFIGS_AGGRESSIVE

ASSET_FEES = {
    'EUR_USD': {'spread_pips': 1.0, 'commission': 7.0},
    'GBP_USD': {'spread_pips': 1.2, 'commission': 7.0},
    'USD_JPY': {'spread_pips': 1.0, 'commission': 7.0},
    'USD_CHF': {'spread_pips': 1.5, 'commission': 7.0},
    'AUD_USD': {'spread_pips': 1.2, 'commission': 7.0},
    'NZD_USD': {'spread_pips': 1.5, 'commission': 7.0},
    'USD_CAD': {'spread_pips': 1.5, 'commission': 7.0},
    'EUR_GBP': {'spread_pips': 1.5, 'commission': 7.0},
    'EUR_JPY': {'spread_pips': 1.5, 'commission': 7.0},
    'GBP_JPY': {'spread_pips': 2.0, 'commission': 7.0},
    'EUR_CHF': {'spread_pips': 1.5, 'commission': 7.0},
    'EUR_AUD': {'spread_pips': 2.0, 'commission': 7.0},
    'EUR_CAD': {'spread_pips': 2.0, 'commission': 7.0},
    'GBP_CHF': {'spread_pips': 2.0, 'commission': 7.0},
    'GBP_AUD': {'spread_pips': 2.5, 'commission': 7.0},
    'GBP_CAD': {'spread_pips': 2.5, 'commission': 7.0},
    'AUD_JPY': {'spread_pips': 1.5, 'commission': 7.0},
    'AUD_NZD': {'spread_pips': 2.0, 'commission': 7.0},
    'AUD_CAD': {'spread_pips': 2.0, 'commission': 7.0},
    'NZD_JPY': {'spread_pips': 2.0, 'commission': 7.0},
    'CAD_JPY': {'spread_pips': 2.0, 'commission': 7.0},
    'XAU_USD': {'spread_pips': 25.0, 'commission': 6.0},
    'XAG_USD': {'spread_pips': 2.0, 'commission': 5.0},
    'BTC_USD': {'fee_pct': 0.20},
    'ETH_USD': {'fee_pct': 0.20},
    'SPX500_USD': {'spread_points': 0.5, 'commission': 0.0},
    'NAS100_USD': {'spread_points': 1.0, 'commission': 0.0},
    'US30_USD': {'spread_points': 2.0, 'commission': 0.0},
    'WTICO_USD': {'spread_pips': 3.0, 'commission': 0.0},
}


def calculate_trade_fee(symbol: str, lot_size: float, entry_price: float) -> float:
    if symbol not in ASSET_FEES:
        return 0.0
    
    fee_config = ASSET_FEES[symbol]
    
    if 'fee_pct' in fee_config:
        notional = entry_price * lot_size
        return notional * (fee_config['fee_pct'] / 100)
    
    if symbol not in ASSET_CONFIGS:
        return 0.0
    
    asset_cfg = ASSET_CONFIGS[symbol]
    pip_value = asset_cfg.get('pip_value', 10.0)
    
    if 'spread_pips' in fee_config:
        spread_cost = fee_config['spread_pips'] * pip_value * lot_size
    elif 'spread_points' in fee_config:
        spread_cost = fee_config['spread_points'] * lot_size
    else:
        spread_cost = 0.0
    
    commission = fee_config.get('commission', 0.0) * lot_size
    
    return spread_cost + commission


@dataclass
class ChallengeState:
    starting_balance: float = 10000.0
    current_balance: float = 10000.0
    current_equity: float = 10000.0
    daily_starting_equity: float = 10000.0
    step: int = 1
    profitable_days: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    trades_today: int = 0
    daily_pnl: float = 0.0
    current_date: str = ""
    passed_step1: bool = False
    passed_step2: bool = False
    failed: bool = False
    fail_reason: str = ""
    step1_days: int = 0
    step2_days: int = 0


@dataclass
class TradeSignal:
    symbol: str
    direction: str
    entry: float
    stop_loss: float
    take_profit: float
    lot_size: float
    risk_usd: float
    potential_profit: float
    risk_reward: float
    confluence: List[str]
    timestamp: str


def fetch_candles(symbol: str, count: int = 2000, granularity: str = 'H4', 
                  from_time: str = None, to_time: str = None) -> List[Dict]:
    url = f'{BASE_URL}/v3/instruments/{symbol}/candles'
    params = {'granularity': granularity, 'price': 'MBA'}
    
    if from_time and to_time:
        params['from'] = from_time
        params['to'] = to_time
    else:
        params['count'] = count
    
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code == 200:
            data = response.json()
            return [{'time': c.get('time'), 
                     'open': float(c.get('mid', {}).get('o', 0)),
                     'high': float(c.get('mid', {}).get('h', 0)),
                     'low': float(c.get('mid', {}).get('l', 0)),
                     'close': float(c.get('mid', {}).get('c', 0))}
                    for c in data.get('candles', []) if c.get('complete')]
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
    return []


def calculate_ema(closes: List[float], period: int) -> List[float]:
    if len(closes) < period:
        return []
    ema = [sum(closes[:period]) / period]
    mult = 2 / (period + 1)
    for i in range(period, len(closes)):
        ema.append(closes[i] * mult + ema[-1] * (1 - mult))
    return ema


def calculate_atr(candles: List[Dict], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 0.0001
    trs = []
    for i in range(1, len(candles)):
        tr = max(candles[i]['high'] - candles[i]['low'],
                 abs(candles[i]['high'] - candles[i-1]['close']),
                 abs(candles[i]['low'] - candles[i-1]['close']))
        trs.append(tr)
    return sum(trs[-period:]) / period


def calculate_rsi(closes: List[float], period: int = 14) -> List[float]:
    """Calculate RSI indicator."""
    if len(closes) < period + 1:
        return []
    
    rsi_values = []
    gains = []
    losses = []
    
    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        gains.append(max(0, change))
        losses.append(max(0, -change))
    
    if len(gains) < period:
        return []
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    for i in range(period - 1):
        rsi_values.append(50)
    
    if avg_loss == 0:
        rsi_values.append(100)
    else:
        rs = avg_gain / avg_loss
        rsi_values.append(100 - (100 / (1 + rs)))
    
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rsi_values.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))
    
    return rsi_values


def find_order_blocks(candles: List[Dict]) -> List[Dict]:
    obs = []
    for i in range(60, len(candles) - 1):
        curr, next_bar = candles[i], candles[i + 1]
        curr_body = abs(curr['close'] - curr['open'])
        next_body = abs(next_bar['close'] - next_bar['open'])
        if next_body > curr_body * 1.5:
            if next_bar['close'] < next_bar['open'] and curr['close'] > curr['open']:
                obs.append({'type': 'bearish', 'high': curr['high'], 'low': curr['low'], 'idx': i})
            elif next_bar['close'] > next_bar['open'] and curr['close'] < curr['open']:
                obs.append({'type': 'bullish', 'high': curr['high'], 'low': curr['low'], 'idx': i})
    return obs


def find_fair_value_gaps(candles: List[Dict]) -> List[Dict]:
    fvgs = []
    for i in range(2, len(candles)):
        c1, c3 = candles[i - 2], candles[i]
        if c3['low'] > c1['high']:
            fvgs.append({'type': 'bullish', 'high': c3['low'], 'low': c1['high'], 'idx': i})
        elif c3['high'] < c1['low']:
            fvgs.append({'type': 'bearish', 'high': c1['low'], 'low': c3['high'], 'idx': i})
    return fvgs


def find_liquidity_sweeps(candles: List[Dict], lookback: int = 30) -> List[Dict]:
    sweeps = []
    for i in range(lookback + 1, len(candles)):
        curr = candles[i]
        highs = [candles[j]['high'] for j in range(i - lookback, i)]
        lows = [candles[j]['low'] for j in range(i - lookback, i)]
        if curr['high'] > max(highs) and curr['close'] < max(highs):
            sweeps.append({'idx': i, 'dir': 'bearish'})
        elif curr['low'] < min(lows) and curr['close'] > min(lows):
            sweeps.append({'idx': i, 'dir': 'bullish'})
    return sweeps


def calculate_lot_size(risk_usd: float, sl_pips: float, pip_value: float) -> float:
    """Calculate lot size based on risk and stop loss distance."""
    if sl_pips <= 0 or pip_value <= 0:
        return 0.01
    lot_size = risk_usd / (sl_pips * pip_value)
    lot_size = max(0.01, min(lot_size, 10.0))
    return round(lot_size, 2)


def calculate_sl_pips(entry: float, sl: float, pip_size: float) -> float:
    """Calculate stop loss distance in pips."""
    return abs(entry - sl) / pip_size


def check_daily_drawdown(state: ChallengeState, potential_loss: float) -> bool:
    """Check if trade would violate daily drawdown limit."""
    max_daily_loss = state.daily_starting_equity * (CHALLENGE_CONFIG['daily_drawdown_pct'] / 100)
    return (state.daily_pnl - potential_loss) >= -max_daily_loss


def check_max_drawdown(state: ChallengeState, potential_loss: float) -> bool:
    """Check if trade would violate max drawdown limit."""
    min_equity = state.starting_balance * (1 - CHALLENGE_CONFIG['max_drawdown_pct'] / 100)
    return (state.current_balance - potential_loss) >= min_equity


def find_swing_points(candles: List[Dict], lookback: int = 5) -> Tuple[List[Dict], List[Dict]]:
    """Find swing highs and lows for S/R identification."""
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(candles) - lookback):
        is_swing_high = True
        is_swing_low = True
        
        for j in range(1, lookback + 1):
            if candles[i]['high'] <= candles[i - j]['high'] or candles[i]['high'] <= candles[i + j]['high']:
                is_swing_high = False
            if candles[i]['low'] >= candles[i - j]['low'] or candles[i]['low'] >= candles[i + j]['low']:
                is_swing_low = False
        
        if is_swing_high:
            swing_highs.append({'idx': i, 'price': candles[i]['high'], 'time': candles[i]['time']})
        if is_swing_low:
            swing_lows.append({'idx': i, 'price': candles[i]['low'], 'time': candles[i]['time']})
    
    return swing_highs, swing_lows


def get_htf_bias(symbol: str, target_date: str) -> Dict:
    """Get Higher Timeframe bias from Weekly/Daily structure."""
    try:
        daily_candles = fetch_candles(symbol, count=100, granularity='D')
        weekly_candles = fetch_candles(symbol, count=30, granularity='W')
        
        if len(daily_candles) < 30 or len(weekly_candles) < 10:
            return {'bias': 'NEUTRAL', 'zones': []}
        
        d_highs, d_lows = find_swing_points(daily_candles, lookback=3)
        w_highs, w_lows = find_swing_points(weekly_candles, lookback=2)
        
        w_structure = detect_market_structure(w_highs, w_lows, len(weekly_candles) - 1)
        d_structure = detect_market_structure(d_highs, d_lows, len(daily_candles) - 1)
        
        if w_structure['bias'] == 'BULLISH' and d_structure['bias'] == 'BULLISH':
            bias = 'STRONG_BULLISH'
        elif w_structure['bias'] == 'BEARISH' and d_structure['bias'] == 'BEARISH':
            bias = 'STRONG_BEARISH'
        elif w_structure['bias'] == 'BULLISH' or d_structure['bias'] == 'BULLISH':
            bias = 'BULLISH'
        elif w_structure['bias'] == 'BEARISH' or d_structure['bias'] == 'BEARISH':
            bias = 'BEARISH'
        else:
            bias = 'NEUTRAL'
        
        daily_demand = find_demand_zones(daily_candles, d_lows)
        daily_supply = find_supply_zones(daily_candles, d_highs)
        
        zones = []
        for z in daily_demand[-5:]:
            zones.append({'type': 'demand', 'high': z['high'], 'low': z['low'], 'strength': z['strength']})
        for z in daily_supply[-5:]:
            zones.append({'type': 'supply', 'high': z['high'], 'low': z['low'], 'strength': z['strength']})
        
        return {'bias': bias, 'zones': zones, 'daily_structure': d_structure, 'weekly_structure': w_structure}
    except Exception:
        return {'bias': 'NEUTRAL', 'zones': []}


def detect_market_structure(swing_highs: List[Dict], swing_lows: List[Dict], current_idx: int) -> Dict:
    """Detect market structure: HH/HL (bullish) or LH/LL (bearish)."""
    recent_highs = [h for h in swing_highs if h['idx'] < current_idx][-4:]
    recent_lows = [l for l in swing_lows if l['idx'] < current_idx][-4:]
    
    if len(recent_highs) < 2 or len(recent_lows) < 2:
        return {'bias': 'NEUTRAL', 'structure': []}
    
    hh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i]['price'] > recent_highs[i-1]['price'])
    hl_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i]['price'] > recent_lows[i-1]['price'])
    
    lh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i]['price'] < recent_highs[i-1]['price'])
    ll_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i]['price'] < recent_lows[i-1]['price'])
    
    if hh_count >= 1 and hl_count >= 1:
        return {'bias': 'BULLISH', 'structure': ['HH', 'HL'], 'last_low': recent_lows[-1] if recent_lows else None}
    elif lh_count >= 1 and ll_count >= 1:
        return {'bias': 'BEARISH', 'structure': ['LH', 'LL'], 'last_high': recent_highs[-1] if recent_highs else None}
    else:
        return {'bias': 'NEUTRAL', 'structure': []}


def find_demand_zones(candles: List[Dict], swing_lows: List[Dict]) -> List[Dict]:
    """Find demand zones (buy zones) at significant lows followed by impulse moves up."""
    zones = []
    for low in swing_lows:
        idx = low['idx']
        if idx + 3 >= len(candles):
            continue
        
        impulse_move = candles[idx + 3]['close'] - candles[idx]['low']
        range_sum = sum(c['high'] - c['low'] for c in candles[max(0, idx-10):idx])
        avg_range = range_sum / 10 if idx >= 10 and range_sum > 0 else 0.0001
        
        if avg_range > 0 and impulse_move > avg_range * 2:
            zone_high = candles[idx]['open'] if candles[idx]['close'] > candles[idx]['open'] else candles[idx]['close']
            zone_low = candles[idx]['low']
            zones.append({'idx': idx, 'high': zone_high, 'low': zone_low, 'type': 'demand', 'strength': impulse_move / avg_range})
    
    return zones


def find_supply_zones(candles: List[Dict], swing_highs: List[Dict]) -> List[Dict]:
    """Find supply zones (sell zones) at significant highs followed by impulse moves down."""
    zones = []
    for high in swing_highs:
        idx = high['idx']
        if idx + 3 >= len(candles):
            continue
        
        impulse_move = candles[idx]['high'] - candles[idx + 3]['close']
        range_sum = sum(c['high'] - c['low'] for c in candles[max(0, idx-10):idx])
        avg_range = range_sum / 10 if idx >= 10 and range_sum > 0 else 0.0001
        
        if avg_range > 0 and impulse_move > avg_range * 2:
            zone_low = candles[idx]['open'] if candles[idx]['close'] < candles[idx]['open'] else candles[idx]['close']
            zone_high = candles[idx]['high']
            zones.append({'idx': idx, 'high': zone_high, 'low': zone_low, 'type': 'supply', 'strength': impulse_move / avg_range})
    
    return zones


def check_3_candle_rule(candles: List[Dict], zone_high: float, zone_low: float, direction: str, start_idx: int) -> bool:
    """Check if 3 consecutive candles hold inside the zone (V3 entry confirmation)."""
    if start_idx + 3 >= len(candles):
        return False
    
    consecutive_holds = 0
    for j in range(3):
        c = candles[start_idx + j]
        if direction == 'LONG':
            if c['close'] >= zone_low and c['low'] >= zone_low * 0.998:
                consecutive_holds += 1
            else:
                consecutive_holds = 0
        else:
            if c['close'] <= zone_high and c['high'] <= zone_high * 1.002:
                consecutive_holds += 1
            else:
                consecutive_holds = 0
    
    return consecutive_holds >= 3


def find_structure_target(swing_highs: List[Dict], swing_lows: List[Dict], direction: str, entry_idx: int, entry_price: float) -> Optional[float]:
    """Find structure-based TP (prior swing high/low) instead of Fib extensions."""
    if direction == 'LONG':
        future_targets = [h['price'] for h in swing_highs if h['idx'] < entry_idx and h['price'] > entry_price]
        if future_targets:
            return max(future_targets[-3:]) if len(future_targets) >= 3 else max(future_targets)
    else:
        future_targets = [l['price'] for l in swing_lows if l['idx'] < entry_idx and l['price'] < entry_price]
        if future_targets:
            return min(future_targets[-3:]) if len(future_targets) >= 3 else min(future_targets)
    return None


def load_candles_with_fallback(symbol: str, granularity: str = 'H4') -> List[Dict]:
    """Load candles from CSV if available, otherwise use OANDA API."""
    try:
        from data_loader import load_ohlcv_from_csv, df_to_candle_list
        df = load_ohlcv_from_csv(symbol)
        candles = df_to_candle_list(df)
        for c in candles:
            if isinstance(c['time'], str):
                pass
            else:
                c['time'] = c['time'].strftime('%Y-%m-%dT%H:%M:%S')
        return candles
    except (FileNotFoundError, Exception):
        return fetch_candles(symbol, count=5000, granularity=granularity)


def run_challenge_backtest(month: int, year: int) -> Dict:
    """Run backtest using V3 HTF Confluence Strategy with Archer EMA methodology."""
    
    target_month = f"{year}-{month:02d}"
    
    all_signals = []
    
    for symbol, config in ASSET_CONFIGS.items():
        candles = load_candles_with_fallback(symbol, granularity='H4')
        if len(candles) < 200:
            continue
        
        pip_value = config['pip_value']
        pip_size = config['pip_size']
        
        closes = [c['close'] for c in candles]
        ema_21 = calculate_ema(closes, 21)
        ema_86 = calculate_ema(closes, 86)
        
        swing_highs, swing_lows = find_swing_points(candles, lookback=5)
        demand_zones = find_demand_zones(candles, swing_lows)
        supply_zones = find_supply_zones(candles, swing_highs)
        
        last_signal_idx = 0
        cooldown = 3
        
        for i in range(100, len(candles) - 4):
            candle_month = candles[i]['time'][:7]
            if candle_month != target_month:
                continue
            
            if i - last_signal_idx < cooldown:
                continue
            
            price = candles[i]['close']
            
            ema21_idx = i - 21
            ema86_idx = i - 86
            
            if ema21_idx < 0 or ema86_idx < 0:
                continue
            if ema21_idx >= len(ema_21) or ema86_idx >= len(ema_86):
                continue
            
            current_ema21 = ema_21[ema21_idx]
            current_ema86 = ema_86[ema86_idx]
            prev_ema21 = ema_21[ema21_idx - 1] if ema21_idx > 0 else current_ema21
            prev_ema86 = ema_86[ema86_idx - 1] if ema86_idx > 0 else current_ema86
            
            atr = calculate_atr(candles[max(0, i-20):i+1])
            
            structure = detect_market_structure(swing_highs, swing_lows, i)
            
            ema_bullish = current_ema21 > current_ema86 and prev_ema21 <= prev_ema86
            ema_bearish = current_ema21 < current_ema86 and prev_ema21 >= prev_ema86
            ema_bull_trend = current_ema21 > current_ema86
            ema_bear_trend = current_ema21 < current_ema86
            
            trend = None
            confluence = []
            
            active_demand = [z for z in demand_zones if z['idx'] < i and i - z['idx'] < 60 and z['low'] <= price <= z['high'] * 1.01]
            active_supply = [z for z in supply_zones if z['idx'] < i and i - z['idx'] < 60 and z['low'] * 0.99 <= price <= z['high']]
            
            if structure['bias'] == 'BULLISH' and ema_bull_trend:
                if active_demand:
                    best_zone = max(active_demand, key=lambda x: x['strength'])
                    if check_3_candle_rule(candles, best_zone['high'], best_zone['low'], 'LONG', i):
                        trend = 'LONG'
                        confluence.append('HTF_BULLISH')
                        confluence.append('DEMAND_ZONE')
                        confluence.append('EMA_TREND')
                        confluence.append('3_CANDLE_HOLD')
                elif ema_bullish:
                    trend = 'LONG'
                    confluence.append('HTF_BULLISH')
                    confluence.append('EMA_CROSS')
                elif price < current_ema21 < current_ema86 * 1.005:
                    trend = 'LONG'
                    confluence.append('HTF_BULLISH')
                    confluence.append('EMA_PULLBACK')
            
            elif structure['bias'] == 'BEARISH' and ema_bear_trend:
                if active_supply:
                    best_zone = max(active_supply, key=lambda x: x['strength'])
                    if check_3_candle_rule(candles, best_zone['high'], best_zone['low'], 'SHORT', i):
                        trend = 'SHORT'
                        confluence.append('HTF_BEARISH')
                        confluence.append('SUPPLY_ZONE')
                        confluence.append('EMA_TREND')
                        confluence.append('3_CANDLE_HOLD')
                elif ema_bearish:
                    trend = 'SHORT'
                    confluence.append('HTF_BEARISH')
                    confluence.append('EMA_CROSS')
                elif price > current_ema21 > current_ema86 * 0.995:
                    trend = 'SHORT'
                    confluence.append('HTF_BEARISH')
                    confluence.append('EMA_PULLBACK')
            
            if trend is None:
                if ema_bullish and price > current_ema21:
                    trend = 'LONG'
                    confluence.append('EMA_CROSS_ENTRY')
                elif ema_bearish and price < current_ema21:
                    trend = 'SHORT'
                    confluence.append('EMA_CROSS_ENTRY')
            
            if trend is None and ema_bull_trend:
                if price > current_ema21 and price < current_ema21 * 1.02:
                    trend = 'LONG'
                    confluence.append('EMA_TREND_CONT')
            
            if trend is None and ema_bear_trend:
                if price < current_ema21 and price > current_ema21 * 0.98:
                    trend = 'SHORT'
                    confluence.append('EMA_TREND_CONT')
            
            if trend is None:
                candle = candles[i]
                body = abs(candle['close'] - candle['open'])
                total_range = candle['high'] - candle['low']
                if total_range > 0 and body / total_range > 0.7:
                    if candle['close'] > candle['open'] and ema_bull_trend:
                        trend = 'LONG'
                        confluence.append('STRONG_CANDLE')
                    elif candle['close'] < candle['open'] and ema_bear_trend:
                        trend = 'SHORT'
                        confluence.append('STRONG_CANDLE')
            
            if trend is None:
                continue
            
            entry = price
            
            structure_tp = find_structure_target(swing_highs, swing_lows, trend, i, entry)
            
            if trend == 'LONG':
                sl = entry - atr * 0.8
                if structure_tp and structure_tp > entry:
                    tp = structure_tp
                else:
                    tp = entry + atr * 3.0
            else:
                sl = entry + atr * 0.8
                if structure_tp and structure_tp < entry:
                    tp = structure_tp
                else:
                    tp = entry - atr * 3.0
            
            rr_ratio = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
            if rr_ratio < 1.5:
                continue
            
            exit_reason = 'OPEN'
            for j in range(i + 1, min(i + 80, len(candles))):
                bar = candles[j]
                if trend == 'LONG':
                    if bar['low'] <= sl:
                        exit_reason = 'SL'
                        break
                    if bar['high'] >= tp:
                        exit_reason = 'TP'
                        break
                else:
                    if bar['high'] >= sl:
                        exit_reason = 'SL'
                        break
                    if bar['low'] <= tp:
                        exit_reason = 'TP'
                        break
            
            if exit_reason == 'OPEN':
                continue
            
            last_signal_idx = i
            sl_pips = calculate_sl_pips(entry, sl, pip_size)
            
            all_signals.append({
                'time': candles[i]['time'],
                'date': candles[i]['time'][:10],
                'symbol': symbol,
                'direction': trend,
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'tp_rr': rr_ratio,
                'sl_pips': sl_pips,
                'pip_value': pip_value,
                'result': exit_reason,
                'confluence': confluence
            })
    
    all_signals.sort(key=lambda x: x['time'])
    
    state = ChallengeState()
    all_trades = []
    daily_results = {}
    processed_dates = set()
    step1_pass_date = None
    step2_pass_date = None
    total_fees = 0.0
    
    for sig in all_signals:
        if state.failed:
            break
        if state.passed_step2:
            break
        
        candle_date = sig['date']
        
        if candle_date not in processed_dates:
            if state.current_date and state.daily_pnl > 0:
                state.profitable_days += 1
            
            if state.current_date:
                daily_results[state.current_date] = {
                    'pnl': state.daily_pnl,
                    'trades': state.trades_today,
                    'balance': state.current_balance
                }
            
            processed_dates.add(candle_date)
            state.current_date = candle_date
            state.daily_starting_equity = max(state.current_balance, state.current_equity)
            state.daily_pnl = 0.0
            state.trades_today = 0
        
        if state.trades_today >= CHALLENGE_CONFIG['max_trades_per_day']:
            continue
        
        risk_usd = state.current_balance * (CHALLENGE_CONFIG['risk_per_trade_pct'] / 100)
        lot_size = calculate_lot_size(risk_usd, sig['sl_pips'], sig['pip_value'])
        potential_profit = risk_usd * sig['tp_rr']
        
        fee = calculate_trade_fee(sig['symbol'], lot_size, sig['entry'])
        
        if not check_daily_drawdown(state, risk_usd):
            continue
        if not check_max_drawdown(state, risk_usd):
            continue
        
        if sig['result'] == 'TP':
            gross_pnl = potential_profit
        else:
            gross_pnl = -risk_usd
        
        pnl = gross_pnl - fee
        total_fees += fee
        
        state.current_balance += pnl
        state.current_equity = state.current_balance
        state.daily_pnl += pnl
        state.trades_today += 1
        state.total_trades += 1
        if gross_pnl > 0:
            state.winning_trades += 1
        
        trade = {
            'trade_num': state.total_trades,
            'symbol': sig['symbol'],
            'date': candle_date,
            'time': sig['time'],
            'direction': sig['direction'],
            'entry': round(sig['entry'], 5),
            'stop_loss': round(sig['sl'], 5),
            'take_profit': round(sig['tp'], 5),
            'lot_size': lot_size,
            'risk_usd': round(risk_usd, 2),
            'result': sig['result'],
            'gross_pnl': round(gross_pnl, 2),
            'fee': round(fee, 2),
            'pnl': round(pnl, 2),
            'balance': round(state.current_balance, 2),
            'confluence': '+'.join(sig['confluence']),
            'step': state.step
        }
        all_trades.append(trade)
        
        min_equity = 10000 * (1 - CHALLENGE_CONFIG['max_drawdown_pct'] / 100)
        if state.current_balance < min_equity:
            state.failed = True
            state.fail_reason = f"Max drawdown hit (balance ${state.current_balance:.2f} < ${min_equity:.2f})"
            break
        
        max_daily_loss = state.daily_starting_equity * (CHALLENGE_CONFIG['daily_drawdown_pct'] / 100)
        if state.daily_pnl < -max_daily_loss:
            state.failed = True
            state.fail_reason = f"Daily drawdown hit (${state.daily_pnl:.2f} loss > ${max_daily_loss:.2f} limit)"
            break
        
        if state.step == 1 and not state.passed_step1:
            target = 10000 * (1 + CHALLENGE_CONFIG['step1_target_pct'] / 100)
            if state.current_balance >= target and state.profitable_days >= CHALLENGE_CONFIG['min_profitable_days']:
                state.passed_step1 = True
                state.step = 2
                step1_pass_date = candle_date
                state.starting_balance = state.current_balance
        elif state.step == 2 and not state.passed_step2:
            target = state.starting_balance * (1 + CHALLENGE_CONFIG['step2_target_pct'] / 100)
            if state.current_balance >= target:
                state.passed_step2 = True
                step2_pass_date = candle_date
    
    if state.current_date and state.daily_pnl > 0:
        state.profitable_days += 1
    if state.current_date:
        daily_results[state.current_date] = {
            'pnl': state.daily_pnl,
            'trades': state.trades_today,
            'balance': state.current_balance
        }
    
    trading_days = len(processed_dates)
    if step1_pass_date:
        step1_days = len([d for d in processed_dates if d <= step1_pass_date])
    else:
        step1_days = trading_days
    
    if step2_pass_date and step1_pass_date:
        step2_days = len([d for d in processed_dates if step1_pass_date < d <= step2_pass_date])
    else:
        step2_days = 0
    
    total_pnl = state.current_balance - 10000
    win_rate = (state.winning_trades / state.total_trades * 100) if state.total_trades > 0 else 0
    
    gross_pnl = sum(t.get('gross_pnl', t['pnl']) for t in all_trades)
    
    return {
        'month': month,
        'year': year,
        'passed_step1': state.passed_step1,
        'passed_step2': state.passed_step2,
        'failed': state.failed,
        'fail_reason': state.fail_reason,
        'step1_days': step1_days,
        'step2_days': step2_days,
        'total_days': trading_days,
        'total_trades': state.total_trades,
        'winning_trades': state.winning_trades,
        'win_rate': win_rate,
        'profitable_days': state.profitable_days,
        'final_balance': state.current_balance,
        'gross_pnl': gross_pnl,
        'total_fees': total_fees,
        'total_pnl': total_pnl,
        'return_pct': (total_pnl / 10000) * 100,
        'trades': all_trades,
        'daily_results': daily_results
    }


def format_challenge_result(result: Dict) -> discord.Embed:
    """Format challenge backtest result as Discord embed."""
    
    month_name = datetime(result['year'], result['month'], 1).strftime('%B %Y')
    
    if result['failed']:
        color = discord.Color.red()
        status = "FAILED"
        title = f"5%ers Challenge - {month_name}"
    elif result['passed_step2']:
        color = discord.Color.green()
        status = "PASSED BOTH STEPS"
        title = f"5%ers Challenge PASSED - {month_name}"
    elif result['passed_step1']:
        color = discord.Color.gold()
        status = "PASSED STEP 1 ONLY"
        title = f"5%ers Challenge - {month_name}"
    else:
        color = discord.Color.orange()
        status = "IN PROGRESS"
        title = f"5%ers Challenge - {month_name}"
    
    embed = discord.Embed(title=title, color=color)
    
    embed.add_field(name="Status", value=status, inline=True)
    embed.add_field(name="Account", value="$10,000", inline=True)
    embed.add_field(name="Final Balance", value=f"${result['final_balance']:,.2f}", inline=True)
    
    if result['passed_step1']:
        embed.add_field(name="Step 1 (8%)", value=f"PASSED in {result['step1_days']} days", inline=True)
    else:
        embed.add_field(name="Step 1 (8%)", value=f"Not passed ({result['step1_days']} days)", inline=True)
    
    if result['passed_step2']:
        embed.add_field(name="Step 2 (5%)", value=f"PASSED in {result['step2_days']} days", inline=True)
    else:
        embed.add_field(name="Step 2 (5%)", value=f"Not passed", inline=True)
    
    total_days = result['step1_days'] + result['step2_days']
    embed.add_field(name="Total Days", value=f"{total_days}", inline=True)
    
    embed.add_field(name="Total Trades", value=str(result['total_trades']), inline=True)
    embed.add_field(name="Win Rate", value=f"{result['win_rate']:.1f}%", inline=True)
    embed.add_field(name="Profitable Days", value=str(result['profitable_days']), inline=True)
    
    gross_str = f"+${result['gross_pnl']:,.2f}" if result['gross_pnl'] >= 0 else f"-${abs(result['gross_pnl']):,.2f}"
    pnl_str = f"+${result['total_pnl']:,.2f}" if result['total_pnl'] >= 0 else f"-${abs(result['total_pnl']):,.2f}"
    embed.add_field(name="Gross P/L", value=gross_str, inline=True)
    embed.add_field(name="Total Fees", value=f"-${result['total_fees']:,.2f}", inline=True)
    embed.add_field(name="Net P/L", value=pnl_str, inline=True)
    embed.add_field(name="Return", value=f"{result['return_pct']:+.1f}%", inline=True)
    
    if result['failed']:
        embed.add_field(name="Fail Reason", value=result['fail_reason'], inline=False)
    
    if result['trades']:
        recent = result['trades'][-5:]
        trades_str = ""
        for t in recent:
            pnl_str = f"+${t['pnl']:.0f}" if t['pnl'] > 0 else f"-${abs(t['pnl']):.0f}"
            trades_str += f"{t['symbol']} {t['direction']} {t['lot_size']}L | {t['result']} {pnl_str}\n"
        embed.add_field(name="Recent Trades", value=f"```{trades_str}```", inline=False)
    
    return embed


def export_challenge_trades_csv(result: Dict, filename: str = None) -> str:
    """Export challenge trades to CSV file."""
    import csv
    
    if filename is None:
        filename = f"challenge_{result['month']:02d}_{result['year']}_trades.csv"
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Trade#', 'Date', 'Symbol', 'Direction', 'Entry', 'Stop Loss', 'Take Profit', 
                        'Lot Size', 'Risk ($)', 'Result', 'Gross P/L', 'Fee', 'Net P/L', 'Balance', 'Confluence', 'Step'])
        
        for t in result['trades']:
            writer.writerow([
                t['trade_num'], t['date'], t['symbol'], t['direction'], t['entry'],
                t['stop_loss'], t['take_profit'], t['lot_size'], t['risk_usd'],
                t['result'], t.get('gross_pnl', t['pnl']), t.get('fee', 0), t['pnl'], 
                t['balance'], t['confluence'], t['step']
            ])
    
    return filename


if __name__ == "__main__":
    print("Testing 5%ers Challenge Backtest...")
    result = run_challenge_backtest(11, 2024)
    
    print(f"\n{'='*60}")
    print(f"5%ERS 10K CHALLENGE - November 2024")
    print(f"{'='*60}")
    print(f"Passed Step 1: {'YES' if result['passed_step1'] else 'NO'} ({result['step1_days']} days)")
    print(f"Passed Step 2: {'YES' if result['passed_step2'] else 'NO'} ({result['step2_days']} days)")
    print(f"Failed: {'YES - ' + result['fail_reason'] if result['failed'] else 'NO'}")
    print(f"Total Trades: {result['total_trades']}")
    print(f"Win Rate: {result['win_rate']:.1f}%")
    print(f"Profitable Days: {result['profitable_days']}")
    print(f"Final Balance: ${result['final_balance']:,.2f}")
    print(f"Total P/L: ${result['total_pnl']:+,.2f} ({result['return_pct']:+.1f}%)")
    
    filename = export_challenge_trades_csv(result)
    print(f"\nTrades exported to: {filename}")
