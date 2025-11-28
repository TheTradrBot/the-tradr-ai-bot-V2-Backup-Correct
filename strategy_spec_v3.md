# Blueprint HTF Confluence Strategy – Spec (v3)

---

## 1. High-level overview

### 1.1 Type of trader & holding period

- Trades are based on **Daily frameworks** with:
  - **Daily** as the **main framework/setup chart**
  - **4H** as the **execution/entry chart**
  - **Weekly & Monthly** for macro confluence
- Typical holding time: **several days up to ~1–2 weeks**
- The strategy is **not** for:
  - High-frequency scalping
  - Long-term multi-month investing
- Optional filter: avoid **new entries on Friday**.

---

### 1.2 Markets

The framework is generic and can be applied to:

- **FX**  
  - All main Oanda FX pairs (majors and key minors), excluding exotics
- **Commodities**  
  - XAUUSD (Gold), XAGUSD (Silver), major energies (e.g. Crude, Brent)
- **Indices**  
  - e.g. SPX500, NAS100, US30 (depending on broker)
- **Crypto**  
  - Liquid majors only if spreads/feeds are acceptable (e.g. BTC, ETH, SOL)

Instrument requirements:

- Reasonably **clean technical behaviour**
- Respect for **HTF S/R**, **structure** and **impulsive legs**

---

### 1.3 Core tools & concepts

The strategy is built from these core components:

- **Support & Resistance (S/R)**
  - Horizontal levels/zones on **Monthly**, **Weekly**, and **Daily**
  - Used to anchor bias, define locations for setups and plan take profits

- **Market Structure**
  - HH / HL vs LH / LL
  - **Break of Structure (BOS)** and **Change of Character (CHOCH)**
  - Emphasis on **breakaway candles** for valid BOS

- **Fibonacci on Daily impulses**
  - Drawn **only on Daily impulses** that form the structural legs of the framework
  - Always **body → wick**
    - Up impulse: low body → high wick
    - Down impulse: high body → low wick
  - Retracements:
    - **0.618, 0.66, 0.796** (main entry zone)
  - Extensions for TPs:
    - **−0.25, −0.68, −1.00, −1.42, −2.00**

- **Structural Frameworks on the Daily chart (framework = Daily only)**
  - **Head & Shoulders (H&S)** – bearish reversal
  - **Inverse Head & Shoulders** – bullish reversal
  - **Bullish N pattern** – bullish continuation
  - **Bearish V pattern** – bearish continuation
  - The **impulses for Fibonacci** always come from:
    - **Daily leg-1 impulse** for N/V continuation
    - **Daily impulse that breaks the neckline (BOS)** for H&S / inverse H&S

- **Supply & Demand / Order Blocks**
  - Last opposing candle/cluster before a strong impulsive move that breaks structure
  - Serve as refined entry zones inside HTF S/R and Daily fib zones

- **Liquidity & Magnetic Price Levels**
  - Equal highs/lows
  - Previous day/week/month highs and lows
  - Key swing highs/lows at HTF
  - These are **magnets and confluence**, **not hard entry requirements**

- **Top-down HTF Confluence**
  - Monthly → Weekly → Daily to define context and bias
  - 4H used **only for entry confirmation** with a strict **3-candle rule**

> The system is a **confluence-based HTF swing framework**:  
> trades are allowed only when **HTF bias, structure, S/R, Daily zones and Fibonacci** align,  
> **optionally supported by liquidity confluence**, and **confirmed on 4H with the 3-candle rule**.

---

## 2. Timeframes

### 2.1 Timeframe stack

- **Monthly (MN)**
  - Draw major **S/R zones**
  - Mark very large swing highs/lows and major equal highs/lows
  - Identify **macro trend** and multi-year liquidity magnets

- **Weekly (W1) – Primary bias timeframe**
  - Draw **weekly S/R** in alignment with Monthly
  - Identify **weekly structural frameworks**:
    - H&S / inverse H&S
    - Bullish N / Bearish V
  - Determine the **primary directional bias** (bullish, bearish, neutral)

- **Daily (D1) – Framework & setup timeframe (always)**
  - Daily is **always** the **framework chart**
  - Tasks on Daily:
    - Classify current leg as **impulse or correction** within the Weekly swing
    - Map **Daily supply/demand (OBs)** and local S/R
    - Identify **Daily structural frameworks** (H&S, inverse H&S, Bullish N, Bearish V)
    - Select **Daily impulses for Fibonacci**:
      - Leg-1 impulse in N/V
      - Neckline-breaking impulse in H&S / inverse H&S
  - Daily produces the **entry zones** used later on 4H

- **4H (H4) – Execution timeframe**
  - **Entry confirmation ONLY**
  - Responsibilities:
    - Monitor how price behaves inside the **Daily entry zone**
    - Apply the **3-candle close rule**:
      - If 3 consecutive 4H candles close inside the Daily zone:
        - For longs: closes hold **inside/above** the Daily demand + fib zone
        - For shorts: closes hold **inside/below** the Daily supply + fib zone
      - **Enter on the close of the 3rd 4H candle**
    - Refine stop placement if needed

---

### 2.2 HTF bias definition

Bias is defined top-down:

1. **Monthly macro context**
   - Evaluate last 3–5 swings:
     - Sequence of HH/HL → macro uptrend
     - Sequence of LH/LL → macro downtrend
   - Check whether price is:
     - Approaching or rejecting **major Monthly S/R or OB**
     - Close to **macro equal highs/lows** or major liquidity clusters

2. **Weekly primary bias**
   - Weekly is the **main bias timeframe**
   - Identify:
     - Current trend (HH/HL or LH/LL)
     - Latest **BOS** with a clear **breakaway candle**
     - Any forming/formed **H&S / inverse H&S** on Weekly
   - Rules:
     - **Bullish bias** if:
       - Clear HH/HL structure AND
       - Last BOS is up with a strong breakaway candle AND
       - Price is above the last key higher low
     - **Bearish bias** if:
       - Clear LH/LL structure AND
       - Last BOS is down with a strong breakaway candle AND
       - Price is below the last key lower high
     - **Neutral/cautious** if:
       - Weekly is choppy around key HTF S/R, or
       - Recent BOS is unclear or quickly reversed

3. **Daily execution bias**
   - Within the Weekly bias:
     - In a **Weekly uptrend**:
       - Daily impulse phase: wait for pullback before longing
       - Daily correction phase: look for **longs** from Daily demand + HTF S/R + fib
     - In a **Weekly downtrend**:
       - Mirror logic for shorts from Daily supply + HTF S/R + fib
   - Daily CHOCH against Weekly trend:
     - May indicate a deeper pullback to HTF demand/supply
     - Or early stage of Weekly reversal **if** at major HTF level + clear framework

---

### 2.3 When not to trade

No trades when:

- **Conflicting bias**
  - Weekly up, Daily strongly down, and price **not** at a major HTF level/framework
  - Weekly down, Daily strongly up, and price **not** at HTF level/framework
- **Mid-range chop**
  - Price stuck in the middle of a broad Monthly/Weekly range with:
    - No nearby HTF S/R
    - No clean supply/demand zone
    - No obvious liquidity magnet above/below
- **Exhausted trend**
  - Price already hit major **extension targets** (e.g. −1.00, −1.42, −2.00)
  - No fresh, clean pullback zones available

---

## 3. Support & Resistance (S/R)

### 3.1 Definitions

- **Support**  
  A zone where price previously found buying interest and bounced.
- **Resistance**  
  A zone where price previously found selling interest and rejected.

Zones, not single lines:

- Define **bands** around key highs/lows rather than razor-thin lines.
- Allow for **wicks/spikes** beyond the zone.

---

### 3.2 S/R by timeframe

- **Monthly**
  - Major reversal points over years
  - "Macro magnets" for price

- **Weekly**
  - Swing points inside Monthly structure
  - Core S/R used for trading bias

- **Daily**
  - Refines Weekly levels
  - Aligns with Daily OBs and fib zones to define entry regions

---

### 3.3 Interaction with other components

- S/R + **Daily OB** + **Daily fib golden pocket** = primary entry zone
- S/R often lines up with **major liquidity pools** (equal highs/lows, previous highs/lows)
- Structural frameworks (H&S, inverse H&S, N/V) are only traded **if they form at/near HTF S/R**

---

## 4. Market structure

### 4.1 Basic elements

- **Higher High (HH)** – new high above previous swing high
- **Higher Low (HL)** – low above previous swing low
- **Lower High (LH)** – lower than previous swing high
- **Lower Low (LL)** – lower than previous swing low

- **Break of Structure (BOS)**
  - Candle body close that decisively breaks a prior important swing high/low
  - Ideally a **breakaway candle** (large body, strong close beyond level)

- **Change of Character (CHOCH)**
  - First BOS **against** the prevailing trend:
    - Uptrend: break below last key HL
    - Downtrend: break above last key LH

---

### 4.2 Structure by timeframe

- **Daily**
  - Main timeframe to track swing structure for trades
  - Identify:
    - Last key HL or LH
    - Recent BOS/CHOCH
    - Whether current leg is **impulse or correction**

- **4H**
  - Used to refine Daily structure
  - Not used to define the main frameworks
  - 4H structure helps with entry timing but must **obey** Daily/Weekly context

---

### 4.3 Trend & bias on Daily

- **Uptrend**
  - At least two HH/HL sequences
  - No recent strong BOS down
  - Key HL still intact

- **Downtrend**
  - At least two LH/LL sequences
  - No recent strong BOS up
  - Key LH still intact

- **Range/transition**
  - Equal highs/lows
  - Alternating BOS in both directions
  - Often coincides with forming H&S / inverse H&S or accumulation/distribution

---

### 4.4 Structure with S/R, supply/demand, liquidity

- At **HTF support** (Monthly/Weekly S/R + demand):
  - Look for signs of downtrend losing power:
    - Failure to make new LL near support
    - CHOCH or BOS up on Daily
    - Inverse H&S or Bullish N forming
    - Optional: sweeps of equal lows / previous lows then strong rejection

- At **HTF resistance** (Monthly/Weekly S/R + supply):
  - Look for uptrend losing power:
    - Failure to make new HH
    - CHOCH or BOS down on Daily
    - H&S or Bearish V forming
    - Optional: sweeps of equal highs / previous highs then rejection

- Structural frameworks on **Daily only**:
  - BOS that completes a framework (e.g. break of neckline) is a higher-probability signal

---

## 5. Supply & Demand zones

### 5.1 Definitions

- **Demand zone (bullish order block)**
  - Last **bearish candle/cluster** before a strong impulsive move up that:
    - Breaks structure (BOS) or
    - Takes significant liquidity
  - Zone boundaries:
    - From open of last bearish candle down to its lowest wick
    - Optionally include small neighbouring candles of same base

- **Supply zone (bearish order block)**
  - Last **bullish candle/cluster** before a strong impulsive move down that:
    - Breaks structure or
    - Sweeps equal highs
  - Zone boundaries:
    - From open of last bullish candle up to its highest wick
    - Optionally include adjacent small candles

---

### 5.2 Valid zone criteria

A zone is **valid** if:

1. Move away from zone is clearly **impulsive**:
   - Large body candles
   - Minimal overlap
   - Ideally a **breakaway candle**

2. The move away causes:
   - A **BOS** on that timeframe or higher, or
   - A clear liquidity grab (equal highs/lows, prior day/week high/low)

3. Zone is located:
   - Near **HTF S/R**, and/or
   - Around a **Fibonacci retracement** of a key Daily swing

---

### 5.3 Fresh vs used zones

- **Fresh zone**
  - Price has not yet returned since the impulsive move
  - Highest-probability reaction

- **First retest**
  - First revisit after the impulsive move
  - Often where we plan entries

- **Used/degraded zone**
  - Zone has been tapped multiple times
  - Each touch reduces reliability

---

### 5.4 Timeframe for zones

- Primary zones for entries are on **Daily**
- Weekly/MN zones provide **context**, not exact entry
- 4H can refine **sub-zones** inside the Daily OB if needed, but the main zone is still derived from D1

---

### 5.5 How zones appear in different contexts

- In trend:
  - Uptrend:
    - Buy from **demand** at or near support
  - Downtrend:
    - Sell from **supply** at or near resistance

- Around BOS/CHOCH:
  - After a BOS, the **origin OB** often becomes the pullback entry zone
  - After CHOCH at HTF level, a new OB marks the early stage of a new trend

---

### 5.6 Use in entries, stops, targets

- **Entries (Daily zone + 4H confirmation)**
  - Look to enter **inside the Daily demand/supply zone**, ideally where:
    - Zone overlaps with the **Daily fib golden pocket (0.618–0.796)** of the correct Daily impulse
    - Price then prints the **4H 3-candle close pattern inside the zone**

- **Stops**
  - Longs:
    - Below the lowest wick of the Daily demand zone
  - Shorts:
    - Above the highest wick of the Daily supply zone
  - Optionally a little beyond obvious local liquidity (equal highs/lows) to avoid easy stop hunts

- **Targets (Daily fib extensions + structure)**
  - Long from demand:
    - Next HTF supply or prior structural high
  - Short from supply:
    - Next HTF demand or prior structural low

---

## 6. Fibonacci

### 6.1 How to draw

- Use **body → wick** on Daily impulses only:
  - **Up impulse**: Anchor from **low body** to **high wick**
  - **Down impulse**: Anchor from **high body** to **low wick**

---

### 6.2 Retracement levels

| Level | Role |
|-------|------|
| 0.382 | Shallow retracement zone start |
| 0.50 | Mid-point |
| 0.618 | Golden pocket low |
| 0.66 | Golden pocket mid |
| 0.786 | Golden pocket high |
| 0.886 | Deep retracement (last resort) |

Primary entry zone: **0.618–0.786** (golden pocket)

---

### 6.3 Extension levels

| Level | Role |
|-------|------|
| −0.25 | First structural target |
| −0.68 | TP2 zone |
| −1.00 | 100% extension / measured move |
| −1.42 | Extended target |
| −2.00 | Full extension |

---

### 6.4 Which impulses to use

- **Bullish N (continuation)**:
  - Fib on **leg-1** (the impulse that breaks previous high)
- **Bearish V (continuation)**:
  - Fib on **leg-1** (the impulse that breaks previous low)
- **Inverse H&S (reversal)**:
  - Fib on the **neckline-breaking impulse** (the move that creates the BOS above neckline)
- **H&S (reversal)**:
  - Fib on the **neckline-breaking impulse** (the move that creates the BOS below neckline)

---

## 7. 4H Confirmation – 3-Candle Rule

### 7.1 Role of 4H

4H is used to:

- Confirm that the **Daily framework (setup)** is being respected
- Provide a **precise, mechanically defined entry trigger**
- Avoid blindly entering at Daily levels without evidence of holding

---

### 7.2 Required 4H signal – new core rule

For both longs and shorts, the **only required 4H trigger** is:

- **Three consecutive 4H candles closing inside the Daily entry zone**, then
- **Enter on the close of the 3rd candle**

Definitions:

- **Daily entry zone (longs)**:
  - Overlap of:
    - Daily **demand** / OB
    - Daily **fib golden pocket (0.618–0.796)** of the correct Daily impulse
    - Ideally HTF S/R support

- **Daily entry zone (shorts)**:
  - Overlap of:
    - Daily **supply** / OB
    - Daily **fib golden pocket (0.618–0.796)** of the correct Daily impulse
    - Ideally HTF S/R resistance

4H rules:

- Longs:
  - 3 consecutive 4H closes **inside and respecting the zone as support**
- Shorts:
  - 3 consecutive 4H closes **inside and respecting the zone as resistance**
- No requirement for 4H BOS/CHOCH or mandatory sweeps;
  - Those can appear as **supporting confluence**, not conditions

---

### 7.3 Optional secondary 4H signals

Not required for entry, but can add confidence:

- Candle patterns:
  - Strong engulfing candle in trade direction
  - Long lower wicks at demand (for longs)
  - Long upper wicks at supply (for shorts)
- Minor structure shifts:
  - Small 4H CHOCH in direction of HTF trend

If the 3-candle rule is met but secondary signals are ugly, risk can be adjusted, but the core rule still stands.

---

## 8. Entry Criteria

### 8.1 LONG setup checklist

A valid LONG must satisfy:

1. **HTF bias**
   - Weekly bias **bullish**, or
   - Weekly neutral but at strong Monthly support with clear bullish Daily framework (inverse H&S, Bullish N)

2. **Location**
   - Price located at/near:
     - Monthly/Weekly **support S/R**
     - **Daily demand / OB** that originated a Daily impulse
     - **0.618–0.796 retracement** of the correct Daily impulse (leg-1 for N, neckline-break impulse for inverse H&S)

3. **Daily market structure**
   - Either HL sequence intact, or
   - CHOCH/BOS up forming at HTF support
   - No recent strong Daily BOS down through key support

4. **Structural framework (if present)**
   - Inverse H&S or Bullish N pattern at HTF level
   - Not mandatory, but increases conviction

5. **4H confirmation – 3-candle rule**
   - Price trades into the Daily zone (demand + fib golden pocket)
   - On 4H:
     - Price prints **3 consecutive 4H closes inside the zone**
   - **Entry:**
     - Go long at the **close of the 3rd 4H candle**

6. **Risk/Reward**
   - Stop placed logically below:
     - Daily demand zone low and relevant swing low
   - First target (prior high or −0.25 extension) offers at least **2R**

---

### 8.2 SHORT setup checklist

Mirror conditions for shorts:

1. **HTF bias**
   - Weekly bias **bearish**, or
   - Weekly neutral at strong Monthly resistance with clear bearish Daily framework (H&S, Bearish V)

2. **Location**
   - At/near:
     - Monthly/Weekly resistance
     - Daily supply / OB originating the Daily impulse down
     - 0.618–0.796 retracement of the correct Daily impulse (leg-1 for V, neckline-break impulse for H&S)

3. **Daily structure**
   - LH/LL structure intact or CHOCH/BOS down at resistance

4. **Framework (if present)**
   - H&S or Bearish V pattern at HTF level

5. **4H confirmation – 3-candle rule**
   - Price trades into the Daily zone (supply + fib golden pocket)
   - On 4H:
     - 3 consecutive closes inside the zone, respecting it as resistance
   - **Entry:**
     - Short at close of the 3rd 4H candle

6. **Risk/Reward**
   - Initial TP (prior low or −0.25 extension) must give at least 2R

---

## 9. Risk Management & Trade Management

### 9.1 Per-trade risk

- Configurable per account, typical range:
  - **0.5%–1.5%** risk per trade
- Lot size is computed from:
  - Risk amount
  - Distance from entry to structural stop

---

### 9.2 Correlation & exposure

- Limit simultaneous risk on strongly correlated pairs:
  - Example: cap total risk at **2–3%** across all EUR-cross longs, etc.
- Avoid over-stacking several trades that are effectively the same macro bet.

---

### 9.3 News & timing filters

- Optional:
  - Avoid fresh entries shortly before high-impact news for that currency/asset.
- Late-week behaviour:
  - Optionally skip new entries on **Friday** or use reduced size.

---

### 9.4 Trade management

- Initial SL:
  - At structural invalidation (beyond Daily zone and key swing)
- TP ladder:
  - TP1 near structure + −0.25
  - TP2 at −0.68
  - TP3 at −1.00
  - Optional extended TPs at −1.42 and −2.00
- Breakeven policy:
  - Optional: move SL to BE after TP1
- Trailing stops:
  - Based on new structure:
    - For longs: trail below higher lows
    - For shorts: trail above lower highs

If price invalidates the structural story (e.g. breaks key swing against the position), consider early exit even if SL not yet hit.

---

## 10. Optimization Goals (for backtesting)

For systematic testing and optimization (per asset, per year):

- **Target number of trades**: ≥60 per year per asset
- **Target win-rate**: 70–100%
- **Target yearly return**: +50% to +400%

These goals are **targets for optimization**, achieved by tuning parameters while respecting the core strategy logic.

---

## 11. Implementation Notes

### 11.1 Single source of truth

- All strategy logic must be in one place (strategy_core.py)
- Both live scanning and backtesting call the same functions
- Parameter changes automatically affect both

### 11.2 Price validation

- Backtest prices must match real OANDA OHLC data
- Validate against reference CSV for EUR/USD
- Entry/exit prices must be within candle high/low

### 11.3 Holiday filtering

- No new trades on major FX holidays
- Skip Christmas, New Year, Easter Monday, etc.

### 11.4 Machine-learning style optimization

- Run backtests for each asset-year
- Compare results to targets
- Adjust parameters within spec boundaries
- Re-run until targets met
