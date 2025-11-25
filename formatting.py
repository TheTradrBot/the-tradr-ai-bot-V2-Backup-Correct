from __future__ import annotations

from typing import List
from strategy import ScanResult


def _yes_no_from_text(ok_text: str, note: str) -> str:
    """
    Helper to decide Y/N flags based on note text.
    We keep it simple + robust: just check for key phrases.
    """
    return "Y" if ok_text in note else "N"


def format_scan_group(group_name: str, results: List[ScanResult]) -> str:
    """
    Format a group of ScanResult objects into a compact multi-line summary
    that still shows the full 7 Blueprint pillars:

    - htf  = HTF bias
    - loc  = Location (S/R + supply/demand, top/bottom of range)
    - fib  = Weekly/Daily golden pocket
    - liq  = Liquidity (external pool, equal highs/lows, sweeps)
    - struct = Weekly/Daily structure + Daily frameworks
    - 4H   = 4H confirmation (BOS / continuation)
    - rr   = R/R quality + TP1–TP5 structure
    """
    if not results:
        return f"📊 {group_name} scan\n_No instruments to show._"

    # Sort by symbol for stable ordering
    results = sorted(results, key=lambda r: r.symbol)

    lines: List[str] = []
    lines.append(f"📊 {group_name} scan")

    for res in results:
        status_tag = (
            "ACTIVE" if res.status == "active"
            else "INP" if res.status == "in_progress"
            else "SCAN"
        )

        # Infer flags from the rich notes
        htf_flag = "Y" if ("HTF trend alignment" in res.htf_bias
                           or "HTF reversal bias" in res.htf_bias) else "N"

        loc_flag = "Y" if (
            "price in lower" in res.location_note
            or "price in upper" in res.location_note
            or "Tier" in res.location_note
            or "supply zone" in res.location_note
            or "demand zone" in res.location_note
        ) else "N"

        fib_flag = "Y" if "price inside golden pocket" in res.fib_note else "N"

        liq_flag = "Y" if (
            "liquidity" in res.liquidity_note
            or "sweep" in res.liquidity_note
        ) else "N"

        struct_flag = "Y" if "Structure supports" in res.structure_note else "N"

        h4_flag = "Y" if "H4: structure aligned" in res.confirmation_note else "N"

        rr_flag = "Y" if "Approx first target" in res.rr_note else "N"

        lines.append(
            f"{res.symbol} | {res.direction.upper()} | "
            f"{res.confluence_score}/7 ({status_tag}) – "
            f"htf={htf_flag}, loc={loc_flag}, fib={fib_flag}, "
            f"liq={liq_flag}, struct={struct_flag}, 4H={h4_flag}, rr={rr_flag}"
        )

    return "\n".join(lines)
