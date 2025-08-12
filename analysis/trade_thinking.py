#!/usr/bin/env python3
"""
Trade Thinking Scorer

Lightweight heuristic rubric to score whether a prediction "thinks like a trader".
This is intended for monitoring/analysis and optional rollout selection; it should
NOT replace the LLM-based outcome reward used for GSPO training.

Scoring (weights sum to 1.0):
  - Ticker/instrument (0.2): mentions a concrete instrument or symbol
  - Direction (0.2): explicit long/short/buy/sell
  - Timeframe (0.1): mentions a near-term window (for next_day) or relevant horizon
  - Entry/exit (0.2): presence of numeric stop/target/entry cues
  - Catalyst linkage (0.2): mentions a named event/factor likely to drive the move
  - Risk specificity (0.1): non-generic downside scenario cues

All checks are simple regex/keyword heuristics; adjust as needed.
"""

from __future__ import annotations

import re
from typing import Optional


WEIGHTS = {
    "instrument": 0.2,
    "direction": 0.2,
    "timeframe": 0.1,
    "entry_exit": 0.2,
    "catalyst": 0.2,
    "risk": 0.1,
}


def score_trade_thinking(text: str, horizon: Optional[str] = None) -> float:
    if not text:
        return 0.0
    t = text.strip()

    score = 0.0

    # Instrument/ticker: uppercase tickers, major ETFs, FX pairs, futures
    instrument_patterns = [
        r"\b[A-Z]{2,5}\b",  # generic uppercase token (catch tickers; noisy but useful)
        r"\bSPY\b|\bQQQ\b|\bTLT\b|\bHYG\b|\bGLD\b|\bUSO\b|\bWTI\b",
        r"\bBTC\b|\bETH\b|\bXAU\b|\bXAG\b",
        r"\bUSD/[A-Z]{3}\b|\b[A-Z]{3}/USD\b|\bEUR/USD\b|\bUSD/JPY\b",
        r"\bS&P\b|\bNasdaq\b|\bTreasur(y|ies)\b",
    ]
    if any(re.search(p, t) for p in instrument_patterns):
        score += WEIGHTS["instrument"]

    # Direction: long/short/buy/sell
    if re.search(r"\b(long|short|buy|sell|overweight|underweight)\b", t, re.IGNORECASE):
        score += WEIGHTS["direction"]

    # Timeframe: next-day or horizon-specific cues
    timeframe_hit = False
    if horizon == "next_day":
        if re.search(r"\b(24\s*-?\s*48h|tomorrow|next day|overnight|1-2\s*days|1\s*day)\b", t, re.IGNORECASE):
            timeframe_hit = True
    elif horizon == "next_month":
        if re.search(r"\b(weeks?|1\s*month|30\s*days)\b", t, re.IGNORECASE):
            timeframe_hit = True
    elif horizon == "next_year":
        if re.search(r"\b(quarters?|12\s*months|year)\b", t, re.IGNORECASE):
            timeframe_hit = True
    if timeframe_hit:
        score += WEIGHTS["timeframe"]

    # Entry/exit: numbers with $/%, or words stop/target/at/around with numbers
    if re.search(r"(stop|target|tp|sl|at|around)\s*(~|≈|=|:)?\s*[$€£]?\d+(\.\d+)?(%|\b)", t, re.IGNORECASE):
        score += WEIGHTS["entry_exit"]
    elif re.search(r"\b\d{1,3}\.?\d?\s*%\b", t):  # percentage
        score += WEIGHTS["entry_exit"]

    # Catalyst: earnings, policy, data prints, deals, guidance
    if re.search(r"\b(earnings|guidance|CPI|PCE|jobs|NFP|inflation|Fed|FOMC|rate\s*(cut|hike|decision)|ECB|BOJ|tariff|deal|merger|acquisition|upgrade|downgrade|headline(s)?)\b", t, re.IGNORECASE):
        score += WEIGHTS["catalyst"]

    # Risk: non-generic downside scenario cues
    if re.search(r"\b(risk|if|unless|downside|stop|invalidated|fails|surprise|hawkish|miss|guide\s*down)\b", t, re.IGNORECASE):
        score += WEIGHTS["risk"]

    # Clamp [0,1]
    score = max(0.0, min(1.0, score))
    return float(score)


__all__ = ["score_trade_thinking", "WEIGHTS"]


