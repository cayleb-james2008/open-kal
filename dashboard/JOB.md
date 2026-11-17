# JOB — Open-Kal console

Operate a live 15-minute Kalshi engine: see whether it is healthy, which slot has an
edge in the current window, and what it has done with the money.

1. Is the engine live, connected and allowed to trade right now?
2. Which slot is quoting or holding, and where is the edge this window?
3. Am I up or down, and on what?
4. What did it just do — fills, exits, halts?
5. Is the recursive loop healthy, and has it promoted anything?

Real data: `GET /api/state` from the engine on this machine. Slots are the five
contract families in `openkal/config.py`; prices, fair values and fills are the
engine's own. Nothing on the screen is example data.
