"""Maker-side research and execution pieces, kept out of the crypto engine.

`order_groups` is the quote kill switch, `rewards` is the public liquidity-program
recorder and scoring model, `rfq_recorder` records the authenticated communications
channel, and `demo_probe` answers the demo-quote capability question. Nothing in
this package is imported by `openkal.engine`; arming one of these lanes cannot arm
the four crypto lanes.
"""
