"""Independent trading lanes.

Each module here owns one frozen rule and its own operator arm record. Nothing in
this package is imported by `openkal.engine` or by the crypto execution path, so a
lane can be armed, paused or retired without changing what the four crypto lanes
do.
"""
