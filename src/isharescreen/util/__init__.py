"""Standalone, protocol-agnostic helpers shared across the codebase.

Nothing here depends on the proxy / frontend / tui packages — these are
leaf utilities that can be imported from anywhere.

Layout:

    pcap_recorder       PcapRecorder + socket taps — write live session
                        traffic to a classic libpcap file (Wireshark /
                        the workspace dissector read it unmodified)
"""
