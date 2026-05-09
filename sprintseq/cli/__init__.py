"""CLI entry points for sprintseq subcommands + shared argparse type parsers."""

import argparse
import math


def quality_to_threshold(q):
    """Convert Phred quality score to probability threshold.

    Q20 → 0.99, Q30 → 0.999, Q13 ≈ 0.95.
    """
    return 1.0 - 10.0 ** (-q / 10.0)


def resolve_threshold_and_label(threshold, quality):
    """Return (probability, directory_label) from --threshold / -Q flags.

    Exactly one of *threshold* or *quality* should be set.
    """
    if quality is not None:
        return quality_to_threshold(quality), f"Q{quality}"
    return threshold, str(threshold)


def parse_cycles(value):
    """Parse a cycle-list spec like '1,2,3,4', '11', or '1-4,11' into a Python list.

    Accepts comma-separated integers and dash-inclusive ranges. Whitespace around
    tokens is tolerated. None and empty input are normalized:
      - None → None (caller falls back to its default)
      - '' / all whitespace / ',,,' → raises ArgumentTypeError (empty result is
        almost always a mistake on the command line, and letting it through
        causes a silent zero-cycle pipeline run).
    """
    if value is None:
        return None
    out = []
    for raw in str(value).split(','):
        part = raw.strip()
        if not part:
            continue
        if '-' in part:
            lo_s, hi_s = part.split('-', 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                raise argparse.ArgumentTypeError(
                    f"Invalid cycle range '{part}': start {lo} is greater than end {hi}."
                )
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    if not out:
        raise argparse.ArgumentTypeError(
            f"Empty cycle list parsed from {value!r}; specify at least one cycle."
        )
    return out


def parse_channels(value):
    """Parse 'cy3,cy5' → ['cy3', 'cy5']. None → None."""
    if value is None:
        return None
    out = [c.strip() for c in str(value).split(',') if c.strip()]
    if not out:
        raise argparse.ArgumentTypeError(
            f"Empty channel list parsed from {value!r}; specify at least one channel."
        )
    return out
