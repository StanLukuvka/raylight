"""Dependency-free process memory telemetry for Ray workers."""

from pathlib import Path


_KB_FIELDS = {"VmRSS": "rss_bytes", "VmHWM": "peak_rss_bytes"}


def process_memory_snapshot(status_path: str | Path = "/proc/self/status") -> dict[str, int | None]:
    """Return current and peak resident memory from Linux ``/proc``."""
    values: dict[str, int | None] = {value: None for value in _KB_FIELDS.values()}
    try:
        lines = Path(status_path).read_text().splitlines()
    except OSError:
        return values

    for line in lines:
        name, separator, raw_value = line.partition(":")
        output_name = _KB_FIELDS.get(name)
        if not separator or output_name is None:
            continue
        fields = raw_value.split()
        if not fields:
            continue
        try:
            values[output_name] = int(fields[0]) * 1024
        except ValueError:
            continue
    return values
