"""Device selection without constructing models or running benchmarks."""


def list_compute_devices():
    """Return device names that the installed runtime can see. CPU is always included."""

    found = ["cpu"]
    try:
        import torch
    except Exception:
        return found
    try:
        if torch.cuda.is_available():
            found.append("cuda")
    except Exception:
        pass
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            found.append("mps")
    except Exception:
        pass
    return found


def resolve_device(requested):
    """Validate a configured device. ``auto`` prefers CUDA, then MPS, then CPU.

    Availability checks do not allocate a model or time a kernel.
    """

    requested = "cpu" if requested is None else str(requested)
    if (
        requested in {"cpu", "mps"}
        or requested.isdigit()
        or requested.startswith("cuda")
    ):
        return requested
    if requested == "auto":
        available = list_compute_devices()
        for choice in ("cuda", "mps", "cpu"):
            if choice in available:
                return choice
    raise ValueError("device must be cpu, mps, cuda, cuda:N, a GPU index, or auto")
