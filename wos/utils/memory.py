import torch

CUDA_OOM_MSG = "CUDA out of memory"


def byte2mb(byte, ndigits=2):
    return round(byte / 1024**2, ndigits)


def catch_oom(fn, *args, **kwargs):
    try:
        return None, fn(*args, **kwargs)
    except RuntimeError as err:
        # catch cuda out of memory errors
        if CUDA_OOM_MSG in str(err):
            return err, None
        raise


def max_mem_allocation(fn, *args, device=None, add_oom=1, empty_cache=True, **kwargs):
    if device is not None and torch.device(device).type == "cuda":
        if empty_cache:
            torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    err, output = catch_oom(fn, *args, **kwargs)
    if err:
        return torch.cuda.get_device_properties(device).total_memory + add_oom, output
    if device is not None and torch.device(device).type == "cuda":
        return torch.cuda.max_memory_allocated(device), output
    return 0.0, output


def check_mem(mem, mem_bound=None, rel_margin=0.0, device=None, throw_error=True):
    # get cuda device memory
    device_mem_bound = float("inf")
    if device is not None and torch.device(device).type == "cuda":
        device_mem_bound = torch.cuda.get_device_properties(device).total_memory

    # compute overall bound
    mem_bound = (
        device_mem_bound if mem_bound is None else min(device_mem_bound, mem_bound)
    )

    print("mem_bound: ", mem_bound)
    # check
    if mem_bound < mem * (1.0 + rel_margin):
        if throw_error:
            raise RuntimeError(
                f"{CUDA_OOM_MSG}. Tried to allocate a maximum of {byte2mb(mem)} MiB and "
                f"{byte2mb(rel_margin * mem)} MiB margin ({byte2mb(mem_bound)} MiB allowed)."
            )
    print(
        f"Allocated {byte2mb(mem)} MiB with {byte2mb(rel_margin * mem)} MiB margin "
        f"({byte2mb(mem_bound)} MiB allowed)."
    )

