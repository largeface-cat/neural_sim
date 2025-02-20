import struct
from typing import List, Tuple, Dict, Any, Callable

FORMAT = 8

def int2hex(n):
    return f"{hex(n)[2:].upper():0>{FORMAT}}"

def hex2int(s, lim=(-2**31, 2**31-1)):
    res = int(s, 16)
    return res if lim[0] <= res <= lim[1] else lim[0] if res < lim[0] else lim[1]

def float2hex(f):
    return int2hex(struct.unpack("<I", struct.pack("<f", f))[0])

def hex2float(s, lim=(-3.4e38, 3.4e38)):
    res = struct.unpack("<f", struct.pack("<I", hex2int(s)))[0]
    return res if lim[0] <= res <= lim[1] else lim[0] if res < lim[0] else lim[1]

def avg(lst):
    if not lst:
        return 0
    return sum(lst)/len(lst)

def deepcopy_dict(d:dict, func:Callable):
    return {k: func(v) for k, v in d.items()}