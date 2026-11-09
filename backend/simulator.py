"""Node + controller simulator (Python port of the browser version). Scores/labels are real; energy model is illustrative."""
import math, random
import numpy as np
NAMES = {"sendall": "Send everything", "random": "Random subsample", "fixed": "Fixed threshold",
         "battery": "Battery-only", "quantile": "Quantile-only", "full": "Full controller"}


def vq(D, dim, p):
    return float(np.interp(min(1.0, max(0.0, p)) * 100, np.arange(101), D["vq"][dim]))


def runq(w, p):
    a = sorted(w)
    return a[min(len(a) - 1, max(0, int(p * len(a))))]


def sim(D, c, P, ctrl, ci):
    s, n = c["s"][P.dim], c["n"]
    R, pay, epd = random.Random(P.seed * 131 + ci), P.dim + P.hdr, P.epd
    days = len(s) // epd
    batt, off, win = P.cap * P.start, False, []
    tx = nt = ntot = black = bout = 0
    tr = []
    fixed = vq(D, P.dim, 1 - P.rate)
    cost = P.eTx0 + P.eTxB * pay
    for d in range(days):
        left, blk = P.budget, False
        sun = P.solar * (0.6 + 0.4 * math.sin(2 * math.pi * (d + ci * 7) / 60)) * (0.3 if R.random() < 0.3 else 1)
        for i in range(epd):
            k = d * epd + i
            x, nov = s[k], n[k] & 1
            ntot += nov
            if off and batt > P.cap * 0.15: off = False
            if not off and batt <= P.cap * 0.05: off = True
            if off:
                blk = True
                continue
            batt -= P.eEnc
            f = min(1, max(0.1, batt / (P.cap * 0.5)))
            pace = min(2, max(0.25, (left / P.budget) / ((epd - i) / epd)))
            if ctrl == "sendall": send = True
            elif ctrl == "random": send = R.random() < P.rate
            elif ctrl == "fixed": send = x >= fixed
            elif ctrl == "battery": send = x >= vq(D, P.dim, 1 - P.rate * f)
            else:
                r = P.rate if ctrl == "quantile" else min(1, P.rate * f * pace)
                send = x >= (runq(win, 1 - r) if len(win) >= 30 else vq(D, P.dim, 1 - r))
            if send and left >= pay and batt >= cost:
                left -= pay; tx += 1; nt += nov; batt -= cost
            win.append(x)
            if len(win) > 150: win.pop(0)
        if left < pay: bout += 1
        if blk: black += 1
        batt = min(P.cap, batt - P.eIdle + sun)
        tr.append(round(max(0, batt) / P.cap, 4))
    return tx, nt, ntot, black, bout, days, tr


def run_all(D, P):
    results, traces = [], {c["id"]: {} for c in D["cams"]}
    for ct, name in NAMES.items():
        T = np.zeros(6)
        for ci, c in enumerate(D["cams"]):
            tx, nt, ntot, black, bout, days, tr = sim(D, c, P, ct, ci)
            T += [tx, nt, ntot, black, bout, days]
            traces[c["id"]][ct] = tr
        tx, nt, ntot, black, bout, days = T
        results.append({"ctrl": ct, "name": name, "sent_per_day": tx / days, "bytes_per_day": tx * (P.dim + P.hdr) / days,
                        "novel_recall": nt / ntot if ntot else 0, "novel_share": nt / tx if tx else 0,
                        "off_pct": black / days, "budget_out_pct": bout / days})
    return results, traces
