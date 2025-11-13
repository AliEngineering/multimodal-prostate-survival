import torch
import numpy as np


def concordance_index(risk, time, event):
    r = risk.cpu().numpy()
    t = time.cpu().numpy()
    e = event.cpu().numpy()

    n = len(r)
    num = 0.0
    den = 0.0

    for i in range(n):
        for j in range(i+1, n):
            if t[i] == t[j]:
                continue
            if e[i] == 1 and t[i] < t[j]:
                den += 1
                num += (r[i] > r[j]) + 0.5 * (r[i] == r[j])
            elif e[j] == 1 and t[j] < t[i]:
                den += 1
                num += (r[j] > r[i]) + 0.5 * (r[j] == r[i])

    return num / den if den > 0 else 0
