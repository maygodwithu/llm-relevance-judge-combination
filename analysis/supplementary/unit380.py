"""CPU unit checks for the 380 trainer arithmetic (no model, no GPU): update/warm-up arithmetic, per-epoch permutation hashes versus the stored 2-epoch runs, and the frozen-N rule."""
import hashlib, json, math, glob, sys
import numpy as np
M, EB = 10000, 16
def plan(updates=0, epochs=1):
    B = updates if updates > 0 else math.ceil(M / EB) * epochs
    n_perm = math.ceil(B * EB / M); warm = math.ceil(0.1 * B)
    return B, n_perm, warm
assert plan(0, 1) == (625, 1, 63) and plan(0, 2) == (1250, 2, 125) and plan(625) == (625, 1, 63) and plan(1250) == (1250, 2, 125)   # new path reproduces the 255 settings exactly
for N in (2252, 2253):
    B, n_perm, warm = plan(N); assert B * EB <= n_perm * M and n_perm == 4 and warm == math.ceil(0.1 * N)
def hashes(seed, n):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        p = [int(i) for i in rng.permutation(M)]; out.append(hashlib.sha256(np.array(p, dtype=np.int64).tobytes()).hexdigest())
    return out
ok = 0
for f in sorted(glob.glob(sys.argv[1] + '/stats_E255EP2_T1_s*.json')):
    st = json.load(open(f)); seed = st['seed']; h = hashes(seed, 4)
    assert h[:2] == st['epoch_order_sha256'], f          # epochs 1-2 of the new plan equal the stored 2-epoch runs
    ok += 1
print('unit380 PASS: arithmetic ok; epoch-permutation hashes equal the stored 2-epoch runs for', ok, 'seeds')
