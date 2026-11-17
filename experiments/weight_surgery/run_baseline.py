import sys, time, json, numpy as np
import ws_lib as L
L.assert_original_untouched()
p = L.load_params()
t=time.time()
a = L.evaluate(p, seed=1000, n_per_scen=int(sys.argv[1]) if len(sys.argv)>1 else 70, verbose=True)
print("first eval", time.time()-t)
t=time.time()
a = L.evaluate(p, seed=1001, n_per_scen=int(sys.argv[1]) if len(sys.argv)>1 else 70)
print("second eval", time.time()-t)
for k,r in L.summarize(a, by_scen=True).items(): print(L.fmt_row(k,r))
print("score", L.score(a))
