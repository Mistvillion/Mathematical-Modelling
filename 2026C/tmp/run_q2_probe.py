"""Run the real solve_rolling_model with instrumentation to find slow/failing solves."""
import sys, time, traceback
sys.path.insert(0, "scripts_tables")
import numpy as np
import importlib
mod = importlib.import_module("2_rolling_optimization")

CALLS = {"n": 0, "slow": [], "t": 0.0}
orig_intraday = mod.solve_intraday_step
def timed_intraday(**kwargs):
    t0 = time.time()
    try:
        out = orig_intraday(**kwargs)
    finally:
        dt = time.time() - t0
        CALLS["n"] += 1
        CALLS["t"] += dt
        if dt > 0.5:
            CALLS["slow"].append((str(kwargs.get("day")), kwargs.get("period_index", -1) + 1, dt))
        if CALLS["n"] % 500 == 0:
            print(f"  intraday calls={CALLS['n']} total={CALLS['t']:.1f}s last_slow={CALLS['slow'][-3:]}", flush=True)
    return out
mod.solve_intraday_step = timed_intraday

orig_plan = mod.solve_validated_plan
def timed_plan(*a, **k):
    t0 = time.time()
    try:
        return orig_plan(*a, **k)
    finally:
        dt = time.time() - t0
        if dt > 0.5:
            print(f"  slow plan {a[0]} lambda={a[4]:.3f}: {dt:.2f}s", flush=True)
mod.solve_validated_plan = timed_plan

t0 = time.time()
data = mod.load_model_data()
env = mod.ForecastEngine()
try:
    rolling = mod.solve_rolling_model(data, env)
    print(f"SUCCESS in {time.time()-t0:.1f}s", flush=True)
except Exception as exc:
    print(f"FAILED after {time.time()-t0:.1f}s: {type(exc).__name__}: {exc}", flush=True)
    if isinstance(exc, mod.IntradaySolveError):
        for k, v, u in exc.diagnostics:
            print(f"   {k}={v} {u}", flush=True)
        print(f"   completed_days={len(exc.completed_days)} completed_steps={len(exc.completed_steps)}", flush=True)
    traceback.print_exc()
print(f"intraday calls={CALLS['n']} total={CALLS['t']:.1f}s", flush=True)
print(f"slowest: {sorted(CALLS['slow'], key=lambda x: -x[2])[:10]}", flush=True)
