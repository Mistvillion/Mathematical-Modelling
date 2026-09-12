"""Benchmark the current rolling script on the first N days."""
import sys, time
sys.path.insert(0, "scripts_tables")
import numpy as np
import importlib
mod = importlib.import_module("2_rolling_optimization")

N_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 3

t0 = time.time()
data = mod.load_model_data()
print(f"load data: {time.time()-t0:.2f}s", flush=True)
engine = mod.ForecastEngine()

penalties = np.max(data.price) * np.asarray(mod.TERMINAL_PENALTY_MULTIPLIERS)
default_penalty = mod.TERMINAL_PENALTY_MULTIPLIERS.index(1.0)
default_decay = mod.DECAY_DAY_CANDIDATES.index(mod.DEFAULT_DECAY_DAYS)
losses_load, losses_pv, losses_penalty = [], [], []
initial_energy = mod.E_INITIAL

for day_index in range(N_DAYS):
    day = data.dates[day_index]
    t_day = time.time()
    history_dates = data.dates[:day_index]
    load_scores = mod.historical_scores(losses_load, day_index, len(mod.DECAY_DAY_CANDIDATES), data.dates, True)
    pv_scores = mod.historical_scores(losses_pv, day_index, len(mod.DECAY_DAY_CANDIDATES), data.dates)
    penalty_scores = mod.historical_scores(losses_penalty, day_index, len(penalties), data.dates)
    load_index = mod.choose_candidate(load_scores, default_decay)
    pv_index = mod.choose_candidate(pv_scores, default_decay)
    penalty_index = mod.choose_candidate(penalty_scores, default_penalty)
    load_candidates, pv_candidates = engine.candidates(
        day, history_dates, data.actual_load_energy[:day_index],
        data.actual_photovoltaic_energy[:day_index],
        data.cold_start_load_energy, data.cold_start_photovoltaic_energy,
    )
    prior_load, prior_pv = engine.prior_fallback_flags(day, history_dates)
    forecast = mod.Forecast(load_candidates[load_index], pv_candidates[pv_index],
                            engine.latest_training_date(day, history_dates), prior_load, prior_pv)
    t_plan = time.time()
    plans = [mod.solve_validated_plan(day, data.price, forecast, initial_energy, float(p))
             for p in penalties]
    t_plan_end = time.time()
    actual_load = data.actual_load_energy[day_index]
    actual_pv = data.actual_photovoltaic_energy[day_index]
    replays = []
    t_replay = time.time()
    for candidate_index, (candidate_plan, penalty) in enumerate(zip(plans, penalties)):
        candidate_replay = mod.replay_actual_day(day, data.price, candidate_plan, forecast,
                                                 actual_load, actual_pv, initial_energy, float(penalty))
        mod.validate_replay(day, candidate_plan, candidate_replay, actual_load, actual_pv, float(penalty))
        replays.append(candidate_replay)
    t_replay_end = time.time()
    planned_costs = np.array([float(data.price @ p.grid_purchase) for p in plans])
    emergency_costs = np.array([float(mod.EMERGENCY_PRICE_MULTIPLIER * (data.price @ r.emergency_purchase)) for r in replays])
    realized_costs = planned_costs + emergency_costs
    initial_energy = float(replays[penalty_index].stored_energy[-1])
    print(f"{day}: forecast={t_plan-t_day:.2f}s plan(4 candidates)={t_plan_end-t_plan:.2f}s "
          f"replay(4x144x2 solves)={t_replay_end-t_replay:.2f}s daytotal={t_replay_end-t_day:.2f}s "
          f"cost={realized_costs[penalty_index]:.2f} end_energy={initial_energy:.1f}", flush=True)

print(f"TOTAL for {N_DAYS} days: {time.time()-t0:.2f}s -> est. 365d = {(time.time()-t0)/N_DAYS*365/3600:.2f} h", flush=True)
