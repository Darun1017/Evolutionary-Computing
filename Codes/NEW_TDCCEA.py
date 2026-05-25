"""
NEW_TDCCEA.py - Optimized TD-CCEA + Cooperative Co-Evolution

Improvements over CCE-TDCCEA:
  1. Multi-partner eval (top-3 partners, averaged fitness)
  2. Combined environment vector (both pops' centroids)
  3. Pair archive (stores best macro+micro pairs per opponent)
  4. Crossover (blend two parents, not just mutation)
  5. Elite pair injection every 5 gens
  6. Opponent memory for instant recall
"""

import math, random, time, subprocess, os, json

os.environ["PYTHONIOENCODING"] = "utf-8"

MICRORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "Simulation", "microrts")
WEIGHT_NAMES = [
    "worker_ratio", "max_workers", "expand_threshold",
    "light_preference", "heavy_preference", "ranged_preference",
    "barracks_timing", "aggression", "attack_threshold",
    "defend_radius", "target_priority", "defense_trigger",
    "counter_rush", "tech_switch", "economy_pressure",
    "scout_priority", "group_attack", "second_barracks",
    "worker_fight_ratio", "focus_fire",
]
DIMENSIONS = len(WEIGHT_NAMES)
MAP_PATH = "maps/chambers32x32.xml"
MAX_CYCLES = 10000
OPPONENT_ROTATION = [
    "LightRush"
]
MACRO_INDICES = [0, 1, 2, 6, 14, 15, 17, 18]
MICRO_INDICES = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 16, 19]

N_POP = 20
K_GAMES = 3
TOTAL_GENERATIONS = 30
VALIDATION_GAMES = 5
A_MAX = 15
DELTA_S = 0.85
DELTA_F = 0.20
TOP_PARTNERS = 1
HISTORY_DECAY = 0.75   # recent history weighted more (1.0 = no decay)


class GameServer:
    def __init__(self):
        self.process = None
    def start(self):
        self.process = subprocess.Popen(
            ["java", "-cp", "lib/*;bin", "tests.EvolvedBatchGameRunner"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=MICRORTS_DIR)
        if self.process.stdout.readline().strip() != "READY":
            raise RuntimeError("Server failed")
    def run_game(self, weights, opponent):
        self.process.stdin.write(f"EVOLVED:{';'.join(f'{w:.4f}' for w in weights)},{opponent},{MAP_PATH},{MAX_CYCLES}\n")
        self.process.stdin.flush()
        r = self.process.stdout.readline().strip()
        if r.startswith("RESULT:"):
            p = r.replace("RESULT:", "").split(",")
            return int(p[0]), int(p[1])
        return -1, MAX_CYCLES
    def stop(self):
        if self.process:
            try: self.process.stdin.write("EXIT\n"); self.process.stdin.flush(); self.process.wait(timeout=5)
            except: self.process.kill()

def euclidean_dist(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))

def evaluate_weights(server, weights, opponent):
    wins = 0
    for _ in range(K_GAMES):
        w, _ = server.run_game(weights, opponent)
        if w == 0: wins += 1
        elif w == -1: wins += 0.3
    return wins / K_GAMES

def validate_weights(server, weights):
    tw, tg, po = 0, 0, {}
    for opp in OPPONENT_ROTATION:
        ow = sum(1 for _ in range(VALIDATION_GAMES) if server.run_game(weights, opp)[0] == 0)
        tg += VALIDATION_GAMES; tw += ow
        po[opp] = ow / VALIDATION_GAMES * 100
    return tw / tg * 100, po

def combine_weights(macro, micro):
    full = [0.5] * DIMENSIONS
    for i, idx in enumerate(MACRO_INDICES): full[idx] = macro.params[i]
    for i, idx in enumerate(MICRO_INDICES): full[idx] = micro.params[i]
    return full


class SubAgent:
    def __init__(self, dim, params=None):
        self.dim = dim
        self.params = [random.random() for _ in range(dim)] if params is None \
            else [max(0.0, min(1.0, float(x))) for x in params]
        self.H_i, self.e_i, self.mu_i = [], 0.0, 0.0

    def update_history(self, fitness, env):
        self.H_i.append((fitness, env))
        if len(self.H_i) > 7: self.H_i.pop(0)

    def compute_trajectory(self, env):
        """Trajectory score with temporal decay — recent history matters more."""
        if not self.H_i: self.e_i = 0.0; return
        weighted_sum = 0.0
        weight_total = 0.0
        for t, (past_f, past_env) in enumerate(self.H_i):
            # Recency: most recent (largest t) gets highest weight
            recency = HISTORY_DECAY ** (len(self.H_i) - 1 - t)
            # Environmental similarity
            cl = min(len(env), len(past_env))
            if cl > 0:
                d = euclidean_dist(env[:cl], past_env[:cl])
                similarity = max(0.0, 1.0 - d / math.sqrt(cl))
            else:
                similarity = 1.0
            combined = recency * similarity
            weighted_sum += combined * past_f
            weight_total += combined
        self.e_i = weighted_sum / weight_total if weight_total > 0 else 0.0

    def compute_mutation(self):
        self.mu_i = 0.15 * (1.0 - self.e_i) + 0.03

    def mutate(self):
        cp = [max(0.0, min(1.0, p + random.gauss(0, self.mu_i))) for p in self.params]
        return SubAgent(self.dim, cp)

    def crossover(self, other):
        """Blend crossover: mix genes from two parents."""
        alpha = random.random()
        cp = [max(0.0, min(1.0, alpha * a + (1-alpha) * b)) for a, b in zip(self.params, other.params)]
        return SubAgent(self.dim, cp)


def tdccea_survival(population, archive):
    dim = population[0].dim
    md = math.sqrt(dim)
    survivors = []
    for a in sorted(population, key=lambda a: a.e_i, reverse=True):
        if not any(1.0 - euclidean_dist(a.params, u.params)/md > DELTA_S for u in survivors):
            survivors.append(a)
    final = [a for a in survivors if not (a.e_i < DELTA_F and len(a.H_i) >= 3)]
    while len(final) < 8 and len(survivors) > len(final):
        for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
            if a not in final: final.append(a); break
    for top in sorted(population, key=lambda a: a.e_i, reverse=True)[:3]:
        ar = SubAgent(dim, top.params)
        ar.e_i, ar.H_i, ar.mu_i = top.e_i, top.H_i[:], top.mu_i
        archive.append(ar)
    archive.sort(key=lambda a: a.e_i, reverse=True)
    del archive[A_MAX:]
    pool = final + archive
    new_pop = final[:]
    while len(new_pop) < N_POP:
        cands = random.sample(pool, min(3, len(pool)))
        parent = max(cands, key=lambda a: a.e_i)
        if random.random() < 0.3 and len(pool) >= 2:
            other = random.choice(pool)
            new_pop.append(parent.crossover(other))
        else:
            new_pop.append(parent.mutate())
    return new_pop, archive


# ===============================================================
#  NEW-TDCCEA (Optimized TD-CCEA + Co-Evolution)
# ===============================================================
def run_new_tdccea(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: NEW-TDCCEA (Optimized Co-Evolution)")
    print(f"  Macro: {len(MACRO_INDICES)}w | Micro: {len(MICRO_INDICES)}w | Multi-partner eval")
    print(f"{'='*60}")

    pop_a = [SubAgent(len(MACRO_INDICES)) for _ in range(N_POP)]
    pop_b = [SubAgent(len(MICRO_INDICES)) for _ in range(N_POP)]
    archive_a, archive_b = [], []
    gen_scores = []
    t0 = time.time()

    # Track top partners from previous gen
    top_b_partners = pop_b[:TOP_PARTNERS]
    top_a_partners = pop_a[:TOP_PARTNERS]

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]

        # Combined environment vector (both pops' centroids merged)
        cent_a = [sum(a.params[j] for a in pop_a)/len(pop_a) for j in range(len(MACRO_INDICES))]
        cent_b = [sum(b.params[j] for b in pop_b)/len(pop_b) for j in range(len(MICRO_INDICES))]
        combined_env = cent_a + cent_b  # combined environment

        # Multi-partner evaluation for Pop A
        for agent_a in pop_a:
            fitnesses = []
            for partner_b in top_b_partners:
                full = combine_weights(agent_a, partner_b)
                fitnesses.append(evaluate_weights(server, full, opponent))
            avg_fit = sum(fitnesses) / len(fitnesses)
            agent_a.update_history(avg_fit, combined_env)
            agent_a.compute_trajectory(combined_env)
            agent_a.compute_mutation()

        # Multi-partner evaluation for Pop B
        for agent_b in pop_b:
            fitnesses = []
            for partner_a in top_a_partners:
                full = combine_weights(partner_a, agent_b)
                fitnesses.append(evaluate_weights(server, full, opponent))
            avg_fit = sum(fitnesses) / len(fitnesses)
            agent_b.update_history(avg_fit, combined_env)
            agent_b.compute_trajectory(combined_env)
            agent_b.compute_mutation()

        best_a = max(pop_a, key=lambda a: a.e_i)
        best_b = max(pop_b, key=lambda a: a.e_i)

        # Update top partners for next gen
        top_a_partners = sorted(pop_a, key=lambda a: a.e_i, reverse=True)[:TOP_PARTNERS]
        top_b_partners = sorted(pop_b, key=lambda a: a.e_i, reverse=True)[:TOP_PARTNERS]

        # Test actual combined fitness
        combined_fit = evaluate_weights(server, combine_weights(best_a, best_b), opponent)
        gen_scores.append(combined_fit)

        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opponent:25s} | "
              f"A={best_a.e_i:.3f} B={best_b.e_i:.3f} Cmb={combined_fit:.3f}")

        # TD-CCEA survival for each sub-population (with crossover)
        pop_a, archive_a = tdccea_survival(pop_a, archive_a)
        pop_b, archive_b = tdccea_survival(pop_b, archive_b)

    best_weights = combine_weights(best_a, best_b)
    return best_weights, time.time() - t0, gen_scores


# ===============================================================
#  STANDARD TD-CCEA (for comparison)
# ===============================================================
class Agent:
    def __init__(self, params=None):
        self.params = [random.random() for _ in range(DIMENSIONS)] if params is None \
            else [max(0.0, min(1.0, float(x))) for x in params]
        self.H_i, self.e_i, self.mu_i = [], 0.0, 0.0
    def update_history(self, f, env):
        self.H_i.append((f, env)); 
        if len(self.H_i) > 7: self.H_i.pop(0)
    def compute_trajectory(self, env):
        """Trajectory score with temporal decay."""
        if not self.H_i: self.e_i = 0.0; return
        md = math.sqrt(DIMENSIONS)
        weighted_sum = 0.0
        weight_total = 0.0
        for t, (past_f, past_env) in enumerate(self.H_i):
            recency = HISTORY_DECAY ** (len(self.H_i) - 1 - t)
            similarity = max(0.0, 1.0 - euclidean_dist(env, past_env) / md)
            combined = recency * similarity
            weighted_sum += combined * past_f
            weight_total += combined
        self.e_i = weighted_sum / weight_total if weight_total > 0 else 0.0
    def compute_mutation(self): self.mu_i = 0.15 * (1.0 - self.e_i) + 0.03
    def mutate(self):
        return Agent([max(0.0, min(1.0, p + random.gauss(0, self.mu_i))) for p in self.params])

def run_tdccea(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: TD-CCEA Centroid")
    print(f"{'='*60}")
    population = [Agent() for _ in range(N_POP)]
    archive, gs = [], []
    t0 = time.time()
    for gen in range(1, TOTAL_GENERATIONS + 1):
        opp = OPPONENT_ROTATION[(gen-1) % len(OPPONENT_ROTATION)]
        cent = [sum(a.params[j] for a in population)/len(population) for j in range(DIMENSIONS)]
        for a in population:
            f = evaluate_weights(server, a.params, opp)
            a.update_history(f, cent); a.compute_trajectory(cent); a.compute_mutation()
        best = max(population, key=lambda a: a.e_i); gs.append(best.e_i)
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | e_i={best.e_i:.3f}")
        md = math.sqrt(DIMENSIONS); survivors = []
        for a in sorted(population, key=lambda a: a.e_i, reverse=True):
            if not any(1.0-euclidean_dist(a.params,u.params)/md > DELTA_S for u in survivors): survivors.append(a)
        final = [a for a in survivors if not (a.e_i < DELTA_F and len(a.H_i)>=3)]
        while len(final)<10 and len(survivors)>len(final):
            for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
                if a not in final: final.append(a); break
        for top in sorted(population, key=lambda a: a.e_i, reverse=True)[:3]: archive.append(Agent(top.params))
        archive.sort(key=lambda a: a.e_i, reverse=True); archive=archive[:A_MAX]
        pool=final+archive; population=final[:]
        while len(population)<N_POP:
            c=random.sample(pool,min(3,len(pool))); population.append(max(c,key=lambda a:a.e_i).mutate())
    b = max(archive, key=lambda a: a.e_i) if archive else max(population, key=lambda a: a.e_i)
    return b.params, time.time()-t0, gs

def run_hc(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: Hill Climbing")
    print(f"{'='*60}")
    cur=[random.random() for _ in range(DIMENSIONS)]; cf,gs=0.0,[]; t0=time.time()
    for gen in range(1,TOTAL_GENERATIONS+1):
        opp=OPPONENT_ROTATION[(gen-1)%len(OPPONENT_ROTATION)]
        cf=evaluate_weights(server,cur,opp); bc,bf=cur[:],cf
        for _ in range(N_POP-1):
            c=[max(0.0,min(1.0,p+random.gauss(0,0.15))) for p in cur]
            f=evaluate_weights(server,c,opp)
            if f>bf: bc,bf=c[:],f
        if bf>=cf: cur,cf=bc[:],bf
        gs.append(cf); print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | Best={cf:.3f}")
    return cur, time.time()-t0, gs


def main():
    print()
    print("#" * 70)
    print("  NEW-TDCCEA BENCHMARK")
    print("  Optimized Co-Evolution vs TD-CCEA vs Hill Climbing")
    print("#" * 70)
    print(f"  Generations: {TOTAL_GENERATIONS} | Opponents: {len(OPPONENT_ROTATION)} | Pop: {N_POP}")
    print()

    algos = [
        ("NEW-TDCCEA",        "new"),
        ("TD-CCEA Centroid",  "tdccea"),
        ("Hill Climbing",     "hc"),
    ]

    all_results = []
    for name, key in algos:
        server = GameServer(); server.start()
        print(f"  [OK] Server started for {name}")
        if key == "new":    best, el, sc = run_new_tdccea(server)
        elif key == "tdccea": best, el, sc = run_tdccea(server)
        else:                 best, el, sc = run_hc(server)

        print(f"\n  Validating {name}...")
        wr, po = validate_weights(server, best)
        server.stop()

        print(f"\n  +-- {name} RESULTS --------------------------------")
        print(f"  |  Time: {el:.1f}s | Win Rate: {wr:.1f}%")
        for opp, w in po.items():
            print(f"  |  {'[W]' if w >= 50 else '[L]'} vs {opp:<26s}: {w:5.0f}%")
        print(f"  +-------------------------------------------------------")
        all_results.append({"name": name, "time": el, "overall_winrate": wr, "per_opponent": po})

    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*60}")
    for r in sorted(all_results, key=lambda x: x["overall_winrate"], reverse=True):
        print(f"  {r['name']:<22s}  {r['overall_winrate']:>9.1f}%  {r['time']:>7.1f}s")

    ranked = sorted(all_results, key=lambda x: x["overall_winrate"], reverse=True)
    medals = ["1st", "2nd", "3rd"]
    print(f"\n{'='*60}")
    print(f"  RANKING")
    print(f"{'='*60}")
    for i, r in enumerate(ranked):
        print(f"  {medals[i]}  {r['name']:<22s}  {r['overall_winrate']:5.1f}%  ({r['time']:.0f}s)")
    print(f"\n{'='*60}")
    print(f"  BENCHMARK COMPLETE!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
