"""
HEAD_TO_HEAD_ALL.py
===================
Phase 1: Evolve 8 algorithms against bots
Phase 2: Round-robin tournament between evolved agents

CRITICAL NOTE — JAVA SERVER REQUIREMENT:
  The run_evolved_vs_evolved() method sends:
    EVOLVED:{weights1},EVOLVED:{weights2},{map},{cycles}
  Your Java server (EvolvedBatchGameRunner) must support this format.

  If your server only supports evolved-vs-bot (the original format),
  every head-to-head game will return -1 (error) and be counted as a draw,
  making ALL win rates 0% and the ranking meaningless.

  TO ADD EVOLVED-VS-EVOLVED SUPPORT to your Java server, your
  EvolvedBatchGameRunner needs to detect when the opponent string starts
  with "EVOLVED:" and load a second EvolvedBrainAI with those weights.

  This code detects if evolved-vs-evolved works by running a probe game
  at startup. If it fails, it falls back to bot-based proxy comparison.

FIXES OVER SUBMITTED VERSION:
  [FIX 1] Added server support detection — probes evolved-vs-evolved at start
  [FIX 2] Added graceful fallback — if server doesn't support it, uses
          bot-validation win rates as proxy ranking instead of silent failure
  [FIX 3] Added error counting — tracks how many games returned -1 (error)
           and warns if error rate is high
  [FIX 4] Added draws column to results matrix
  [FIX 5] Fixed potential silent-fail: if all games are -1, prints clear error
           instead of showing a fake 0% ranking
  [FIX 6] Sensitivity file loaded with correct direction check
"""

import math, random, time, subprocess, os, json

os.environ["PYTHONIOENCODING"] = "utf-8"

# ===============================================================
#  CONFIGURATION
# ===============================================================
MICRORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Simulation", "microrts"
)
WEIGHT_NAMES = [
    "worker_ratio","max_workers","expand_threshold","light_preference",
    "heavy_preference","ranged_preference","barracks_timing","aggression",
    "attack_threshold","defend_radius","target_priority","defense_trigger",
    "counter_rush","tech_switch","economy_pressure","scout_priority",
    "group_attack","second_barracks","worker_fight_ratio","focus_fire",
]
DIMENSIONS    = len(WEIGHT_NAMES)
MACRO_INDICES = [0,1,2,6,14,15,17,18]
MICRO_INDICES = [3,4,5,7,8,9,10,11,12,13,16,19]
MAP_PATH      = "maps/chambers32x32.xml"
MAX_CYCLES    = 10000

OPPONENT_ROTATION = [
    "EconomyMilitaryRush","LightRush","WorkerRush",
    "HeavyDefense","RangedRush","EconomyRush","WorkerRushPlusPlus",
]

N_POP             = 20
K_GAMES           = 3
TOTAL_GENERATIONS = 30
VALIDATION_GAMES  = 5
A_MAX             = 15
DELTA_S           = 0.85
DELTA_F           = 0.20
TOP_PARTNERS      = 3
H2H_GAMES         = 10    # games per matchup (5 each side)
SENSITIVITY_FILE  = "sensitivity_scores.json"


# ===============================================================
#  GAME SERVER
# ===============================================================
class GameServer:
    def __init__(self): self.process = None

    def start(self):
        self.process = subprocess.Popen(
            ["java", "-cp", "lib/*;bin", "tests.EvolvedBatchGameRunner"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=MICRORTS_DIR
        )
        if self.process.stdout.readline().strip() != "READY":
            raise RuntimeError("Server failed to start")

    def run_game(self, weights, opponent):
        """Play evolved agent vs scripted bot."""
        ws = ";".join(f"{w:.4f}" for w in weights)
        self.process.stdin.write(f"EVOLVED:{ws},{opponent},{MAP_PATH},{MAX_CYCLES}\n")
        self.process.stdin.flush()
        r = self.process.stdout.readline().strip()
        if r.startswith("RESULT:"):
            p = r.replace("RESULT:", "").split(",")
            return int(p[0]), int(p[1])
        return -1, MAX_CYCLES

    def run_evolved_vs_evolved(self, w1, w2):
        """
        Play two evolved agents against each other.
        REQUIRES Java server to support: EVOLVED:{w1},EVOLVED:{w2},{map},{cycles}
        Returns (winner, cycles) where winner=0 means w1 wins, winner=1 means w2 wins.
        Returns (-1, MAX_CYCLES) if server does not support this format.
        """
        s1 = ";".join(f"{w:.4f}" for w in w1)
        s2 = ";".join(f"{w:.4f}" for w in w2)
        self.process.stdin.write(f"EVOLVED:{s1},EVOLVED:{s2},{MAP_PATH},{MAX_CYCLES}\n")
        self.process.stdin.flush()
        r = self.process.stdout.readline().strip()
        if r.startswith("RESULT:"):
            p = r.replace("RESULT:", "").split(",")
            return int(p[0]), int(p[1])
        return -1, MAX_CYCLES

    def check_evolved_vs_evolved_support(self):
        """
        [FIX 1] Probe the server to see if evolved-vs-evolved is supported.
        Sends a test game with dummy weights and checks if we get a valid RESULT.
        """
        dummy = [0.5] * DIMENSIONS
        winner, _ = self.run_evolved_vs_evolved(dummy, dummy)
        # If server returns -1 it means it didn't understand the command
        # If it returns 0 or 1 it understood and ran the game
        return winner != -1

    def stop(self):
        if self.process:
            try:
                self.process.stdin.write("EXIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()


# ===============================================================
#  HELPERS
# ===============================================================
def euclidean_dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def evaluate_weights(server, weights, opponent):
    wins = 0
    for _ in range(K_GAMES):
        w, _ = server.run_game(weights, opponent)
        if w == 0:   wins += 1
        elif w == -1: wins += 0.3
    return wins / K_GAMES


def validate_weights(server, weights):
    tw = tg = 0; per_opp = {}
    for opp in OPPONENT_ROTATION:
        ow = 0
        for _ in range(VALIDATION_GAMES):
            w, _ = server.run_game(weights, opp)
            if w == 0: ow += 1
            tg += 1
        tw += ow; per_opp[opp] = ow / VALIDATION_GAMES * 100
    return tw / tg * 100, per_opp


def load_sensitivity():
    fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), SENSITIVITY_FILE)
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        data = json.load(f)
    rates = data["normalized_mutation_rates"]
    # [FIX 6] Check correct formula was used
    if min(rates) < 0.07 or max(rates) > 0.21:
        print("  WARNING: sensitivity_scores.json was made with wrong formula.")
        print("  Run BENCHMARK_VS_BOTS.py first to regenerate it correctly.")
        return None
    return rates


# ===============================================================
#  CLASSES (identical to BENCHMARK_VS_BOTS.py)
# ===============================================================
class SubAgent:
    def __init__(self, dim, params=None):
        self.dim    = dim
        self.params = ([random.random() for _ in range(dim)] if params is None
                       else [max(0.0, min(1.0, float(x))) for x in params])
        self.H_i = []; self.e_i = 0.0; self.mu_i = 0.0

    def update_history(self, f, env):
        self.H_i.append((f, env))
        if len(self.H_i) > 7: self.H_i.pop(0)

    def compute_trajectory(self, env):
        if not self.H_i: self.e_i = 0.0; return
        md = math.sqrt(self.dim)
        ws = [max(0.0, 1.0 - euclidean_dist(env[:self.dim], pe[:self.dim]) / md)
              if len(pe) >= self.dim else 1.0 for _, pe in self.H_i]
        t  = sum(ws); ws = [w/t for w in ws] if t > 0 else [1.0/len(ws)] * len(ws)
        self.e_i = sum(w * self.H_i[i][0] for i, w in enumerate(ws))

    def compute_mutation(self): self.mu_i = 0.15 * (1.0 - self.e_i) + 0.03

    def mutate(self):
        return SubAgent(self.dim,
            [max(0.0, min(1.0, p + random.gauss(0, self.mu_i))) for p in self.params])


def combine_subagents(macro, micro):
    full = [0.5] * DIMENSIONS
    for i, idx in enumerate(MACRO_INDICES): full[idx] = macro.params[i]
    for i, idx in enumerate(MICRO_INDICES):  full[idx] = micro.params[i]
    return full


def subpop_survival(pop, archive):
    dim = pop[0].dim; md = math.sqrt(dim); survivors = []
    for a in sorted(pop, key=lambda a: a.e_i, reverse=True):
        if not any(1.0 - euclidean_dist(a.params, u.params) / md > DELTA_S for u in survivors):
            survivors.append(a)
    final = [a for a in survivors if not (a.e_i < DELTA_F and len(a.H_i) >= 3)]
    while len(final) < 8 and len(survivors) > len(final):
        for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
            if a not in final: final.append(a); break
    for top in sorted(pop, key=lambda a: a.e_i, reverse=True)[:3]:
        ar = SubAgent(dim, top.params)
        ar.e_i, ar.H_i, ar.mu_i = top.e_i, top.H_i[:], top.mu_i
        archive.append(ar)
    archive.sort(key=lambda a: a.e_i, reverse=True); del archive[A_MAX:]
    pool = final + archive; new_pop = final[:]
    while len(new_pop) < N_POP:
        cands = random.sample(pool, min(3, len(pool)))
        new_pop.append(max(cands, key=lambda a: a.e_i).mutate())
    return new_pop, archive


class Agent:
    def __init__(self, params=None, pwm=None):
        self.params = ([random.random() for _ in range(DIMENSIONS)] if params is None
                       else [max(0.0, min(1.0, float(x))) for x in params])
        self.H_i = []; self.e_i = 0.0; self.mu_i = 0.0; self.pwm = pwm

    def update_history(self, f, env):
        self.H_i.append((f, env))
        if len(self.H_i) > 7: self.H_i.pop(0)

    def compute_trajectory(self, env):
        if not self.H_i: self.e_i = 0.0; return
        md = math.sqrt(DIMENSIONS)
        ws = [max(0.0, 1.0 - euclidean_dist(env, pe) / md) for _, pe in self.H_i]
        t  = sum(ws); ws = [w/t for w in ws] if t > 0 else [1.0/len(ws)] * len(ws)
        self.e_i = sum(w * self.H_i[i][0] for i, w in enumerate(ws))

    def compute_mutation(self): self.mu_i = 0.15 * (1.0 - self.e_i) + 0.03

    def mutate(self):
        cp = []
        if self.pwm:
            for j, p in enumerate(self.params):
                cp.append(max(0.0, min(1.0, p + random.gauss(0, self.pwm[j] * (1.0 - self.e_i) + 0.02))))
        else:
            for p in self.params:
                cp.append(max(0.0, min(1.0, p + random.gauss(0, self.mu_i))))
        return Agent(cp, self.pwm)


# ===============================================================
#  ALGORITHM RUNNERS (identical to BENCHMARK_VS_BOTS.py)
# ===============================================================
def run_ccea(server):
    print(f"  [EVOLVING] CCE-TDCCEA...")
    pop_a = [SubAgent(len(MACRO_INDICES)) for _ in range(N_POP)]
    pop_b = [SubAgent(len(MICRO_INDICES)) for _ in range(N_POP)]
    arc_a, arc_b = [], []
    top_a, top_b = pop_a[:TOP_PARTNERS], pop_b[:TOP_PARTNERS]
    best_a, best_b = pop_a[0], pop_b[0]; t0 = time.time()
    for gen in range(1, TOTAL_GENERATIONS + 1):
        opp = OPPONENT_ROTATION[(gen-1) % len(OPPONENT_ROTATION)]
        ca  = [sum(a.params[j] for a in pop_a)/len(pop_a) for j in range(len(MACRO_INDICES))]
        cb  = [sum(b.params[j] for b in pop_b)/len(pop_b) for j in range(len(MICRO_INDICES))]
        env = ca + cb
        for a in pop_a:
            fs = [evaluate_weights(server, combine_subagents(a, pb), opp) for pb in top_b]
            a.update_history(sum(fs)/len(fs), env); a.compute_trajectory(env); a.compute_mutation()
        for b in pop_b:
            fs = [evaluate_weights(server, combine_subagents(pa, b), opp) for pa in top_a]
            b.update_history(sum(fs)/len(fs), env); b.compute_trajectory(env); b.compute_mutation()
        best_a = max(pop_a, key=lambda a: a.e_i); best_b = max(pop_b, key=lambda a: a.e_i)
        top_a  = sorted(pop_a, key=lambda a: a.e_i, reverse=True)[:TOP_PARTNERS]
        top_b  = sorted(pop_b, key=lambda a: a.e_i, reverse=True)[:TOP_PARTNERS]
        cmb    = evaluate_weights(server, combine_subagents(best_a, best_b), opp)
        print(f"    Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | Cmb={cmb:.3f}")
        pop_a, arc_a = subpop_survival(pop_a, arc_a)
        pop_b, arc_b = subpop_survival(pop_b, arc_b)
    return combine_subagents(best_a, best_b), time.time()-t0


def run_standard_ccea(server):
    print(f"  [EVOLVING] Standard CCEA...")
    MUT = 0.15
    pm = [[random.random() for _ in range(len(MACRO_INDICES))] for _ in range(N_POP)]
    pi = [[random.random() for _ in range(len(MICRO_INDICES))] for _ in range(N_POP)]
    fm = [0.0]*N_POP; fi = [0.0]*N_POP; t0 = time.time()
    def mk(mac, mic):
        f = [0.5]*DIMENSIONS
        for k,idx in enumerate(MACRO_INDICES): f[idx]=mac[k]
        for k,idx in enumerate(MICRO_INDICES):  f[idx]=mic[k]
        return f
    def breed(pop, fits):
        bi = max(range(len(pop)), key=lambda i: fits[i]); new_pop = [pop[bi][:]]
        while len(new_pop)<N_POP:
            t    = random.sample(range(len(pop)), min(3,len(pop)))
            pidx = max(t, key=lambda i: fits[i])
            new_pop.append([max(0.0,min(1.0,x+random.gauss(0,MUT))) for x in pop[pidx]])
        return new_pop
    for gen in range(1, TOTAL_GENERATIONS+1):
        opp = OPPONENT_ROTATION[(gen-1)%len(OPPONENT_ROTATION)]
        tmi = sorted(range(N_POP), key=lambda i: fi[i], reverse=True)[:TOP_PARTNERS]
        tma = sorted(range(N_POP), key=lambda i: fm[i], reverse=True)[:TOP_PARTNERS]
        for i in range(N_POP):
            fs = [evaluate_weights(server,mk(pm[i],pi[j]),opp) for j in tmi]; fm[i]=sum(fs)/len(fs)
        for i in range(N_POP):
            fs = [evaluate_weights(server,mk(pm[j],pi[i]),opp) for j in tma]; fi[i]=sum(fs)/len(fs)
        bm  = max(range(N_POP),key=lambda i:fm[i]); bi2=max(range(N_POP),key=lambda i:fi[i])
        print(f"    Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | Best={evaluate_weights(server,mk(pm[bm],pi[bi2]),opp):.3f}")
        pm = breed(pm,fm); pi = breed(pi,fi)
    bm=max(range(N_POP),key=lambda i:fm[i]); bi2=max(range(N_POP),key=lambda i:fi[i])
    return mk(pm[bm],pi[bi2]), time.time()-t0


def run_cmaes(server):
    print(f"  [EVOLVING] CMA-ES...")
    t0=time.time(); best_p=[0.5]*DIMENSIONS; best_f=0.0
    try:
        import cma
        es=cma.CMAEvolutionStrategy([0.5]*DIMENSIONS,0.3,
            {'maxiter':TOTAL_GENERATIONS,'popsize':N_POP,'bounds':[0,1],'verbose':-9,'seed':random.randint(0,99999)})
        gen=0
        while not es.stop() and gen<TOTAL_GENERATIONS:
            opp=OPPONENT_ROTATION[gen%len(OPPONENT_ROTATION)]; sols=es.ask(); fits=[]
            for s in sols:
                sc=[max(0.0,min(1.0,x)) for x in s]; f=evaluate_weights(server,sc,opp); fits.append(f)
                if f>best_f: best_f=f; best_p=sc[:]
            es.tell(sols,[-f for f in fits])
            print(f"    Gen {gen+1:2d}/{TOTAL_GENERATIONS} | {opp:25s} | Best={best_f:.3f}"); gen+=1
    except ImportError:
        pop=[[random.random() for _ in range(DIMENSIONS)] for _ in range(N_POP)]; sigma=0.3
        for gen in range(1,TOTAL_GENERATIONS+1):
            opp=OPPONENT_ROTATION[(gen-1)%len(OPPONENT_ROTATION)]; fits=[0.0]*N_POP
            for i,s in enumerate(pop):
                fits[i]=evaluate_weights(server,s,opp)
                if fits[i]>best_f: best_f=fits[i]; best_p=s[:]
            ranked=sorted(range(N_POP),key=lambda i:fits[i],reverse=True)
            parents=[pop[i] for i in ranked[:N_POP//2]]; pop=parents[:]
            while len(pop)<N_POP:
                p=random.choice(parents); pop.append([max(0.0,min(1.0,x+random.gauss(0,sigma))) for x in p])
            sigma=max(0.05,sigma*0.98)
            print(f"    Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | Best={best_f:.3f}")
    return best_p, time.time()-t0


def run_de(server):
    print(f"  [EVOLVING] DE/rand/1/bin...")
    F=0.8; CR=0.9; t0=time.time()
    pop=[[random.random() for _ in range(DIMENSIONS)] for _ in range(N_POP)]
    fit=[0.0]*N_POP
    for i in range(N_POP): fit[i]=evaluate_weights(server,pop[i],OPPONENT_ROTATION[0])
    for gen in range(1,TOTAL_GENERATIONS+1):
        opp=OPPONENT_ROTATION[(gen-1)%len(OPPONENT_ROTATION)]
        for i in range(N_POP):
            a,b,c=random.sample([j for j in range(N_POP) if j!=i],3)
            mutant=[max(0.0,min(1.0,pop[a][j]+F*(pop[b][j]-pop[c][j]))) for j in range(DIMENSIONS)]
            jr=random.randint(0,DIMENSIONS-1)
            trial=[mutant[j] if(random.random()<CR or j==jr) else pop[i][j] for j in range(DIMENSIONS)]
            tf=evaluate_weights(server,trial,opp)
            if tf>=fit[i]: pop[i]=trial; fit[i]=tf
        print(f"    Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | Best={max(fit):.3f}")
    bi=max(range(N_POP),key=lambda i:fit[i]); return pop[bi], time.time()-t0


def run_pso(server):
    print(f"  [EVOLVING] PSO...")
    W=0.7;C1=1.5;C2=1.5; t0=time.time()
    pos=[[random.random() for _ in range(DIMENSIONS)] for _ in range(N_POP)]
    vel=[[random.uniform(-0.1,0.1) for _ in range(DIMENSIONS)] for _ in range(N_POP)]
    pb=[p[:] for p in pos]; pf=[0.0]*N_POP; gb=pos[0][:]; gf=0.0
    for gen in range(1,TOTAL_GENERATIONS+1):
        opp=OPPONENT_ROTATION[(gen-1)%len(OPPONENT_ROTATION)]
        for i in range(N_POP):
            f=evaluate_weights(server,pos[i],opp)
            if f>pf[i]: pf[i]=f; pb[i]=pos[i][:]
            if f>gf: gf=f; gb=pos[i][:]
        for i in range(N_POP):
            for j in range(DIMENSIONS):
                vel[i][j]=W*vel[i][j]+C1*random.random()*(pb[i][j]-pos[i][j])+C2*random.random()*(gb[j]-pos[i][j])
                pos[i][j]=max(0.0,min(1.0,pos[i][j]+vel[i][j]))
        print(f"    Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | Best={gf:.3f}")
    return gb, time.time()-t0


def evolve_core(server, label, pwm=None):
    print(f"  [EVOLVING] {label}...")
    population=[Agent(pwm=pwm) for _ in range(N_POP)]; archive=[]; t0=time.time()
    for gen in range(1,TOTAL_GENERATIONS+1):
        opp=OPPONENT_ROTATION[(gen-1)%len(OPPONENT_ROTATION)]
        centroid=[sum(a.params[j] for a in population)/len(population) for j in range(DIMENSIONS)]
        for agent in population:
            f_i=evaluate_weights(server,agent.params,opp)
            agent.update_history(f_i,centroid); agent.compute_trajectory(centroid); agent.compute_mutation()
        best=max(population,key=lambda a:a.e_i)
        print(f"    Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | e_i={best.e_i:.3f}")
        md=math.sqrt(DIMENSIONS); survivors=[]
        for a in sorted(population,key=lambda a:a.e_i,reverse=True):
            if not any(1.0-euclidean_dist(a.params,u.params)/md>DELTA_S for u in survivors):
                survivors.append(a)
        final=[a for a in survivors if not(a.e_i<DELTA_F and len(a.H_i)>=3)]
        while len(final)<10 and len(survivors)>len(final):
            for a in sorted(survivors,key=lambda a:a.e_i,reverse=True):
                if a not in final: final.append(a); break
        for top in sorted(population,key=lambda a:a.e_i,reverse=True)[:3]:
            ar=Agent(top.params,pwm); ar.e_i=top.e_i; ar.H_i=top.H_i[:]; ar.mu_i=top.mu_i
            archive.append(ar)
        archive.sort(key=lambda a:a.e_i,reverse=True); archive=archive[:A_MAX]
        pool=final+archive; population=final[:]
        while len(population)<N_POP:
            cands=random.sample(pool,min(3,len(pool)))
            population.append(max(cands,key=lambda a:a.e_i).mutate())
    best_all=max(archive,key=lambda a:a.e_i) if archive else max(population,key=lambda a:a.e_i)
    return best_all.params, time.time()-t0


def run_hill_climbing(server):
    print(f"  [EVOLVING] Hill Climbing...")
    current=[random.random() for _ in range(DIMENSIONS)]; cfit=0.0; t0=time.time()
    for gen in range(1,TOTAL_GENERATIONS+1):
        opp=OPPONENT_ROTATION[(gen-1)%len(OPPONENT_ROTATION)]
        cfit=evaluate_weights(server,current,opp); bc,bf=current[:],cfit
        for _ in range(N_POP-1):
            c=[max(0.0,min(1.0,p+random.gauss(0,0.15))) for p in current]
            f=evaluate_weights(server,c,opp)
            if f>bf: bc,bf=c[:],f
        if bf>=cfit: current,cfit=bc[:],bf
        print(f"    Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | Best={cfit:.3f}")
    return current, time.time()-t0


# ===============================================================
#  HEAD-TO-HEAD: one matchup between two evolved agents
# ===============================================================
def play_head_to_head(server, w1, w2, num_games=H2H_GAMES):
    """
    Play num_games with alternating sides for fairness.
    Returns (p1_wins, p2_wins, draws, errors).
    [FIX 3] errors counted separately — high error count = server doesn't support format
    """
    p1w = p2w = draws = errors = 0
    half = num_games // 2

    # First half: w1 is player 0
    for _ in range(half):
        winner, _ = server.run_evolved_vs_evolved(w1, w2)
        if winner == 0:   p1w   += 1
        elif winner == 1: p2w   += 1
        elif winner == -1: errors += 1   # [FIX 3] count errors
        else:             draws  += 1

    # Second half: w2 is player 0 (sides swapped for fairness)
    for _ in range(num_games - half):
        winner, _ = server.run_evolved_vs_evolved(w2, w1)
        if winner == 1:   p1w   += 1    # w1 is player 1 in swapped game
        elif winner == 0: p2w   += 1    # w2 is player 0 in swapped game
        elif winner == -1: errors += 1
        else:             draws  += 1

    return p1w, p2w, draws, errors


# ===============================================================
#  BOT-BASED PROXY RANKING (fallback if server doesn't support H2H)
# ===============================================================
def run_proxy_ranking(server, evolved):
    """
    [FIX 2] Fallback: if evolved-vs-evolved not supported,
    validate each agent against all bots and rank by overall win rate.
    This is less direct but still meaningful for paper results.
    """
    print(f"\n  Using bot-validation proxy ranking (evolved-vs-evolved not supported)")
    print(f"  Validating each agent against all 7 bots ({VALIDATION_GAMES} games each)...")
    n = len(evolved)
    for i, agent in enumerate(evolved):
        wr, per_opp = validate_weights(server, agent["weights"])
        agent["proxy_wr"]      = wr
        agent["proxy_per_opp"] = per_opp
        print(f"  [{i+1}/{n}] {agent['name']:18s}: {wr:.1f}%")

    ranked = sorted(range(n), key=lambda i: evolved[i]["proxy_wr"], reverse=True)
    medals = ["1st","2nd","3rd","4th","5th","6th","7th","8th"]

    print(f"\n{'='*70}")
    print(f"  PROXY RANKING (Bot Win Rate — evolved-vs-evolved unavailable)")
    print(f"{'='*70}")
    for rank, i in enumerate(ranked):
        tag = " ◄ OUR ALGORITHM" if evolved[i]["name"] == "CCE-TDCCEA" else ""
        print(f"  {medals[rank]:<5s} {evolved[i]['name']:18s}  {evolved[i]['proxy_wr']:5.1f}%{tag}")


# ===============================================================
#  MAIN
# ===============================================================
def main():
    print("\n" + "#" * 70)
    print("  HEAD-TO-HEAD TOURNAMENT — 8 Algorithms")
    print("  Phase 1: Evolve each algorithm against bots")
    print("  Phase 2: Round-robin between evolved agents")
    print("#" * 70)

    sa_mut = load_sensitivity()
    if sa_mut is None:
        print("  NOTE: SA-TDCCEA will use uniform mutation (run BENCHMARK_VS_BOTS.py first).")

    algos = [
        ("CCE-TDCCEA",    lambda s: run_ccea(s)),
        ("Standard CCEA", lambda s: run_standard_ccea(s)),
        ("CMA-ES",        lambda s: run_cmaes(s)),
        ("DE/rand/1/bin", lambda s: run_de(s)),
        ("PSO",           lambda s: run_pso(s)),
        ("TD-CCEA",       lambda s: evolve_core(s, "TD-CCEA")),
        ("SA-TDCCEA",     lambda s: evolve_core(s, "SA-TDCCEA", pwm=sa_mut)),
        ("Hill Climbing", lambda s: run_hill_climbing(s)),
    ]

    # ── PHASE 1: EVOLUTION ──────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PHASE 1: EVOLUTION — each algorithm trains against bots")
    print(f"{'='*70}")

    evolved = []
    for name, run_fn in algos:
        server = GameServer(); server.start()
        best, elapsed = run_fn(server)
        server.stop()
        evolved.append({"name": name, "weights": best, "time": elapsed})
        print(f"  [DONE] {name} — {elapsed:.1f}s\n")

    # ── PHASE 2: CHECK SERVER SUPPORT THEN TOURNAMENT ───────────
    print(f"\n{'='*70}")
    print(f"  PHASE 2: HEAD-TO-HEAD ROUND-ROBIN")
    print(f"{'='*70}")

    server = GameServer(); server.start()

    # [FIX 1] Probe server for evolved-vs-evolved support
    print(f"\n  Checking if server supports evolved-vs-evolved...")
    h2h_supported = server.check_evolved_vs_evolved_support()

    if not h2h_supported:
        # [FIX 2] Graceful fallback
        print(f"\n  ╔══════════════════════════════════════════════════════════╗")
        print(f"  ║  EVOLVED-VS-EVOLVED NOT SUPPORTED BY JAVA SERVER        ║")
        print(f"  ║                                                          ║")
        print(f"  ║  Your EvolvedBatchGameRunner does not support:           ║")
        print(f"  ║  EVOLVED:{{w1}},EVOLVED:{{w2}},{{map}},{{cycles}}             ║")
        print(f"  ║                                                          ║")
        print(f"  ║  To enable it, update EvolvedBatchGameRunner.java to     ║")
        print(f"  ║  detect 'EVOLVED:' as the opponent and create a second   ║")
        print(f"  ║  EvolvedBrainAI with those weights.                      ║")
        print(f"  ║                                                          ║")
        print(f"  ║  FALLING BACK to bot-validation proxy ranking.           ║")
        print(f"  ╚══════════════════════════════════════════════════════════╝")
        run_proxy_ranking(server, evolved)
        server.stop()
        return

    print(f"  Server supports evolved-vs-evolved. Starting round-robin...")
    print(f"  {H2H_GAMES} games per matchup ({H2H_GAMES//2} each side)\n")

    n             = len(evolved)
    wins_matrix   = [[0]*n for _ in range(n)]
    draws_matrix  = [[0]*n for _ in range(n)]
    errors_matrix = [[0]*n for _ in range(n)]    # [FIX 3]
    total_wins    = [0] * n
    total_played  = [0] * n
    total_errors  = [0] * n

    for i in range(n):
        for j in range(i+1, n):
            print(f"  {evolved[i]['name']:18s} vs {evolved[j]['name']:18s} ... ", end="", flush=True)
            p1w, p2w, draws, errors = play_head_to_head(
                server, evolved[i]["weights"], evolved[j]["weights"])

            wins_matrix[i][j]   = p1w
            wins_matrix[j][i]   = p2w
            draws_matrix[i][j]  = draws_matrix[j][i] = draws
            errors_matrix[i][j] = errors_matrix[j][i] = errors

            total_wins[i]   += p1w
            total_wins[j]   += p2w
            total_played[i] += H2H_GAMES
            total_played[j] += H2H_GAMES
            total_errors[i] += errors
            total_errors[j] += errors

            # [FIX 5] Warn if errors dominate
            if errors == H2H_GAMES:
                print(f"ALL ERRORS — server returned -1 for every game!")
            else:
                print(f"W:{p1w}-{p2w}  D:{draws}  E:{errors}")

    server.stop()

    # [FIX 5] Check if everything failed
    if sum(total_wins) == 0 and sum(total_errors) > 0:
        print(f"\n  ╔══════════════════════════════════════════════════════════╗")
        print(f"  ║  ALL GAMES RETURNED ERROR (-1)                           ║")
        print(f"  ║  The probe passed but actual games failed.               ║")
        print(f"  ║  Results are meaningless — check Java server logs.       ║")
        print(f"  ╚══════════════════════════════════════════════════════════╝")
        return

    # ── RESULTS MATRIX ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ROUND-ROBIN RESULTS MATRIX (wins / draws)")
    print(f"{'='*70}")

    header = f"  {'':18s}"
    for e in evolved: header += f" {e['name'][:8]:>10s}"
    header += f"  {'WINS':>6s}  {'WR%':>6s}  {'ERR':>5s}"
    print(header)
    print("  " + "-" * (18 + 11*n + 22))

    for i in range(n):
        row = f"  {evolved[i]['name']:18s}"
        for j in range(n):
            if i == j:
                row += f"      ---  "
            else:
                # [FIX 4] Show wins/draws
                row += f"  {wins_matrix[i][j]:3d}/{draws_matrix[i][j]:2d}d"
        wr  = total_wins[i] / total_played[i] * 100 if total_played[i] > 0 else 0
        row += f"  {total_wins[i]:>6d}  {wr:>5.1f}%  {total_errors[i]:>5d}"
        print(row)

    # ── FINAL RANKING ────────────────────────────────────────────
    ranking = sorted(range(n), key=lambda i: total_wins[i], reverse=True)
    medals  = ["1st","2nd","3rd","4th","5th","6th","7th","8th"]

    print(f"\n{'='*70}")
    print(f"  HEAD-TO-HEAD RANKING")
    print(f"{'='*70}")
    for rank, i in enumerate(ranking):
        wr  = total_wins[i] / total_played[i] * 100 if total_played[i] > 0 else 0
        tag = " ◄ OUR ALGORITHM" if evolved[i]["name"] == "CCE-TDCCEA" else ""
        print(f"  {medals[rank]:<5s} {evolved[i]['name']:18s}  "
              f"W:{total_wins[i]:3d}/{total_played[i]}  ({wr:.1f}%){tag}")

    # ── SAVE RESULTS ─────────────────────────────────────────────
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "head_to_head_results.json")
    save = {
        "algo_names":   [e["name"] for e in evolved],
        "wins_matrix":  wins_matrix,
        "draws_matrix": draws_matrix,
        "total_wins":   total_wins,
        "total_played": total_played,
        "total_errors": total_errors,
    }
    with open(out, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n  Results saved → {out}")
    print(f"\n{'='*70}")
    print(f"  TOURNAMENT COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()