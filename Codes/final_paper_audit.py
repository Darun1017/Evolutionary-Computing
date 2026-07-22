print("="*65)
print("SO__5_.pdf — FINAL COMPLETE ANALYSIS")
print("="*65)

# FIXES confirmed in this version
print("\n✅ CONFIRMED FIXES vs SO_4_.pdf:")
print("-"*65)
fixed = [
    ("Page 1", "CRITICAL FIXED", "'last 10 generations' → 'last 7 generations' — fixed after 5 versions"),
    ("Page 4", "CRITICAL FIXED", "Algorithm 1 Line 13: 'top-m' → 'top-3' — now defined"),
    ("Page 4", "CRITICAL FIXED", "Section IV: 'Cooperative' → 'Competitive' — name now consistent"),
    ("Page 4", "HIGH FIXED",    "'five generations' × 3 → 'seven generations' everywhere"),
    ("Page 4", "HIGH FIXED",    "Section IV.A: 'TDCCEA' → 'TD-CCEA' — hyphen added"),
    ("Page 6", "MEDIUM FIXED",  "Hardware language: now formally written"),
    ("Page 6", "MEDIUM FIXED",  "Opponent rotation: 'one bot per generation in a rotating schedule'"),
    ("Page 7", "HIGH FIXED",    "Table V replaced with Z-stat + effect size r format"),
    ("Page 2", "HIGH PARTIAL",  "Duplicate author in GESY removed — but tilde ˜ remains"),
]
for page, status, detail in fixed:
    print(f"  [{page}] {status}: {detail}")

# NAME CONSISTENCY CHECK
print("\n\n📋 ALGORITHM NAME CONSISTENCY CHECK:")
print("-"*65)
names = [
    ("Title",           "Trajectory-Driven Competitive Co-Evolutionary", "✅"),
    ("Abstract",        "Trajectory-Driven Competitive Co-Evolutionary", "✅"),
    ("Intro Page 1",    "Trajectory-Driven Competitive Co-Evolutionary", "✅"),
    ("Algorithm 1",     "Trajectory-Driven Competitive Co-Evolutionary", "✅"),
    ("Section IV",      "Trajectory-Driven Competitive Coevolutionary",  "⚠️  Missing hyphen: Co-Evolutionary vs Coevolutionary"),
    ("Index Terms",     "Competitive co-evolution",                       "✅"),
]
for loc, name, verdict in names:
    print(f"  {verdict} [{loc}]: {name}")

print("\n  Note: Title/Abstract/Algo1 use 'Co-Evolutionary' (hyphenated)")
print("        Section IV uses 'Coevolutionary' (no hyphen) — minor inconsistency")

# REMAINING PROBLEMS
print("\n\n❌ REMAINING PROBLEMS:")
print("-"*65)
problems = [
    (3, "CRITICAL", "Section III.I, last sentence",
     "Citation is now [] (empty brackets) instead of [?]. Still unfilled. Changed symbol but content not filled.",
     "Fill with: [1] M.A. Hevia Fajardo and P.K. Lehre, 'Runtime Analysis of a Co-Evolutionary Algorithm: Overcoming Negative Drift in Maximin-Optimisation,' 2023."),

    (2, "HIGH", "Section II.A, GESY heading",
     "'Marino and Toledo (2022) – GESY: ˜ a genetic programming approach' — tilde ˜ artifact still present after author duplicate removed. Looks like a LaTeX citation command that wasn't resolved.",
     "Remove the ˜ symbol. Should read: 'Marino and Toledo (2022) – GESY: a genetic programming approach...'"),

    (4, "HIGH", "Section III.L — planned vs actual analysis mismatch",
     "Section III.L describes analyses never performed: (1) fitness well plotting, (2) Nash equilibrium divergence score computation, (3) O(n^1.5) time complexity verification. None of these appear in Section VI Results. A reviewer reading III.L will look for these results and find only win rate tables.",
     "Either: (A) Remove sub-sections III.L.2, III.L.3 since Cui et al. RL comparison and Nash divergence analysis were never executed, or (B) Add a sentence in Results: 'The Nash equilibrium analysis described in Section III.L.3 is reserved for future work; the present evaluation focuses on win rate comparison.'"),

    (6, "HIGH", "Section V, ablation bullet point",
     "'time-based trajectory analysis' — wrong terminology. The algorithm is trajectory-driven, not time-based. 'Time-based' implies time decay which is not what the algorithm does.",
     "Change to 'trajectory-based analysis' throughout."),

    (7, "HIGH", "Results, paragraph after Table IV",
     "'both approaches were excluded from the final evaluation and are reserved for future work for this comparison and might be used for future studies' — 'for this comparison' is grammatically incorrect here.",
     "Remove 'for this comparison'. Should read: 'both approaches were excluded from the final evaluation and are reserved for future investigation.'"),

    (8, "HIGH", "Conclusion, paragraph 5",
     "'ahead of every baseline tested — including...Hill Climbing (69.71%)' — Table V explicitly shows Hill Climbing p=0.065, n.s. (not significant). Claiming 'every baseline' without qualification contradicts your own Table V.",
     "Add qualifier: '...and Hill Climbing (69.71%, p=0.065, n.s.), where CCE-TDCCEA\\'s higher mean win rate (80.11% vs 69.71%) and lower variance (±11.17 vs ±18.62) demonstrate a practical advantage despite borderline statistical significance.'"),

    (7, "MEDIUM", "Table V, Base and Ours columns",
     "Table V reports median values (Ours = 82.9%) but Table IV reports mean values (CCE-TDCCEA Avg = 80.11%). Two different numbers for the same algorithm appear in the same results section. Note at bottom says 'Mdn' but a reader scanning tables will see 80.11 and 82.9 and be confused.",
     "Remove Base and Ours columns from Table V. The win rates are already in Table IV. Or add a note directly in Table V header: 'Median win rates; mean values in Table IV.'"),

    (8, "MEDIUM", "Conclusion, paragraphs 2 and 3",
     "Two full paragraphs re-explain the trajectory scoring mechanism ('The trajectory score is computed by measuring how similar...'). Conclusions should not re-describe the methodology.",
     "Condense to one sentence: 'The trajectory mechanism stabilises fitness evaluation by down-weighting historical performance from dissimilar environments, making selection pressure more reliable than single-game outcomes allow.'"),

    (8, "LOW", "Conclusion, last sentence",
     "'testing whether the approach transfers to more complex strategic settings like Diplomacy and Battleship' — speculative and disconnected from paper scope.",
     "Replace with: 'extending to additional microRTS map configurations, scaling the cooperative decomposition to three or more sub-populations, and investigating transfer to other real-time strategy simulators.'"),

    (1, "LOW", "Introduction, paragraph 4",
     "'Trajectory-Driven Competitive Co-Evolutionary Algorithm(TD-CCEA)' — missing space before opening parenthesis.",
     "Add space: 'Algorithm (TD-CCEA)'"),
]

crit = sum(1 for p,s,*_ in problems if s=="CRITICAL")
high = sum(1 for p,s,*_ in problems if s=="HIGH")
med  = sum(1 for p,s,*_ in problems if s=="MEDIUM")
low  = sum(1 for p,s,*_ in problems if s=="LOW")

print(f"\nTotal: {len(problems)}  |  Critical:{crit}  High:{high}  Medium:{med}  Low:{low}\n")

for page,sev,loc,prob,fix in problems:
    flag = "🔴" if sev=="CRITICAL" else "🟠" if sev=="HIGH" else "🟡" if sev=="MEDIUM" else "🔵"
    print(f"{flag} [{sev}] Page {page} — {loc}")
    print(f"   Problem: {prob}")
    print(f"   Fix:     {fix}")
    print()

# VERSION PROGRESS
print("="*65)
print("PROGRESS ACROSS ALL VERSIONS:")
print("="*65)
versions = [
    ("SO_1_",                    "~21 issues", "First draft — 6D vector, no Wilcoxon, wrong claims"),
    ("SO_2_",                    "~18 issues", "Wilcoxon added, parameters added"),
    ("SO_3_",                    "~14 issues", "6D→20D fixed, pruning order, section renamed"),
    ("project_phase_paper__2_",  "~13 issues", "CCE formally described, SA formulas fixed, DP added"),
    ("SO_4_",                    "14 issues",  "Partially fixed Section IV name, Table V format improved"),
    ("SO_5_ (current)",          f"{len(problems)} issues",  "W=7 fixed, top-3 fixed, seven gen fixed, Table V improved"),
]
for ver, count, note in versions:
    print(f"  {ver:<30} {count:<12} {note}")

print(f"\n  Remaining blockers before submission: 1 critical + 5 high = 6 items")
print(f"  Paper is ~85% ready for submission")
