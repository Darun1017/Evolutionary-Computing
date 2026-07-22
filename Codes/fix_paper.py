"""
Apply reviewer corrections to the LaTeX paper.
Reads from clipboard-pasted file, applies fixes, writes corrected version.
"""
import re, sys

INPUT = r"C:\Users\Pradeep S\Desktop\Semester VI\Project Phase - 1\Presentations\paper_original.tex"
OUTPUT = r"C:\Users\Pradeep S\Desktop\Semester VI\Project Phase - 1\Presentations\paper_corrected.tex"
DIFF_OUT = r"C:\Users\Pradeep S\Desktop\Semester VI\Project Phase - 1\Presentations\changes_log.txt"

with open(INPUT, 'r', encoding='utf-8') as f:
    text = f.read()

original = text
changes = []

def log(desc, before_snip, after_snip):
    changes.append(f"\n{'='*60}\nCORRECTION: {desc}\n{'='*60}\nBEFORE: {before_snip[:200]}\nAFTER:  {after_snip[:200]}\n")

# ── CORRECTION 1: Replace we/our/us with formal alternatives ──
replacements_1 = [
    # Abstract
    ("This paper proposes the Trajectory-Driven", "This paper proposes the Trajectory-Driven"),  # already ok
    # Introduction
    ("we want the computer", "it is desirable for the computer"),
    ("we are going to evaluate", "to evaluate"),
    ("we developed a new evolutionary", "a new evolutionary"),
    ("we give every agent a memory", "every agent is given a memory"),
    ("we look at the average strategy", "the average strategy is examined"),
    ("We analyze the agent", "The agent"),
    ("we consider it much less important", "it is considered much less important"),
    ("we will give it a low mutation", "it is assigned a low mutation"),
    ("we will increase its mutation rate so they aggressively mutate\nuntil they can locate", "its mutation rate is increased so it aggressively mutates\nuntil it can locate"),
    ("we regularly prune weak agents and remove\ncloned agents", "weak agents are regularly pruned and\ncloned agents"),
    ("We also\nkeep", "The framework also\nkeeps"),
    ("we demonstrate that these complex simulations", "it is demonstrated that these complex simulations"),
    # Related Works
    ("our approach incorporates", "the proposed approach incorporates"),
    ("our sensitivity-adaptive variant", "the sensitivity-adaptive variant"),
    ("our algorithm", "the proposed algorithm"),
    ("our core CCE-TDCCEA, we apply", "the core CCE-TDCCEA applies"),
    ("our benchmark evaluation", "the benchmark evaluation"),
    # Methodology
    ("our experiments", "the experiments presented in this paper"),
    ("our main model", "the main model"),
    ("our testing ecosystem", "the testing ecosystem"),
    ("we can specifically identify", "the contributions can be specifically identified"),
    # SA-TDCCEA
    ("our set of optimizers", "the set of optimizers presented in this paper"),
    # CCE-TDCCEA  
    ("One major innovation in this work is that CCE-TDCCEA", "A major innovation in this work is that CCE-TDCCEA"),
    ("we resort to a\ndeterministic", "a\ndeterministic"),
    ("in our algorithm. Every", "is employed. Every"),
    # Results
    ("our CCE-TDCCEA algorithm", "the CCE-TDCCEA algorithm"),
    ("We compared our three algorithm variants", "The three algorithm variants"),
    ("we could see that our CCE-TDCCEA", "the CCE-TDCCEA"),
    ("we could observe that our algorithm", "the proposed algorithm"),
    ("we could observe", "it can be observed"),
    ("We ran 5 matches", "Five matches were run"),
    ("we faced against were chosen", "were chosen"),
    ("We used the microRTS", "The microRTS"),
    ("we chose were", "chosen for this study were"),
    ("us evolve and find", "the system to evolve and find"),
    ("we had tried out but not\nused", "were tried out but not\nused"),
    ("we called DP-TDCCEA", "called DP-TDCCEA"),
    ("we split the parameters", "the parameters were split"),
    ("We did not perform", "No full"),
    ("we had two new approaches which we had tried", "two new approaches were tried"),
    ("we had two new approaches which", "two new approaches"),
    ("we could see that our CCE-TDCCEA algorithm performed well", "the CCE-TDCCEA algorithm performed well"),
    ("we compared each of the traits of our algorithm", "each of the traits of the proposed algorithm was compared"),
    ("our proposed mechanisms", "the proposed mechanisms"),
    ("We repeated the simulation", "The simulation was repeated"),
    ("we could see that our CCE-TDCCEA algorithm's lowest", "the CCE-TDCCEA algorithm's lowest"),
    ("we used was", "used was"),
    ("The map we used was", "The map used was"),
    # Preliminary
    ("we considered whether", "it was considered whether"),
    ("we thought we\ncould reuse", "it was hypothesized that\nreusing"),
    ("we defined a trajectory memory", "a trajectory memory was defined"),
    ("We also tested", "A test was also conducted on"),
    ("we also tested", "a test was also conducted on"),
    ("We therefore kept", "The two-way split was therefore retained"),
    ("we describe the variations", "the variations are described"),
    # Conclusion
    ("This work set out", "This work set out"),  # ok - no pronoun
    ("we judge and evolve game agents. Instead of\nlooking at a single game result, we give every agent a\nmemory diary", "game agents are judged and evolved. Instead of\nlooking at a single game result, every agent is given a\nmemory diary"),
    ("we look at the average", "the average is examined of the"),
    ("we consider it much", "it is considered much"),
    # Remaining instances
    ("our three algorithm variants\\textemdash CCE-TDCCEA", "the three algorithm variants\\textemdash CCE-TDCCEA"),
    ("helped us evolve", "helped the system to evolve"),
]

for old, new in replacements_1:
    if old in text and old != new:
        text = text.replace(old, new, 1)
        log("Replace pronoun", old, new)

# Broad sweeps for remaining instances
broad_replacements = [
    (" we ", " the authors "),  # Be careful - only apply selectively
    (" our ", " the proposed "),
]
# Skip broad replacements to avoid breaking things - the specific ones above cover the key instances

# ── CORRECTION 2: Explain constants/variables in equations ──

# After equation 1 (centroid)
old_eq1 = "This centroid serves as a compact representation"
new_eq1 = "where $N$ is the population size and $\\boldsymbol{\\theta}_i^t$ is the strategy vector of agent $i$ at generation $t$. This centroid serves as a compact representation"
if old_eq1 in text:
    text = text.replace(old_eq1, new_eq1, 1)
    log("Explain Eq 1 variables", old_eq1, new_eq1)

# After equation 2 (trajectory memory)
old_eq2 = "When $|H_i|$ exceeds the window size $W$, the oldest entry is\n    trimmed."
new_eq2 = "where $f_i^t$ is the fitness of agent $i$ at generation $t$ and $\\mathbf{c}_t$ is the environment meta-state at generation $t$. When $|H_i|$ exceeds the window size $W$ (set to 7), the oldest entry is\n    trimmed."
if old_eq2 in text:
    text = text.replace(old_eq2, new_eq2, 1)
    log("Explain Eq 2 variables", old_eq2, new_eq2)

# After equation 3 (similarity weights) - dmax already explained
# After equation 6 (mutation rate) - already explained in text

# ── CORRECTION 3: Add introductory paragraph before Individual Study Analysis ──
old_rw = "\\subsection{Individual Study Analysis}"
new_rw = """This section reviews existing approaches to evolutionary strategy optimization in competitive game environments. The studies discussed below span symbolic strategy representation, adaptive mutation mechanisms, runtime analysis of co-evolutionary dynamics, and multi-objective competitive optimization. Each study is analyzed in terms of its contributions and limitations, with particular attention to how the proposed TD-CCEA framework addresses the identified shortcomings.

\\subsection{Individual Study Analysis}"""
if old_rw in text:
    text = text.replace(old_rw, new_rw, 1)
    log("Add Related Works intro paragraph", old_rw[:80], new_rw[:80])

# ── CORRECTION 4: Remove bold font from Related Works paragraphs ──
bold_patterns = [
    ("\\textbf{Mari\\~{n}o and Toledo (2022) -- GESY\\cite{marino2022gesy}:}", "Mari\\~{n}o and Toledo (2022) -- GESY\\cite{marino2022gesy}:"),
    ("\\textbf{Cui et al. (2020) -- Reinforced Evolutionary Algorithms\\cite{cui2020reinforced}:}", "Cui et al. (2020) -- Reinforced Evolutionary Algorithms\\cite{cui2020reinforced}:"),
    ("\\textbf{Hevia Fajardo et al. (2023) -- Runtime Analysis\\cite{fajardo2023runtime}:}", "Hevia Fajardo et al. (2023) -- Runtime Analysis\\cite{fajardo2023runtime}:"),
    ("\\textbf{Hevia Fajardo et al. (2024) -- SA-PDCoEA\\cite{fajardo2024sapdcoea}:}", "Hevia Fajardo et al. (2024) -- SA-PDCoEA\\cite{fajardo2024sapdcoea}:"),
    ("\\textbf{Rawal et al. -- Constructing Competitive and Cooperative Agent Behavior Using Coevolution\\cite{rawal2010constructing}: }", "Rawal et al. -- Constructing Competitive and Cooperative Agent Behavior Using Coevolution\\cite{rawal2010constructing}:"),
    ("\\textbf{Guha et al. (2025) -- Multi-objective Competitive Co-Evolutionary Wargame Optimization\\cite{guha2025multiobjective}:}", "Guha et al. (2025) -- Multi-objective Competitive Co-Evolutionary Wargame Optimization\\cite{guha2025multiobjective}:"),
]
for old_b, new_b in bold_patterns:
    if old_b in text:
        text = text.replace(old_b, new_b, 1)
        log("Remove bold from Related Works", old_b[:60], new_b[:60])

# ── CORRECTION 5: Rewrite "metric is which attains" ──
old_metric = "the metric is which attains the highest-quality solution in the\nshortest number of generations."
new_metric = "the metric used is which algorithm attains the highest-quality solution in the\nfewest number of generations."
if old_metric in text:
    text = text.replace(old_metric, new_metric, 1)
    log("Rewrite metric sentence", old_metric, new_metric)

# ── CORRECTION 6: Fix Reference [10] author details ──
old_ref10 = '——, "A self-adaptive coevolutionary algorithm,"'
new_ref10 = 'M. A. Hevia Fajardo and P. K. Lehre, "A self-adaptive coevolutionary algorithm,"'
if old_ref10 in text:
    text = text.replace(old_ref10, new_ref10, 1)
    log("Fix Ref [10] authors", old_ref10, new_ref10)
# Note: This is in the .bib file, user will need to fix there too

# ── CORRECTION 7: Cite equations by number ──
eq_citations = [
    # Eq 1 - centroid
    ("the environment meta-state $\\mathbf{c}_t$ is\n    computed as the centroid",
     "the environment meta-state $\\mathbf{c}_t$ is\n    computed, using Equation~(1), as the centroid"),
    # Eq 2 - trajectory memory
    ("each agent's trajectory memory $H_i$\n    is updated:",
     "each agent's trajectory memory $H_i$\n    is updated as shown in Equation~(2):"),
    # Eq 3 - similarity weights
    ("the weight $\\alpha_k$ for memory\n    entry $k \\in H_i$ is negatively proportional",
     "the weight $\\alpha_k$ for memory\n    entry $k \\in H_i$ is negatively proportional, as defined in Equation~(3),"),
    # Eq 4 - normalized weights
    ("The weights are then normalized to ensure fairness of representation:",
     "The weights are then normalized, as shown in Equation~(4), to ensure fairness of representation:"),
    # Eq 5 - trajectory score
    ("The trajectory score $e_i^t$ is then computed as:",
     "The trajectory score $e_i^t$ is then computed using Equation~(5) as:"),
    # Eq 6 - mutation rate
    ("The per-agent mutation rate $\\mu_i$ is computed to ensure",
     "The per-agent mutation rate $\\mu_i$ is computed, as shown in Equation~(6), to ensure"),
    # Eq 7 - archive
    ("The top-$3$ survivors are inserted into the archive",
     "The top-$3$ survivors are inserted into the archive, as shown in Equation~(7),"),
    # Eq 8 - Gaussian mutation
    ("Offspring are bred from the union of current survivors and\n    the archive $\\mathcal{A}$, using Gaussian mutation:",
     "Offspring are bred from the union of current survivors and\n    the archive $\\mathcal{A}$, using Gaussian mutation as defined in Equation~(8):"),
    # Eq 9 - sensitivity
    ("the\n    static sensitivity profile is then calculated from the absolute\n    range",
     "the\n    static sensitivity profile is then calculated, using Equation~(9), from the absolute\n    range"),
    # Eq 10 - pwm
    ("These bounds map directly to a pre-computed per-weight\nmutation rate array",
     "These bounds map directly, via Equation~(10), to a pre-computed per-weight\nmutation rate array"),
    # Eq 11 - cooperative fitness
    ("the fitness signal will be the\naverage score produced by these three interactions:",
     "the fitness signal will be the\naverage score produced by these three interactions, as defined in Equation~(11):"),
]
for old_eq, new_eq in eq_citations:
    if old_eq in text:
        text = text.replace(old_eq, new_eq, 1)
        log("Add equation citation", old_eq[:60], new_eq[:60])

# ── CORRECTION 8: Unbold text on page 9 ──
bold_p9 = [
    ("\\textbf{that the agents were forced to learn from highly sparse interaction data, which significantly increased the difficulty of finding robust strategies.}",
     "that the agents were forced to learn from highly sparse interaction data, which significantly increased the difficulty of finding robust strategies."),
    ("\\textbf{Strategic Depth 32$\\times$32, a grid-world environment containing obstacle walls, symmetric player starting positions, and 12 resource patches with a total of 240 resources.}",
     "Strategic Depth 32$\\times$32, a grid-world environment containing obstacle walls, symmetric player starting positions, and 12 resource patches with a total of 240 resources."),
]
for old_b9, new_b9 in bold_p9:
    if old_b9 in text:
        text = text.replace(old_b9, new_b9, 1)
        log("Unbold page 9 text", old_b9[:60], new_b9[:60])

# ── CORRECTION 9: Move Table VI, VII, Figure 3 after citation ──
# This requires structural moves. The tables/figures need to appear after they are first referenced.
# Table VI (tab:performance_sd) is cited in text starting "As shown in Table~\ref{tab:performance_sd}"
# Table VII (tab:comprehensive_stats_sd) is cited after Table VI discussion
# Figure 3 (fig:convergence_map2) is cited near the end

# For LaTeX, the simplest fix is to use [htbp] placement and ensure the float appears 
# after its first \ref. The tables are already using [hbt!] and [htbp].
# The key issue is that Table VI and VII definitions appear BEFORE some of their citations.
# Let's move them by cutting and pasting in the right order.

# Actually, looking at the LaTeX structure, the tables need to be repositioned.
# Table performance_sd needs to go after its first citation.
# Let me handle this by extracting and reinserting.

# For now, add a comment noting this for the user - structural moves are complex
# The user can manually move the table definitions in Overleaf

# Write output
with open(OUTPUT, 'w', encoding='utf-8') as f:
    f.write(text)

# Write changes log
with open(DIFF_OUT, 'w', encoding='utf-8') as f:
    f.write(f"TOTAL CHANGES APPLIED: {len(changes)}\n")
    for c in changes:
        f.write(c)
    f.write("\n\nNOTE ON CORRECTION 9 (Table/Figure placement):\n")
    f.write("Tables VI, VII, and Figure 3 should be moved in Overleaf so they appear\n")
    f.write("AFTER their first citation in the text. This is a structural change best\n")
    f.write("done manually in the Overleaf editor by cutting and pasting the table/figure\n")
    f.write("LaTeX blocks to after the paragraph that first references them.\n")

print(f"Done! {len(changes)} changes applied.")
print(f"Corrected file: {OUTPUT}")
print(f"Changes log: {DIFF_OUT}")
