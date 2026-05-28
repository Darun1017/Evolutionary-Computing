/*
 * EvolvedBrainAI.java
 * An AI whose behavior is parameterized by 20 continuous weights [0,1].
 * These weights are evolved by the TD-CCEA Python algorithm.
 *
 * Weight mapping (must match WEIGHT_NAMES in Python):
 *   w0  worker_ratio        — economy vs military balance
 *   w1  max_workers          — worker cap (scaled to 1-12)
 *   w2  expand_threshold     — when to build 2nd base
 *   w3  light_preference     — Light unit priority
 *   w4  heavy_preference     — Heavy unit priority
 *   w5  ranged_preference    — Ranged unit priority
 *   w6  barracks_timing      — when to build barracks (resource threshold)
 *   w7  aggression           — attack eagerness
 *   w8  attack_threshold     — min army size before attacking
 *   w9  defend_radius        — defense perimeter distance
 *   w10 target_priority      — target workers(0) vs military(1)
 *   w11 defense_trigger      — when to pull back to defend
 *   w12 counter_rush         — workers fight during rush
 *   w13 tech_switch          — counter enemy composition
 *   w14 economy_pressure     — prioritize destroying economy
 *   w15 scout_priority       — early scouting
 *   w16 group_attack         — wait for army to group up
 *   w17 second_barracks      — when to build 2nd barracks
 *   w18 worker_fight_ratio   — permanent worker fighters
 *   w19 focus_fire           — concentrate attacks on one target
 */
package ai.abstraction;

import ai.abstraction.pathfinding.AStarPathFinding;
import ai.abstraction.pathfinding.PathFinding;
import ai.core.AI;
import ai.core.ParameterSpecification;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import rts.GameState;
import rts.PhysicalGameState;
import rts.Player;
import rts.PlayerAction;
import rts.units.Unit;
import rts.units.UnitType;
import rts.units.UnitTypeTable;

public class EvolvedBrainAI extends AbstractionLayerAI {

    Random r = new Random();
    protected UnitTypeTable utt;
    UnitType workerType;
    UnitType baseType;
    UnitType barracksType;
    UnitType lightType;
    UnitType heavyType;
    UnitType rangedType;

    // The 20 evolved weights
    double[] weights;

    // Convenience accessors
    double workerRatio()       { return weights[0]; }
    int    maxWorkers()        { return 1 + (int)(weights[1] * 11); } // 1-12
    double expandThreshold()   { return weights[2]; }
    double lightPref()         { return weights[3]; }
    double heavyPref()         { return weights[4]; }
    double rangedPref()        { return weights[5]; }
    double barracksTiming()    { return weights[6]; }
    double aggression()        { return weights[7]; }
    int    attackThreshold()   { return 1 + (int)(weights[8] * 5); } // 1-6 army units
    int    defendRadius()      { return 2 + (int)(weights[9] * 14); } // 2-16 tiles
    double targetPriority()    { return weights[10]; }
    double defenseTrigger()    { return weights[11]; }
    double counterRush()       { return weights[12]; }
    double techSwitch()        { return weights[13]; }
    double economyPressure()   { return weights[14]; }
    double scoutPriority()     { return weights[15]; }
    double groupAttack()       { return weights[16]; }
    double secondBarracks()    { return weights[17]; }
    double workerFightRatio()  { return weights[18]; }
    double focusFire()         { return weights[19]; }

    public EvolvedBrainAI(UnitTypeTable a_utt, double[] a_weights) {
        this(a_utt, new AStarPathFinding(), a_weights);
    }

    public EvolvedBrainAI(UnitTypeTable a_utt, PathFinding a_pf, double[] a_weights) {
        super(a_pf);
        weights = a_weights;
        reset(a_utt);
    }

    public void reset() {
        super.reset();
    }

    public void reset(UnitTypeTable a_utt) {
        utt = a_utt;
        workerType = utt.getUnitType("Worker");
        baseType = utt.getUnitType("Base");
        barracksType = utt.getUnitType("Barracks");
        lightType = utt.getUnitType("Light");
        heavyType = utt.getUnitType("Heavy");
        rangedType = utt.getUnitType("Ranged");
    }

    @Override
    public AI clone() {
        return new EvolvedBrainAI(utt, pf, weights.clone());
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        List<ParameterSpecification> parameters = new ArrayList<>();
        parameters.add(new ParameterSpecification("PathFinding", PathFinding.class, new AStarPathFinding()));
        return parameters;
    }

    // ────────────────────────────────────────────────────────────────
    //  Main action loop
    // ────────────────────────────────────────────────────────────────
    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        Player p = gs.getPlayer(player);

        // Count units
        int nWorkers = 0, nBases = 0, nBarracks = 0, nArmy = 0;
        int nEnemyWorkers = 0, nEnemyArmy = 0;
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player) {
                if (u.getType() == workerType) nWorkers++;
                else if (u.getType() == baseType) nBases++;
                else if (u.getType() == barracksType) nBarracks++;
                else if (u.getType().canAttack) nArmy++;
            } else if (u.getPlayer() >= 0) {
                if (u.getType().canHarvest) nEnemyWorkers++;
                else if (u.getType().canAttack) nEnemyArmy++;
            }
        }

        // Detect if we're being rushed (enemy army/workers near our base early)
        boolean beingRushed = detectRush(player, gs, pgs);

        // ── Base behavior ──
        for (Unit u : pgs.getUnits()) {
            if (u.getType() == baseType && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                baseBehavior(u, p, pgs, nWorkers, nBases, nBarracks);
            }
        }

        // ── Barracks behavior ──
        for (Unit u : pgs.getUnits()) {
            if (u.getType() == barracksType && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                barracksBehavior(u, p, pgs, nArmy);
            }
        }

        // ── Military unit behavior ──
        for (Unit u : pgs.getUnits()) {
            if (u.getType().canAttack && !u.getType().canHarvest
                    && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                militaryBehavior(u, p, gs, pgs, nArmy, beingRushed);
            }
        }

        // ── Worker behavior ──
        List<Unit> workers = new LinkedList<>();
        for (Unit u : pgs.getUnits()) {
            if (u.getType().canHarvest && u.getPlayer() == player) {
                workers.add(u);
            }
        }
        workersBehavior(workers, p, gs, pgs, nBases, nBarracks, nArmy, beingRushed);

        return translateActions(player, gs);
    }

    // ────────────────────────────────────────────────────────────────
    //  Detect enemy rush
    // ────────────────────────────────────────────────────────────────
    private boolean detectRush(int player, GameState gs, PhysicalGameState pgs) {
        if (gs.getTime() > 200) return false; // only early game

        Unit myBase = null;
        for (Unit u : pgs.getUnits()) {
            if (u.getType() == baseType && u.getPlayer() == player) {
                myBase = u;
                break;
            }
        }
        if (myBase == null) return false;

        int threatCount = 0;
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() >= 0 && u.getPlayer() != player && u.getType().canAttack) {
                int d = Math.abs(u.getX() - myBase.getX()) + Math.abs(u.getY() - myBase.getY());
                if (d < defendRadius()) threatCount++;
            }
        }
        return threatCount >= 2;
    }

    // ────────────────────────────────────────────────────────────────
    //  Base: train workers based on workerRatio and maxWorkers
    // ────────────────────────────────────────────────────────────────
    private void baseBehavior(Unit u, Player p, PhysicalGameState pgs,
                              int nWorkers, int nBases, int nBarracks) {
        int workerCap = maxWorkers();
        // If we have barracks, let workerRatio influence the cap
        if (nBarracks > 0) {
            workerCap = Math.max(1, (int)(workerCap * workerRatio()));
        }

        if (nWorkers < workerCap && p.getResources() >= workerType.cost) {
            train(u, workerType);
        }
    }

    // ────────────────────────────────────────────────────────────────
    //  Barracks: produce military units based on preferences
    // ────────────────────────────────────────────────────────────────
    private void barracksBehavior(Unit u, Player p, PhysicalGameState pgs, int nArmy) {
        // Normalize preferences
        double totalPref = lightPref() + heavyPref() + rangedPref();
        if (totalPref == 0) totalPref = 1;

        double lightP = lightPref() / totalPref;
        double heavyP = heavyPref() / totalPref;
        // rangedP is the remainder

        // Roll to decide which unit to produce
        double roll = r.nextDouble();
        UnitType chosen;
        if (roll < lightP) {
            chosen = lightType;
        } else if (roll < lightP + heavyP) {
            chosen = heavyType;
        } else {
            chosen = rangedType;
        }

        if (p.getResources() >= chosen.cost) {
            train(u, chosen);
        }
    }

    // ────────────────────────────────────────────────────────────────
    //  Military: attack or defend based on aggression, grouping, etc.
    // ────────────────────────────────────────────────────────────────
    private void militaryBehavior(Unit u, Player p, GameState gs,
                                  PhysicalGameState pgs, int nArmy,
                                  boolean beingRushed) {
        // If being rushed and defense_trigger is high, defend
        if (beingRushed && defenseTrigger() > 0.4) {
            Unit threat = findClosestEnemy(u, p, pgs);
            if (threat != null) {
                attack(u, threat);
                return;
            }
        }

        // Group attack: wait until we have enough army
        if (groupAttack() > 0.5 && nArmy < attackThreshold()) {
            // Stay near base
            Unit myBase = findMyBase(p, pgs);
            if (myBase != null) {
                int d = Math.abs(u.getX() - myBase.getX()) + Math.abs(u.getY() - myBase.getY());
                if (d > defendRadius()) {
                    // Move back toward base
                    attack(u, findClosestEnemy(u, p, pgs) != null ?
                           findClosestEnemy(u, p, pgs) : myBase);
                    return;
                }
            }
            // Idle near base, waiting for more troops
            return;
        }

        // Find target based on focus_fire and target_priority
        Unit target = findTarget(u, p, pgs);
        if (target != null) {
            attack(u, target);
        }
    }

    // ────────────────────────────────────────────────────────────────
    //  Target selection: economy_pressure, target_priority, focus_fire
    // ────────────────────────────────────────────────────────────────
    private Unit findTarget(Unit u, Player p, PhysicalGameState pgs) {
        Unit bestTarget = null;
        int bestScore = Integer.MAX_VALUE;

        for (Unit u2 : pgs.getUnits()) {
            if (u2.getPlayer() >= 0 && u2.getPlayer() != p.getID()) {
                int d = Math.abs(u2.getX() - u.getX()) + Math.abs(u2.getY() - u.getY());
                int score = d;

                // economy_pressure: prefer economic targets
                if (economyPressure() > 0.5) {
                    if (u2.getType().isStockpile || u2.getType().canHarvest) {
                        score -= 20; // bonus for economic targets
                    }
                }

                // target_priority: low = target workers, high = target military
                if (targetPriority() < 0.4) {
                    // Prefer workers/bases
                    if (u2.getType().canHarvest || u2.getType().isStockpile) {
                        score -= 15;
                    }
                } else if (targetPriority() > 0.6) {
                    // Prefer military
                    if (u2.getType().canAttack && !u2.getType().canHarvest) {
                        score -= 15;
                    }
                }

                if (score < bestScore) {
                    bestScore = score;
                    bestTarget = u2;
                }
            }
        }

        // Focus fire: if enabled, all units attack the same target (closest enemy)
        if (focusFire() > 0.5 && bestTarget == null) {
            bestTarget = findClosestEnemy(u, p, pgs);
        }

        return bestTarget;
    }

    // ────────────────────────────────────────────────────────────────
    //  Workers: harvest, build, scout, fight
    // ────────────────────────────────────────────────────────────────
    private void workersBehavior(List<Unit> workers, Player p, GameState gs,
                                 PhysicalGameState pgs, int nBases,
                                 int nBarracks, int nArmy, boolean beingRushed) {
        int resourcesUsed = 0;
        List<Unit> freeWorkers = new LinkedList<>(workers);

        if (workers.isEmpty()) return;

        List<Integer> reservedPositions = new LinkedList<>();

        // 1) Build base if we have none
        if (nBases == 0 && !freeWorkers.isEmpty()) {
            if (p.getResources() >= baseType.cost + resourcesUsed) {
                Unit u = freeWorkers.remove(0);
                buildIfNotAlreadyBuilding(u, baseType, u.getX(), u.getY(),
                                          reservedPositions, p, pgs);
                resourcesUsed += baseType.cost;
            }
        }

        // 2) Build barracks based on barracksTiming
        //    Low value = build early, high value = build late (need more resources first)
        int barracksResourceThreshold = (int)(barracksTiming() * 8); // 0-8 resources before building
        if (nBarracks == 0 && !freeWorkers.isEmpty()) {
            if (p.getResources() >= barracksType.cost + resourcesUsed + barracksResourceThreshold) {
                Unit u = freeWorkers.remove(0);
                buildIfNotAlreadyBuilding(u, barracksType, u.getX(), u.getY(),
                                          reservedPositions, p, pgs);
                resourcesUsed += barracksType.cost;
            }
        }

        // 3) Second barracks if secondBarracks weight says so
        if (nBarracks == 1 && !freeWorkers.isEmpty() && secondBarracks() > 0.4
                && nArmy >= 2) {
            if (p.getResources() >= barracksType.cost + resourcesUsed) {
                Unit u = freeWorkers.remove(0);
                buildIfNotAlreadyBuilding(u, barracksType, u.getX(), u.getY(),
                                          reservedPositions, p, pgs);
                resourcesUsed += barracksType.cost;
            }
        }

        // 4) Expand (2nd base) if expand_threshold triggers it
        if (nBases == 1 && nBarracks > 0 && !freeWorkers.isEmpty()
                && expandThreshold() < 0.5 && nArmy >= 2) {
            if (p.getResources() >= baseType.cost + resourcesUsed) {
                Unit u = freeWorkers.remove(0);
                // Build near a distant resource if possible
                Unit res = findDistantResource(p, pgs);
                int bx = (res != null) ? res.getX() + 1 : u.getX();
                int by = (res != null) ? res.getY() + 1 : u.getY();
                buildIfNotAlreadyBuilding(u, baseType, bx, by,
                                          reservedPositions, p, pgs);
                resourcesUsed += baseType.cost;
            }
        }

        // 5) Scout: send a worker to explore
        if (scoutPriority() > 0.5 && !freeWorkers.isEmpty() && gs.getTime() < 100) {
            Unit scout = freeWorkers.remove(freeWorkers.size() - 1);
            Unit enemy = findClosestEnemy(scout, p, pgs);
            if (enemy != null) {
                // Move toward enemy to scout
                move(scout, enemy.getX(), enemy.getY());
            } else {
                // Move toward center of map
                move(scout, pgs.getWidth() / 2, pgs.getHeight() / 2);
            }
        }

        // 6) Counter rush: send workers to fight if being rushed
        if (beingRushed && counterRush() > 0.4) {
            int fightersToSend = Math.max(1, (int)(freeWorkers.size() * counterRush()));
            for (int i = 0; i < fightersToSend && !freeWorkers.isEmpty(); i++) {
                Unit u = freeWorkers.remove(freeWorkers.size() - 1);
                Unit enemy = findClosestEnemy(u, p, pgs);
                if (enemy != null) attack(u, enemy);
            }
        }

        // 7) Permanent worker fighters based on workerFightRatio
        int workerFighters = (int)(freeWorkers.size() * workerFightRatio());
        for (int i = 0; i < workerFighters && !freeWorkers.isEmpty(); i++) {
            Unit u = freeWorkers.remove(freeWorkers.size() - 1);
            Unit enemy = findClosestEnemy(u, p, pgs);
            if (enemy != null) {
                attack(u, enemy);
            }
        }

        // 8) Remaining workers harvest
        for (Unit u : freeWorkers) {
            Unit closestBase = null;
            Unit closestResource = null;
            int closestDistance = 0;
            for (Unit u2 : pgs.getUnits()) {
                if (u2.getType().isResource) {
                    int d = Math.abs(u2.getX() - u.getX()) + Math.abs(u2.getY() - u.getY());
                    if (closestResource == null || d < closestDistance) {
                        closestResource = u2;
                        closestDistance = d;
                    }
                }
            }
            closestDistance = 0;
            for (Unit u2 : pgs.getUnits()) {
                if (u2.getType().isStockpile && u2.getPlayer() == p.getID()) {
                    int d = Math.abs(u2.getX() - u.getX()) + Math.abs(u2.getY() - u.getY());
                    if (closestBase == null || d < closestDistance) {
                        closestBase = u2;
                        closestDistance = d;
                    }
                }
            }
            if (closestResource != null && closestBase != null) {
                AbstractAction aa = getAbstractAction(u);
                if (aa instanceof Harvest) {
                    Harvest h_aa = (Harvest) aa;
                    if (h_aa.getTarget() != closestResource || h_aa.getBase() != closestBase) {
                        harvest(u, closestResource, closestBase);
                    }
                } else {
                    harvest(u, closestResource, closestBase);
                }
            } else {
                // No resources or no base — attack!
                Unit enemy = findClosestEnemy(u, p, pgs);
                if (enemy != null) attack(u, enemy);
            }
        }
    }

    // ────────────────────────────────────────────────────────────────
    //  Helpers
    // ────────────────────────────────────────────────────────────────
    private Unit findClosestEnemy(Unit u, Player p, PhysicalGameState pgs) {
        Unit closest = null;
        int closestDist = Integer.MAX_VALUE;
        for (Unit u2 : pgs.getUnits()) {
            if (u2.getPlayer() >= 0 && u2.getPlayer() != p.getID()) {
                int d = Math.abs(u2.getX() - u.getX()) + Math.abs(u2.getY() - u.getY());
                if (d < closestDist) {
                    closest = u2;
                    closestDist = d;
                }
            }
        }
        return closest;
    }

    private Unit findMyBase(Player p, PhysicalGameState pgs) {
        for (Unit u : pgs.getUnits()) {
            if (u.getType() == baseType && u.getPlayer() == p.getID()) {
                return u;
            }
        }
        return null;
    }

    private Unit findDistantResource(Player p, PhysicalGameState pgs) {
        Unit myBase = findMyBase(p, pgs);
        if (myBase == null) return null;

        Unit farthest = null;
        int farthestDist = 0;
        for (Unit u : pgs.getUnits()) {
            if (u.getType().isResource) {
                int d = Math.abs(u.getX() - myBase.getX()) + Math.abs(u.getY() - myBase.getY());
                if (d > farthestDist) {
                    farthest = u;
                    farthestDist = d;
                }
            }
        }
        return farthest;
    }
}
