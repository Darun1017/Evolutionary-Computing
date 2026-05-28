/*
 * EvolvedBatchGameRunner.java
 * Runs MULTIPLE microRTS games in a single JVM invocation.
 * Reads game requests from stdin, outputs results to stdout.
 *
 * Supports two input formats:
 *   1. EVOLVED:w0;w1;...;w19,opponent_type,map,max_cycles
 *      Creates an EvolvedBrainAI (player 0) vs named opponent (player 1)
 *
 *   2. ai1_type,ai2_type,map,max_cycles
 *      Creates two named AIs (same as BatchGameRunner)
 *
 * Output: RESULT:winner,cycles
 * Send "EXIT" to terminate.
 */
package tests;

import ai.core.AI;
import ai.*;
import ai.abstraction.WorkerRush;
import ai.abstraction.WorkerRushPlusPlus;
import ai.abstraction.LightRush;
import ai.abstraction.HeavyRush;
import ai.abstraction.RangedRush;
import ai.abstraction.LightDefense;
import ai.abstraction.HeavyDefense;
import ai.abstraction.RangedDefense;
import ai.abstraction.WorkerDefense;
import ai.abstraction.EconomyRush;
import ai.abstraction.EconomyMilitaryRush;
import ai.abstraction.EvolvedBrainAI;
import ai.abstraction.pathfinding.BFSPathFinding;

import rts.GameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.units.UnitTypeTable;

import java.io.BufferedReader;
import java.io.InputStreamReader;

public class EvolvedBatchGameRunner {

    static UnitTypeTable utt = new UnitTypeTable();

    public static AI createAI(String type) {
        switch (type) {
            case "WorkerRush":
                return new WorkerRush(utt, new BFSPathFinding());
            case "LightRush":
                return new LightRush(utt, new BFSPathFinding());
            case "HeavyRush":
                return new HeavyRush(utt, new BFSPathFinding());
            case "RandomBiased":
                return new RandomBiasedAI();
            case "Random":
                return new RandomAI();
            case "Passive":
                return new PassiveAI();
            case "RangedRush":
                return new RangedRush(utt, new BFSPathFinding());
            case "WorkerRushPlusPlus":
                return new WorkerRushPlusPlus(utt, new BFSPathFinding());
            case "LightDefense":
                return new LightDefense(utt, new BFSPathFinding());
            case "HeavyDefense":
                return new HeavyDefense(utt, new BFSPathFinding());
            case "RangedDefense":
                return new RangedDefense(utt, new BFSPathFinding());
            case "WorkerDefense":
                return new WorkerDefense(utt, new BFSPathFinding());
            case "EconomyRush":
                return new EconomyRush(utt, new BFSPathFinding());
            case "EconomyMilitaryRush":
                return new EconomyMilitaryRush(utt, new BFSPathFinding());
            default:
                return new RandomBiasedAI();
        }
    }

    public static AI createEvolvedAI(String weightsStr) {
        String[] parts = weightsStr.split(";");
        double[] weights = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            weights[i] = Double.parseDouble(parts[i].trim());
        }
        return new EvolvedBrainAI(utt, new BFSPathFinding(), weights);
    }

    public static void main(String[] args) throws Exception {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        System.out.println("READY");
        System.out.flush();

        String line;
        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.equals("EXIT")) break;
            if (line.isEmpty()) continue;

            try {
                AI ai1;
                AI ai2;
                String mapPath;
                int maxCycles;

                if (line.startsWith("EVOLVED:")) {
                    // Format: EVOLVED:w0;w1;...;w19,opponent_type,map,max_cycles
                    String payload = line.substring("EVOLVED:".length());
                    // Split on comma, but the weights contain semicolons not commas
                    // So first part (before first comma) is weights, rest are comma-separated
                    int firstComma = payload.indexOf(',');
                    String weightsStr = payload.substring(0, firstComma);
                    String[] rest = payload.substring(firstComma + 1).split(",");
                    String opponentType = rest[0];
                    mapPath = rest[1];
                    maxCycles = Integer.parseInt(rest[2]);

                    ai1 = createEvolvedAI(weightsStr);
                    ai2 = createAI(opponentType);
                } else {
                    // Format: ai1_type,ai2_type,map,max_cycles
                    String[] parts = line.split(",");
                    ai1 = createAI(parts[0]);
                    ai2 = createAI(parts[1]);
                    mapPath = parts[2];
                    maxCycles = Integer.parseInt(parts[3]);
                }

                // Load map and create game
                PhysicalGameState pgs = PhysicalGameState.load(mapPath, utt);
                GameState gs = new GameState(pgs, utt);

                // Run game
                boolean gameover = false;
                do {
                    PlayerAction pa1 = ai1.getAction(0, gs);
                    PlayerAction pa2 = ai2.getAction(1, gs);
                    gs.issueSafe(pa1);
                    gs.issueSafe(pa2);
                    gameover = gs.cycle();
                } while (!gameover && gs.getTime() < maxCycles);

                ai1.gameOver(gs.winner());
                ai2.gameOver(gs.winner());

                System.out.println("RESULT:" + gs.winner() + "," + gs.getTime());
                System.out.flush();

            } catch (Exception e) {
                System.out.println("ERROR:" + e.getMessage());
                System.out.flush();
            }
        }
    }
}
