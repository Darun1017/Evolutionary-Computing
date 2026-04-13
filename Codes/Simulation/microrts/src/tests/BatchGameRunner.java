/*
 * BatchGameRunner.java
 * Runs MULTIPLE microRTS games in a single JVM invocation.
 * Reads game requests from stdin, outputs results to stdout.
 * 
 * Input format (one per line):  ai1_type,ai2_type,map,max_cycles
 * Output format (one per line): RESULT:winner,cycles
 * Send "EXIT" to terminate.
 * 
 * Supports ALL 14 bot strategies for TD-CCEA evolution.
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
import ai.abstraction.pathfinding.BFSPathFinding;

import rts.GameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.units.UnitTypeTable;

import java.io.BufferedReader;
import java.io.InputStreamReader;

public class BatchGameRunner {
    
    static UnitTypeTable utt = new UnitTypeTable();
    
    public static AI createAI(String type) {
        switch (type) {
            // --- Original 6 strategies ---
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
            // --- 8 NEW strategies (total = 14) ---
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
                // Parse: ai1_type,ai2_type,map,max_cycles
                String[] parts = line.split(",");
                String ai1Type = parts[0];
                String ai2Type = parts[1];
                String mapPath = parts[2];
                int maxCycles = Integer.parseInt(parts[3]);
                
                // Load map and create game
                PhysicalGameState pgs = PhysicalGameState.load(mapPath, utt);
                GameState gs = new GameState(pgs, utt);
                
                AI ai1 = createAI(ai1Type);
                AI ai2 = createAI(ai2Type);
                
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
