/*
 * HeadlessGameRunner.java
 * Runs a single microRTS game between two AIs WITHOUT GUI.
 * Outputs the result to stdout for Python EA to read.
 * 
 * Usage: java HeadlessGameRunner <ai1_type> <ai2_type> <map_path> <max_cycles>
 * 
 * AI types: WorkerRush, LightRush, HeavyRush, RandomBiased, Random, Passive
 * 
 * Output format: WINNER:<0|1|-1>,CYCLES:<n>,SCORE0:<s>,SCORE1:<s>
 */
package tests;

import ai.core.AI;
import ai.*;
import ai.abstraction.WorkerRush;
import ai.abstraction.LightRush;
import ai.abstraction.HeavyRush;
import ai.abstraction.pathfinding.BFSPathFinding;

import rts.GameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.units.UnitTypeTable;

public class HeadlessGameRunner {
    
    public static AI createAI(String type, UnitTypeTable utt) {
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
            default:
                return new RandomBiasedAI();
        }
    }
    
    public static void main(String[] args) throws Exception {
        // Parse arguments
        String ai1Type = args.length > 0 ? args[0] : "WorkerRush";
        String ai2Type = args.length > 1 ? args[1] : "RandomBiased";
        String mapPath = args.length > 2 ? args[2] : "maps/8x8/basesWorkers8x8A.xml";
        int maxCycles = args.length > 3 ? Integer.parseInt(args[3]) : 3000;
        
        // Setup game
        UnitTypeTable utt = new UnitTypeTable();
        PhysicalGameState pgs = PhysicalGameState.load(mapPath, utt);
        GameState gs = new GameState(pgs, utt);
        
        AI ai1 = createAI(ai1Type, utt);
        AI ai2 = createAI(ai2Type, utt);
        
        // Run game (headless, no GUI)
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
        
        // Output result for Python to parse
        int winner = gs.winner();
        int cycles = gs.getTime();
        System.out.println("RESULT:" + winner + "," + cycles + "," + ai1Type + "," + ai2Type);
    }
}
