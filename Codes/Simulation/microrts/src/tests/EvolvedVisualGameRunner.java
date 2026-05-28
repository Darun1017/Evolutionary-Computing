/*
 * EvolvedVisualGameRunner.java
 * Runs a single microRTS game between an EvolvedBrainAI and a named opponent
 * WITH a GUI window so you can watch the game in real-time.
 *
 * Usage: java -cp "lib/*;bin" tests.EvolvedVisualGameRunner "EVOLVED:w0;w1;...;w19" OpponentType mapPath
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
import gui.PhysicalGameStatePanel;

import javax.swing.JFrame;
import rts.GameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.units.UnitTypeTable;

public class EvolvedVisualGameRunner {

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

    public static AI createEvolvedAI(String arg, UnitTypeTable utt) {
        // arg format: "EVOLVED:w0;w1;...;w19"
        String weightsStr = arg.substring("EVOLVED:".length());
        String[] parts = weightsStr.split(";");
        double[] weights = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            weights[i] = Double.parseDouble(parts[i].trim());
        }
        return new EvolvedBrainAI(utt, new BFSPathFinding(), weights);
    }

    public static void main(String[] args) throws Exception {
        String ai1Arg = args.length > 0 ? args[0] : "WorkerRush";
        String ai2Type = args.length > 1 ? args[1] : "RandomBiased";
        String mapPath = args.length > 2 ? args[2] : "maps/4x4/basesWorkers4x4.xml";

        UnitTypeTable utt = new UnitTypeTable();
        PhysicalGameState pgs = PhysicalGameState.load(mapPath, utt);
        GameState gs = new GameState(pgs, utt);

        AI ai1;
        if (ai1Arg.startsWith("EVOLVED:")) {
            ai1 = createEvolvedAI(ai1Arg, utt);
        } else {
            ai1 = createAI(ai1Arg, utt);
        }
        AI ai2 = createAI(ai2Type, utt);

        JFrame w = PhysicalGameStatePanel.newVisualizer(gs, 640, 640, false,
                PhysicalGameStatePanel.COLORSCHEME_BLACK);

        int MAXCYCLES = 10000;
        int PERIOD = 20;
        boolean gameover = false;

        long nextTimeToUpdate = System.currentTimeMillis() + PERIOD;
        do {
            if (System.currentTimeMillis() >= nextTimeToUpdate) {
                PlayerAction pa1 = ai1.getAction(0, gs);
                PlayerAction pa2 = ai2.getAction(1, gs);
                gs.issueSafe(pa1);
                gs.issueSafe(pa2);

                gameover = gs.cycle();
                w.repaint();
                nextTimeToUpdate += PERIOD;
            } else {
                try {
                    Thread.sleep(1);
                } catch (Exception e) {
                }
            }
        } while (!gameover && gs.getTime() < MAXCYCLES);

        ai1.gameOver(gs.winner());
        ai2.gameOver(gs.winner());

        System.out.println("Game Over. Winner: " + gs.winner());
        Thread.sleep(3000);
        w.dispose();
    }
}
