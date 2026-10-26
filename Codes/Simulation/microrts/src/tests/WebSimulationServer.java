package tests;

import ai.core.AI;
import ai.abstraction.*;
import ai.abstraction.pathfinding.BFSPathFinding;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import rts.GameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.units.Unit;
import rts.units.UnitTypeTable;

import java.io.*;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * WebSimulationServer
 * 
 * Provides HTTP API and JSON export capabilities for microRTS evolutionary simulations.
 * Supports:
 *   1. vs Bot: 20D Evolved vector vs one of 7 canonical bots
 *   2. vs Player: 20D Evolved vector vs 20D Evolved vector
 * 
 * - Live Server: POST /api/simulate -> returns frame-by-frame match trace
 * - CLI Exporter: java tests.WebSimulationServer export EVOLVED:... Bot map out.json
 */
public class WebSimulationServer {
    private static final int DEFAULT_PORT = 8080;
    private static UnitTypeTable utt = new UnitTypeTable();

    public static void main(String[] args) throws Exception {
        if (args.length > 0 && args[0].equalsIgnoreCase("export")) {
            runExportCli(args);
        } else {
            int port = args.length > 0 ? Integer.parseInt(args[0]) : DEFAULT_PORT;
            startHttpServer(port);
        }
    }

    // =========================================================================
    // HTTP SERVER
    // =========================================================================
    public static void startHttpServer(int port) throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext("/api/simulate", new SimulateHandler());
        server.setExecutor(java.util.concurrent.Executors.newCachedThreadPool());
        server.start();
        System.out.println("==================================================");
        System.out.println(" microRTS Web Simulation Server Running!");
        System.out.println(" Listening on: http://localhost:" + port + "/api/simulate");
        System.out.println(" Supports: vs Bot (7 bots) & vs Player (20D vs 20D)");
        System.out.println(" Press Ctrl+C to stop.");
        System.out.println("==================================================");
    }

    static class SimulateHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            exchange.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
            exchange.getResponseHeaders().set("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
            exchange.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type");

            if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(204, -1);
                return;
            }

            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendString(exchange, 405, "{\"error\": \"Only POST is supported\"}");
                return;
            }

            try {
                InputStream is = exchange.getRequestBody();
                String body = new String(readAllBytes(is), StandardCharsets.UTF_8);

                String mode = parseJsonString(body, "mode", "vs-bot");
                String mapPath = parseJsonString(body, "map", "maps/8x8/basesWorkers8x8.xml");
                int maxCycles = Math.min(parseJsonInt(body, "maxCycles", 3000), 5000);
                int sampleInterval = Math.max(parseJsonInt(body, "sampleInterval", 2), 1);

                String jsonResult;
                if ("vs-player".equalsIgnoreCase(mode) || body.contains("player2_weights") || body.contains("p2_weights")) {
                    double[] p1Weights = parseWeightsArray(body, "player1_weights", "weights");
                    double[] p2Weights = parseWeightsArray(body, "player2_weights", "p2_weights");
                    jsonResult = runSimulationVsPlayer(p1Weights, p2Weights, mapPath, maxCycles, sampleInterval);
                } else {
                    double[] weights = parseWeightsArray(body, "weights", "player1_weights");
                    String opponent = parseJsonString(body, "opponent", "WorkerRush");
                    jsonResult = runSimulationVsBot(weights, opponent, mapPath, maxCycles, sampleInterval);
                }

                sendString(exchange, 200, jsonResult);
            } catch (Exception e) {
                e.printStackTrace();
                String error = "{\"error\": \"" + e.getMessage().replace("\"", "\\\"") + "\"}";
                sendString(exchange, 500, error);
            }
        }
    }

    private static void sendString(HttpExchange exchange, int code, String text) throws IOException {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        exchange.sendResponseHeaders(code, bytes.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(bytes);
        }
    }

    // =========================================================================
    // SIMULATION ENGINE: VS BOT & VS PLAYER
    // =========================================================================
    public static String runSimulationVsBot(double[] weights, String opponentName, String mapPath, int maxCycles, int sampleInterval) throws Exception {
        PhysicalGameState pgs = PhysicalGameState.load(mapPath, utt);
        GameState gs = new GameState(pgs, utt);
        AI ai1 = new EvolvedBrainAI(utt, new BFSPathFinding(), weights);
        AI ai2 = createBot(opponentName, utt);

        return executeMatch(gs, ai1, ai2, "Player 1 (Evolved)", opponentName, mapPath, maxCycles, sampleInterval);
    }

    public static String runSimulationVsPlayer(double[] p1Weights, double[] p2Weights, String mapPath, int maxCycles, int sampleInterval) throws Exception {
        PhysicalGameState pgs = PhysicalGameState.load(mapPath, utt);
        GameState gs = new GameState(pgs, utt);
        AI ai1 = new EvolvedBrainAI(utt, new BFSPathFinding(), p1Weights);
        AI ai2 = new EvolvedBrainAI(utt, new BFSPathFinding(), p2Weights);

        return executeMatch(gs, ai1, ai2, "Player 1 (Evolved)", "Player 2 (Evolved)", mapPath, maxCycles, sampleInterval);
    }

    private static String executeMatch(GameState gs, AI ai1, AI ai2, String p1Name, String p2Name, String mapPath, int maxCycles, int sampleInterval) throws Exception {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int mapW = pgs.getWidth();
        int mapH = pgs.getHeight();

        StringBuilder json = new StringBuilder(1024 * 512);
        json.append("{\n");
        json.append("  \"map\": {\"width\": ").append(mapW).append(", \"height\": ").append(mapH).append(", \"walls\": [");

        boolean firstWall = true;
        for (int y = 0; y < mapH; y++) {
            for (int x = 0; x < mapW; x++) {
                if (pgs.getTerrain(x, y) == PhysicalGameState.TERRAIN_WALL) {
                    if (!firstWall) json.append(",");
                    json.append("[").append(x).append(",").append(y).append("]");
                    firstWall = false;
                }
            }
        }
        json.append("]},\n");
        json.append("  \"p1_name\": \"").append(p1Name).append("\",\n");
        json.append("  \"p2_name\": \"").append(p2Name).append("\",\n");
        json.append("  \"frames\": [\n");

        boolean gameover = false;
        boolean firstFrame = true;
        do {
            PlayerAction pa1 = ai1.getAction(0, gs);
            PlayerAction pa2 = ai2.getAction(1, gs);
            gs.issueSafe(pa1);
            gs.issueSafe(pa2);

            if (gs.getTime() % sampleInterval == 0 || gameover) {
                if (!firstFrame) json.append(",\n");
                appendFrameJson(json, gs);
                firstFrame = false;
            }
            gameover = gs.cycle();
        } while (!gameover && gs.getTime() < maxCycles);

        if (gs.getTime() % sampleInterval != 0) {
            if (!firstFrame) json.append(",\n");
            appendFrameJson(json, gs);
        }

        ai1.gameOver(gs.winner());
        ai2.gameOver(gs.winner());

        json.append("\n  ],\n");
        json.append("  \"winner\": ").append(gs.winner()).append(",\n");
        json.append("  \"cycles\": ").append(gs.getTime()).append("\n");
        json.append("}");
        return json.toString();
    }

    private static void appendFrameJson(StringBuilder sb, GameState gs) {
        PhysicalGameState currentPgs = gs.getPhysicalGameState();
        sb.append("    {");
        sb.append("\"t\":").append(gs.getTime()).append(",");
        sb.append("\"p0_res\":").append(gs.getPlayer(0) != null ? gs.getPlayer(0).getResources() : 0).append(",");
        sb.append("\"p1_res\":").append(gs.getPlayer(1) != null ? gs.getPlayer(1).getResources() : 0).append(",");
        sb.append("\"units\":[");
        boolean firstU = true;
        for (Unit u : currentPgs.getUnits()) {
            if (!firstU) sb.append(",");
            sb.append("{");
            sb.append("\"id\":").append(u.getID()).append(",");
            sb.append("\"p\":").append(u.getPlayer()).append(",");
            sb.append("\"type\":\"").append(u.getType().name).append("\",");
            sb.append("\"x\":").append(u.getX()).append(",");
            sb.append("\"y\":").append(u.getY()).append(",");
            sb.append("\"hp\":").append(u.getHitPoints()).append(",");
            sb.append("\"maxHp\":").append(u.getType().hp).append(",");
            sb.append("\"res\":").append(u.getResources());
            sb.append("}");
            firstU = false;
        }
        sb.append("]}");
    }

    // =========================================================================
    // BOT FACTORY (THE 7 CANONICAL BOTS)
    // =========================================================================
    public static AI createBot(String type, UnitTypeTable utt) {
        switch (type) {
            case "WorkerRush":           return new WorkerRush(utt, new BFSPathFinding());
            case "LightRush":            return new LightRush(utt, new BFSPathFinding());
            case "HeavyRush":            return new HeavyRush(utt, new BFSPathFinding());
            case "RangedRush":           return new RangedRush(utt, new BFSPathFinding());
            case "EconomyRush":          return new EconomyRush(utt, new BFSPathFinding());
            case "EconomyMilitaryRush":  return new EconomyMilitaryRush(utt, new BFSPathFinding());
            case "HeavyDefense":         return new HeavyDefense(utt, new BFSPathFinding());
            default:                     return new WorkerRush(utt, new BFSPathFinding());
        }
    }

    // =========================================================================
    // PARSING HELPERS
    // =========================================================================
    private static double[] parseWeightsArray(String body, String keyPrimary, String keyFallback) {
        String targetKey = "\"" + keyPrimary + "\":";
        int idx = body.indexOf(targetKey);
        if (idx == -1 && keyFallback != null) {
            targetKey = "\"" + keyFallback + "\":";
            idx = body.indexOf(targetKey);
        }
        if (idx == -1) {
            // Default 20D fallback vector
            return new double[]{0.24,0.52,0.52,0.97,0.14,0.29,0.62,0.17,0.71,0.63,0.13,0.06,0.99,0.12,0.00,0.61,0.99,0.46,0.57,1.00};
        }

        int start = body.indexOf("[", idx);
        int end = body.indexOf("]", start);
        if (start == -1 || end == -1) {
            return new double[20];
        }

        String arrayStr = body.substring(start + 1, end).trim();
        if (arrayStr.isEmpty()) return new double[20];

        String[] tokens = arrayStr.split(",");
        double[] weights = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            try {
                weights[i] = Double.parseDouble(tokens[i].trim());
            } catch (Exception e) {
                weights[i] = 0.5;
            }
        }
        return weights;
    }

    private static String parseJsonString(String json, String key, String defaultVal) {
        String pattern = "\"" + key + "\":";
        int idx = json.indexOf(pattern);
        if (idx == -1) return defaultVal;
        int startQuote = json.indexOf("\"", idx + pattern.length());
        if (startQuote == -1) return defaultVal;
        int endQuote = json.indexOf("\"", startQuote + 1);
        if (endQuote == -1) return defaultVal;
        return json.substring(startQuote + 1, endQuote);
    }

    private static int parseJsonInt(String json, String key, int defaultVal) {
        String pattern = "\"" + key + "\":";
        int idx = json.indexOf(pattern);
        if (idx == -1) return defaultVal;
        int start = idx + pattern.length();
        int end = start;
        while (end < json.length() && (Character.isDigit(json.charAt(end)) || json.charAt(end) == ' ')) {
            end++;
        }
        try {
            return Integer.parseInt(json.substring(start, end).trim());
        } catch (Exception e) {
            return defaultVal;
        }
    }

    private static byte[] readAllBytes(InputStream is) throws IOException {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        int nRead;
        byte[] data = new byte[4096];
        while ((nRead = is.read(data, 0, data.length)) != -1) {
            buffer.write(data, 0, nRead);
        }
        return buffer.toByteArray();
    }

    // =========================================================================
    // CLI EXPORT FUNCTIONALITY
    // =========================================================================
    private static void runExportCli(String[] args) throws Exception {
        // Usage: java tests.WebSimulationServer export EVOLVED:0.24;... WorkerRush map.xml out.json
        if (args.length < 5) {
            System.out.println("CLI Usage: export <p1_arg> <p2_arg> <mapPath> <outJsonPath>");
            System.out.println("Example: export EVOLVED:0.24;0.52;... WorkerRush maps/8x8/basesWorkers8x8.xml replay.json");
            return;
        }

        String p1Arg = args[1];
        String p2Arg = args[2];
        String mapPath = args[3];
        String outPath = args[4];

        PhysicalGameState pgs = PhysicalGameState.load(mapPath, utt);
        GameState gs = new GameState(pgs, utt);

        AI ai1 = parseAiArg(p1Arg, utt);
        AI ai2 = parseAiArg(p2Arg, utt);

        String json = executeMatch(gs, ai1, ai2, p1Arg.startsWith("EVOLVED") ? "Evolved Brain P1" : p1Arg,
                                   p2Arg.startsWith("EVOLVED") ? "Evolved Brain P2" : p2Arg,
                                   mapPath, 4000, 2);

        File outFile = new File(outPath);
        if (outFile.getParentFile() != null) outFile.getParentFile().mkdirs();
        try (FileWriter fw = new FileWriter(outFile)) {
            fw.write(json);
        }
        System.out.println("Replay successfully exported to: " + outPath);
    }

    private static AI parseAiArg(String arg, UnitTypeTable utt) {
        if (arg.startsWith("EVOLVED:")) {
            String[] tokens = arg.substring("EVOLVED:".length()).split(";");
            double[] weights = new double[tokens.length];
            for (int i = 0; i < tokens.length; i++) {
                weights[i] = Double.parseDouble(tokens[i].trim());
            }
            return new EvolvedBrainAI(utt, new BFSPathFinding(), weights);
        } else {
            return createBot(arg, utt);
        }
    }
}
