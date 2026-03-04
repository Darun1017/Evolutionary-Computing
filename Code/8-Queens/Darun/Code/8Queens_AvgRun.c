#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 8
#define POP 100
#define MAX_GEN 100
#define MUT_RATE 0.05     // reduced mutation
#define N_RUNS 10

int randInt(int max) {
    return rand() % max;
}

double rand01() {
    return (double)rand() / RAND_MAX;
}

int fitness(int board[]) {
    int conflicts = 0;

    for (int c1 = 0; c1 < N; c1++) {
        for (int c2 = c1 + 1; c2 < N; c2++) {
            int r1 = board[c1];
            int r2 = board[c2];

            if (r1 == r2)
                conflicts++;

            if (abs(r1 - r2) == abs(c1 - c2))
                conflicts++;
        }
    }
    return conflicts;
}

void random_board(int board[]) {
    for (int i = 0; i < N; i++)
        board[i] = randInt(N);
}

void crossover(int parent1[], int parent2[], int child[]) {
    int cut = randInt(N - 1) + 1;

    for (int i = 0; i < cut; i++)
        child[i] = parent1[i];
    for (int i = cut; i < N; i++)
        child[i] = parent2[i];
}

void mutate(int board[]) {
    for (int i = 0; i < N; i++) {
        if (rand01() < MUT_RATE)
            board[i] = randInt(N);
    }
}

/* -------- TOURNAMENT SELECTION (KEY FIX) -------- */
int tournament(int fit[]) {
    int a = randInt(POP);
    int b = randInt(POP);
    return (fit[a] < fit[b]) ? a : b;
}
/* ------------------------------------------------ */

void write_csv(int minFit[N_RUNS][MAX_GEN]) {
    FILE *fp = fopen("results_avg10.csv", "w");
    if (!fp) {
        printf("Error opening CSV file\n");
        return;
    }

    fprintf(fp, "Generation");
    for (int r = 0; r < N_RUNS; r++)
        fprintf(fp, ",Run%d", r + 1);
    fprintf(fp, ",Average\n");

    for (int g = 0; g < MAX_GEN; g++) {
        fprintf(fp, "%d", g + 1);

        int sum = 0;
        for (int r = 0; r < N_RUNS; r++) {
            fprintf(fp, ",%d", minFit[r][g]);
            sum += minFit[r][g];
        }

        fprintf(fp, ",%.2f\n", (double)sum / N_RUNS);
    }

    fclose(fp);
}

int main() {
    srand(time(NULL));

    int minFit[N_RUNS][MAX_GEN] = {0};

    for (int run = 0; run < N_RUNS; run++) {

        int population[POP][N];
        int newPop[POP][N];
        int fit[POP];

        for (int i = 0; i < POP; i++) {
            random_board(population[i]);
            fit[i] = fitness(population[i]);
        }

        for (int gen = 0; gen < MAX_GEN; gen++) {

            int bestFit = fit[0];
            int bestIndex = 0;

            for (int i = 1; i < POP; i++) {
                if (fit[i] < bestFit) {
                    bestFit = fit[i];
                    bestIndex = i;
                }
            }

            minFit[run][gen] = bestFit;

            if (bestFit == 0) {
                for (int g2 = gen + 1; g2 < MAX_GEN; g2++)
                    minFit[run][g2] = 0;
                break;
            }

            /* Elitism */
            for (int j = 0; j < N; j++)
                newPop[0][j] = population[bestIndex][j];

            /* -------- FIXED SELECTION -------- */
            for (int i = 1; i < POP; i++) {
                int p1 = tournament(fit);
                int p2 = tournament(fit);
                crossover(population[p1], population[p2], newPop[i]);
                mutate(newPop[i]);
            }
            /* -------------------------------- */

            for (int i = 0; i < POP; i++) {
                for (int j = 0; j < N; j++)
                    population[i][j] = newPop[i][j];
                fit[i] = fitness(population[i]);
            }
        }

        printf("Run %d completed\n", run + 1);
    }

    write_csv(minFit);
    printf("CSV file written successfully.\n");

    return 0;
}
