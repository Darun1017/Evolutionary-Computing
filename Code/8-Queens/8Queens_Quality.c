#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 8
#define POP 100
#define MAX_GEN 50
#define MUT_RATE 0.05

int nfe = 0;

int randInt(int max) {
    return rand() % max;
}

double rand01() {
    return (double)rand() / RAND_MAX;
}

int fitness(int board[]) {
    int conflicts = 0;
    nfe++;

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

int tournament(int fit[]) {
    int a = randInt(POP);
    int b = randInt(POP);
    return (fit[a] < fit[b]) ? a : b;
}

void write_in_csv(int gen, int bestFit, double avgFit) {

    FILE *fp = fopen("Queens_Quality.csv", "a");

    if (fp != NULL) {
        fprintf(fp, "%d,%d,%.3f\n", gen, bestFit, avgFit);
        fclose(fp);
    }
}

int main() {
    srand(time(NULL));

    clock_t start = clock();

    int population[POP][N];
    int newPop[POP][N];
    int fit[POP];

    for (int i = 0; i < POP; i++) {
        random_board(population[i]);
        fit[i] = fitness(population[i]);
    }

    int bestFit = fit[0];

    for (int gen = 0; gen < MAX_GEN; gen++) {

        bestFit = fit[0];
        int bestIndex = 0;

        for (int i = 1; i < POP; i++) {
            if (fit[i] < bestFit) {
                bestFit = fit[i];
                bestIndex = i;
            }
        }

        for (int j = 0; j < N; j++)
            newPop[0][j] = population[bestIndex][j];

        for (int i = 1; i < POP; i++) {
            int p1 = tournament(fit);
            int p2 = tournament(fit);
            crossover(population[p1], population[p2], newPop[i]);
            mutate(newPop[i]);
        }

        for (int i = 0; i < POP; i++) {
            for (int j = 0; j < N; j++)
                population[i][j] = newPop[i][j];
            fit[i] = fitness(population[i]);
        }
    }

    double avgFit = 0.0;
    for (int i = 0; i < POP; i++)
        avgFit += fit[i];
    avgFit /= POP;

    clock_t stop = clock();

    write_in_csv(MAX_GEN, bestFit, avgFit);

    printf("Stopped at generation %d\n", MAX_GEN);
    printf("Best fitness = %d\n", bestFit);
    printf("Average fitness = %.2f\n", avgFit);
    printf("Time taken = %.3f ms\n",
           (((double)(stop - start)) / CLOCKS_PER_SEC) * 1000);

    return 0;
}
