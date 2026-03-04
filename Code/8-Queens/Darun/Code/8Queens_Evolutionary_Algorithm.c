#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 8
#define POP 100
#define MAX_GEN 100
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

void print_board(int board[]) {
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            if (board[c] == r)
                printf("Q ");
            else
                printf("- ");
        }
        printf("\n");
    }
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


void write_in_csv(clock_t start, clock_t stop,int gen) {
    FILE *fp = fopen("Queens_Data.csv", "a");
    double time = (((double)(stop - start))/CLOCKS_PER_SEC)*1000;
    if (fp != NULL) {
        fprintf(fp, "%d,%d,%f\n", gen, nfe, time);
        fclose(fp);
    }
}

int main() {
    srand(time(NULL));

    clock_t start = clock();

    int population[POP][N];
    int newPop[POP][N];
    int fit[POP];

    int minFitness;

    for (int i = 0; i < POP; i++) {
        random_board(population[i]);
        fit[i] = fitness(population[i]);
    }

    minFitness = fit[0];
    for (int i = 1; i < POP; i++)
        if (fit[i] < minFitness)
            minFitness = fit[i];


    for (int gen = 0; gen < MAX_GEN; gen++) {

        int bestIndex = 0;
        int bestFit = fit[0];

        for (int i = 1; i < POP; i++) {
            if (fit[i] < bestFit) {
                bestFit = fit[i];
                bestIndex = i;
            }
        }

        if (bestFit == 0) {
            printf("Solution found at generation %d\n\n", gen + 1);
            clock_t stop = clock();    
            write_in_csv(start, stop, gen+1);
            printf("Time Taken : %f ms\n", (((double)(stop - start))/CLOCKS_PER_SEC)*1000);

            print_board(population[bestIndex]);
            return 0;
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

        minFitness = fit[0];
        for (int i = 1; i < POP; i++)
            if (fit[i] < minFitness)
                minFitness = fit[i];

    }
    clock_t stop = clock();    
    write_in_csv(start, stop, MAX_GEN);
    printf("Time Taken : %f ms\n", (((double)(stop - start))/CLOCKS_PER_SEC)*1000);

    printf("No perfect solution found\n");
    return 0;
}
