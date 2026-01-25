#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

// ---------------- Configuration ----------------
#define N 16             // Number of nodes in the graph
#define D 3              // Regular degree
#define MAX_ADAPT 2      // Maximum edge rewiring per update
#define LUT_SIZE 256     // Lookup table size for 8-bit tanh
#define T 10             // Number of iterations

// ---------------- Lookup Table for 8-bit tanh ----------------
uint8_t tanh_lut[LUT_SIZE];

// ---------------- Initialize tanh LUT ----------------
void init_tanh_lut() {
    for (int i = 0; i < LUT_SIZE; i++) {
        // Map 0-255 -> -4.0 to 4.0
        double x = 8.0 * i / (double)LUT_SIZE - 4.0;
        double y = tanh(x);
        tanh_lut[i] = (uint8_t)((y + 1.0) * 127.5); // Scale -1..1 to 0..255
    }
}

// ---------------- Sparse matrix-vector multiplication ----------------
// A is d-regular adjacency matrix stored as neighbors
void matvec(uint8_t K_in[N], uint8_t A[D][N], uint8_t K_out[N]) {
    for (int i = 0; i < N; i++) {
        int sum = 0;
        for (int j = 0; j < D; j++) {
            int neighbor = A[j][i];
            sum += K_in[neighbor];
        }
        K_out[i] = sum / D; // Simple average
    }
}

// ---------------- Entropy injection (simple PRNG) ----------------
void inject_entropy(uint8_t K[N]) {
    for (int i = 0; i < N; i++) {
        K[i] ^= rand() & 0xFF;
    }
}

// ---------------- Nonlinear activation ----------------
void nonlinear_activation(uint8_t K[N]) {
    for (int i = 0; i < N; i++) {
        K[i] = tanh_lut[K[i]];
    }
}

// ---------------- Adaptive edge update (dummy example) ----------------
void adapt_edges(uint8_t A[D][N]) {
    // Randomly rewire at most MAX_ADAPT edges
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < MAX_ADAPT; k++) {
            int j = rand() % D;
            int new_neighbor = rand() % N;
            A[j][i] = new_neighbor;
        }
    }
}

// ---------------- Main RNSG update ----------------
int main() {
    uint8_t K[N] = {0};      // Initial state
    uint8_t K_next[N];
    uint8_t A[D][N];          // Graph adjacency (d-regular)
    
    // Initialize adjacency matrix
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            A[j][i] = (i + j + 1) % N;

    // Initialize tanh LUT
    init_tanh_lut();

    // Iterative updates
    for (int t = 0; t < T; t++) {
        matvec(K, A, K_next);
        inject_entropy(K_next);
        nonlinear_activation(K_next);
        adapt_edges(A);

        // Copy K_next -> K for next iteration
        for (int i = 0; i < N; i++)
            K[i] = K_next[i];

        // Optional: print state
        printf("Iteration %d: ", t);
        for (int i = 0; i < N; i++)
            printf("%d ", K[i]);
        printf("\n");
    }

    return 0;
}
