/**
 * example.c — minimal C usage example for eml_math.
 *
 * Compile:
 *   cargo build --release -p eml_c_api
 *   gcc example.c -I. -L../target/release -leml_math -lm -o example
 *   ./example
 */

#include <stdio.h>
#include <math.h>
#include "eml_math.h"

int main(void) {
    /* eml(1, 1) = exp(1) - ln(1) = e */
    printf("eml(1, 1) = %.10f  (expected: e = %.10f)\n",
           eml_tension(1.0, 1.0), exp(1.0));

    /* Mirror-Pulse: (1, 1) -> (1, e) */
    double nx, ny;
    eml_mirror_pulse(1.0, 1.0, &nx, &ny);
    printf("mirror_pulse(1, 1) -> x=%.6f  y=%.6f\n", nx, ny);

    /* Boost: rapidity = 0.5, c = 1 */
    double bx, by;
    eml_boost(1.0, exp(1.0), 0.5, 1.0, &bx, &by);
    double delta_before = eml_minkowski_delta(1.0, exp(1.0), 1, 1.0);
    double delta_after  = eml_minkowski_delta(bx,  by,       1, 1.0);
    printf("boost invariance: delta_before=%.10f  delta_after=%.10f\n",
           delta_before, delta_after);

    /* Causal type */
    printf("causal_type(1, e, 1, 1e-9) = %d  (expected: 0 = lightlike)\n",
           eml_causal_type(1.0, exp(1.0), 1.0, 1e-9));

    /* Octonion: e1 * e2 = e4 */
    double e1[8] = {0,1,0,0,0,0,0,0};
    double e2[8] = {0,0,1,0,0,0,0,0};
    double result[8];
    eml_octonion_mul(e1, e2, result);
    printf("e1 * e2 = e4: result[4] = %.1f  (expected: 1.0)\n", result[4]);

    return 0;
}
