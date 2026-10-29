// Replaces rand() with the generator the Rust port uses (src/rng.rs), one stream for the
// physics and one for the synthesizer, so runs can be compared step by step.
#pragma once
#include <cstdlib>
int es_rand_physics();
int es_rand_synth();
#ifdef ES_RAND_SYNTH
#define rand() es_rand_synth()
#else
#define rand() es_rand_physics()
#endif
