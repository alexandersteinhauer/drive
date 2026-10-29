// The controls of the reference run, per step (shared by reference.cpp and tests/reference.rs
// in spirit; keep both in sync).
#pragma once

static const int SCENARIO_STEPS = 30000;

struct Controls {
    bool dyno;
    double rpm;
    double throttle;
};

static Controls scenario(int step) {
    if (step < 5000) return {true, 900.0, 0.0};
    if (step < 15000) return {true, 900.0 + (step - 5000) / 10000.0 * 3100.0, 1.0};
    if (step < 20000) return {true, 4000.0 - (step - 15000) / 5000.0 * 2000.0, 0.3};
    if (step < 25000) return {false, 0.0, 0.6};
    return {false, 0.0, 0.0};
}
