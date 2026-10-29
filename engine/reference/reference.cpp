// Reference runs of the original engine-sim (C++), to check the Rust port against.
//   reference <script.mr> <spec.json> <trace.txt>
// Compiles the script, writes the engine it builds as the port's EngineSpec JSON, then runs
// the scenario in scenario.h and writes one line per simulation step.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// Everything in engine-sim's classes is read directly.
#define protected public
#define private public
#include "scripting/include/compiler.h"
#include "include/engine.h"
#include "include/simulator.h"
#include "include/piston_engine_simulator.h"
#include "include/direct_throttle_linkage.h"
#include "include/standard_valvetrain.h"
#include "include/units.h"
#include "dependencies/submodules/simple-2d-constraint-solver/include/optimized_nsv_rigid_body_system.h"
#undef protected
#undef private

#include "scenario.h"

static unsigned g_physics = 1, g_synth = 1;
static int next(unsigned &s) {
    s = (s * 1103515245u + 12345u) & 0x7fffffffu;
    return (int)s;
}
int es_rand_physics() { return next(g_physics); }
int es_rand_synth() { return next(g_synth); }

static std::string num(double v) {
    char buf[64];
    snprintf(buf, sizeof buf, "%.17g", v);
    return buf;
}

static std::string fn(const Function *f) {
    std::string s = "{\"filterRadius\":" + num(f->m_filterRadius) + ",\"samples\":[";
    for (int i = 0; i < f->m_size; ++i) {
        if (i) s += ",";
        s += "[" + num(f->m_x[i]) + "," + num(f->m_y[i]) + "]";
    }
    return s + "]}";
}

static std::string cam(const Camshaft *c) {
    std::string s = "{\"advance\":" + num(c->m_advance) + ",\"profile\":" + fn(c->m_lobeProfile) + ",\"lobes\":[";
    for (int i = 0; i < c->m_lobes; ++i) {
        if (i) s += ",";
        s += num(c->m_lobeAngles[i] * 2);
    }
    return s + "]}";
}

static std::string spec(Engine *e) {
    std::ostringstream o;
    auto *t = dynamic_cast<DirectThrottleLinkage *>(e->m_throttle);
    o << "{\"simulationFrequency\":" << num(e->m_initialSimulationFrequency)
      << ",\"fluidSteps\":8"
      << ",\"starterTorque\":" << num(e->m_starterTorque)
      << ",\"starterSpeed\":" << num(e->m_starterSpeed)
      << ",\"throttleGamma\":" << num(t->m_gamma);
    const Fuel &f = e->m_fuel;
    o << ",\"fuel\":{\"molecularMass\":" << num(f.m_molecularMass)
      << ",\"energyDensity\":" << num(f.m_energyDensity)
      << ",\"molecularAfr\":" << num(f.m_molecularAfr)
      << ",\"maxBurningEfficiency\":" << num(f.m_maxBurningEfficiency)
      << ",\"burningEfficiencyRandomness\":" << num(f.m_burningEfficiencyRandomness)
      << ",\"lowEfficiencyAttenuation\":" << num(f.m_lowEfficiencyAttenuation)
      << ",\"maxTurbulenceEffect\":" << num(f.m_maxTurbulenceEffect)
      << ",\"maxDilutionEffect\":" << num(f.m_maxDilutionEffect)
      << ",\"turbulenceToFlameSpeedRatio\":" << fn(f.m_turbulenceToFlameSpeedRatio) << "}";
    const Crankshaft *c = e->getCrankshaft(0);
    o << ",\"crankshaft\":{\"throw\":" << num(c->m_throw) << ",\"mass\":" << num(c->m_m)
      << ",\"flywheelMass\":" << num(c->m_flywheelMass) << ",\"momentOfInertia\":" << num(c->m_I)
      << ",\"frictionTorque\":" << num(c->m_frictionTorque) << ",\"x\":" << num(c->m_p_x)
      << ",\"y\":" << num(c->m_p_y) << ",\"tdc\":" << num(c->m_tdc) << ",\"journals\":[";
    for (int i = 0; i < c->m_rodJournalCount; ++i) o << (i ? "," : "") << num(c->m_rodJournalAngles[i]);
    o << "]}";

    o << ",\"banks\":[";
    int cyl = 0;
    for (int b = 0; b < e->getCylinderBankCount(); ++b) {
        CylinderBank *bank = e->getCylinderBank(b);
        CylinderHead *h = e->getHead(b);
        auto *vt = dynamic_cast<StandardValvetrain *>(h->m_valvetrain);
        o << (b ? "," : "") << "{\"angle\":" << num(bank->m_angle) << ",\"bore\":" << num(bank->m_bore)
          << ",\"deckHeight\":" << num(bank->m_deckHeight) << ",\"x\":" << num(bank->m_x)
          << ",\"y\":" << num(bank->m_y);
        o << ",\"head\":{\"chamberVolume\":" << num(h->m_combustionChamberVolume)
          << ",\"intakeRunnerVolume\":" << num(h->m_intakeRunnerVolume)
          << ",\"intakeRunnerArea\":" << num(h->m_intakeRunnerCrossSectionArea)
          << ",\"exhaustRunnerVolume\":" << num(h->m_exhaustRunnerVolume)
          << ",\"exhaustRunnerArea\":" << num(h->m_exhaustRunnerCrossSectionArea)
          << ",\"intakePortFlow\":" << fn(h->m_intakePortFlow)
          << ",\"exhaustPortFlow\":" << fn(h->m_exhaustPortFlow)
          << ",\"intakeCam\":" << cam(vt->m_intakeCamshaft)
          << ",\"exhaustCam\":" << cam(vt->m_exhaustCamshaft) << "}";
        o << ",\"cylinders\":[";
        for (int i = 0; i < bank->m_cylinderCount; ++i, ++cyl) {
            Piston *p = e->getPiston(cyl);
            ConnectingRod *r = e->getConnectingRod(cyl);
            o << (i ? "," : "") << "{\"journal\":" << r->m_journal
              << ",\"piston\":{\"mass\":" << num(p->m_mass) << ",\"blowby\":" << num(p->m_blowby_k)
              << ",\"compressionHeight\":" << num(p->m_compressionHeight)
              << ",\"wristPinPosition\":" << num(p->m_wristPinLocation)
              << ",\"displacement\":" << num(p->m_displacement) << "}"
              << ",\"rod\":{\"mass\":" << num(r->m_m) << ",\"momentOfInertia\":" << num(r->m_I)
              << ",\"centerOfMass\":" << num(r->m_centerOfMass) << ",\"length\":" << num(r->m_length) << "}"
              << ",\"intake\":" << (h->m_cylinders[i].intake - e->m_intakes)
              << ",\"exhaust\":" << (h->m_cylinders[i].exhaustSystem - e->m_exhaustSystems)
              << ",\"primaryLength\":" << num(h->m_cylinders[i].headerPrimaryLength)
              << ",\"soundAttenuation\":" << num(h->m_cylinders[i].soundAttenuation) << "}";
        }
        o << "]}";
    }
    o << "]";

    o << ",\"intakes\":[";
    for (int i = 0; i < e->getIntakeCount(); ++i) {
        Intake *in = e->getIntake(i);
        o << (i ? "," : "") << "{\"plenumVolume\":" << num(in->m_system.m_state.V)
          << ",\"plenumArea\":" << num(in->m_crossSectionArea) << ",\"flowK\":" << num(in->m_inputFlowK)
          << ",\"idleFlowK\":" << num(in->m_idleFlowK) << ",\"runnerFlowK\":" << num(in->m_runnerFlowRate)
          << ",\"molecularAfr\":" << num(in->m_molecularAfr)
          << ",\"idleThrottlePlate\":" << num(in->m_idleThrottlePlatePosition)
          << ",\"runnerLength\":" << num(in->m_runnerLength)
          << ",\"velocityDecay\":" << num(in->m_velocityDecay) << "}";
    }
    o << "],\"exhausts\":[";
    for (int i = 0; i < e->getExhaustSystemCount(); ++i) {
        ExhaustSystem *x = e->getExhaustSystem(i);
        std::string name = x->m_impulseResponse->getFilename();
        name = name.substr(name.find("sound-library/") + 14);
        o << (i ? "," : "") << "{\"length\":" << num(x->m_length)
          << ",\"collectorArea\":" << num(x->m_collectorCrossSectionArea)
          << ",\"outletFlowK\":" << num(x->m_outletFlowRate)
          << ",\"primaryTubeLength\":" << num(x->m_primaryTubeLength)
          << ",\"primaryFlowK\":" << num(x->m_primaryFlowRate)
          << ",\"velocityDecay\":" << num(x->m_velocityDecay)
          << ",\"audioVolume\":" << num(x->m_audioVolume)
          << ",\"impulseResponse\":\"" << name << "\""
          << ",\"impulseVolume\":" << num(x->m_impulseResponse->getVolume()) << "}";
    }
    IgnitionModule *im = e->getIgnitionModule();
    o << "],\"ignition\":{\"timing\":" << fn(im->m_timingCurve) << ",\"revLimit\":" << num(im->m_revLimit)
      << ",\"limiterDuration\":" << num(im->m_limiterDuration) << ",\"firing\":[";
    for (int i = 0; i < im->m_cylinderCount; ++i) o << (i ? "," : "") << num(im->m_plugs[i].angle);
    o << "]},\"audio\":{\"hfGain\":" << num(e->m_initialHighFrequencyGain)
      << ",\"noise\":" << num(e->m_initialNoise) << ",\"jitter\":" << num(e->m_initialJitter) << "}}";
    return o.str();
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: reference <script.mr> <spec.json> <trace.txt>\n");
        return 2;
    }
    es_script::Compiler compiler;
    compiler.initialize();
    if (!compiler.compile(argv[1])) {
        std::ifstream log("error_log.log");
        std::cerr << log.rdbuf();
        return 1;
    }
    const es_script::Compiler::Output output = compiler.execute();
    compiler.destroy();
    Engine *engine = output.engine;
    std::ofstream(argv[2]) << spec(engine) << "\n";

    Simulator *sim = engine->createSimulator(output.vehicle, output.transmission);
    engine->calculateDisplacement();
    sim->setSimulationFrequency(engine->getSimulationFrequency());
    for (int i = 0; i < engine->getCrankshaftCount(); ++i) {
        engine->getCrankshaft(i)->m_body.v_theta = -engine->getStarterSpeed();
    }
    engine->getIgnitionModule()->m_enabled = true;
    sim->m_dyno.m_hold = true;
    sim->m_starterMotor.m_enabled = false;

    if (getenv("DUMP")) {
        auto *pes = static_cast<PistonEngineSimulator *>(sim);
        for (int i = 0; i < pes->m_system->getRigidBodyCount(); ++i) {
            auto *b = pes->m_system->getRigidBody(i);
            fprintf(stderr, "body %d p=(%.17g, %.17g) th=%.17g v=(%.17g, %.17g, %.17g) m=%.17g I=%.17g\n",
                i, b->p_x, b->p_y, b->theta, b->v_x, b->v_y, b->v_theta, b->m, b->I);
        }
        for (int i = 0; i < pes->m_system->getConstraintCount(); ++i) {
            auto *c = pes->m_system->m_constraints[i];
            fprintf(stderr, "constraint %d rows=%d bodies=%d [%d %d]\n", i, c->getConstraintCount(),
                c->m_bodyCount, c->m_bodies[0]->index, c->m_bodyCount > 1 ? c->m_bodies[1]->index : -1);
        }
        for (int i = 0; i < engine->getCylinderCount(); ++i) {
            fprintf(stderr, "chamber %d V=%.17g P=%.17g\n", i, engine->getChamber(i)->m_system.volume(), engine->getChamber(i)->m_system.pressure());
        }
    }
    // The synthesizer as the game had it, without jitter (its random source differs).
    Synthesizer &synth = sim->synthesizer();
    Synthesizer::AudioParameters audio = synth.getAudioParameters();
    audio.inputSampleNoise = 0;
    audio.airNoise = static_cast<float>(engine->getInitialNoise());
    audio.dF_F_mix = static_cast<float>(engine->getInitialHighFrequencyGain());
    synth.setAudioParameters(audio);
    for (int i = 0; i < engine->getExhaustSystemCount(); ++i) {
        ImpulseResponse *response = engine->getExhaustSystem(i)->getImpulseResponse();
        std::ifstream file(response->getFilename(), std::ios::binary);
        std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        size_t p = 12;
        while (p + 8 <= data.size() && std::string(data.data() + p, 4) != "data") {
            uint32_t size;
            memcpy(&size, data.data() + p + 4, 4);
            p += 8 + size + (size & 1);
        }
        uint32_t size;
        memcpy(&size, data.data() + p + 4, 4);
        std::vector<int16_t> samples(size / 2);
        memcpy(samples.data(), data.data() + p + 8, size);
        synth.initializeImpulseResponse(samples.data(), (unsigned)samples.size(), (float)response->getVolume(), i);
    }
    synth.clearOutput();
    FILE *audioOut = fopen((std::string(argv[3]) + ".pcm").c_str(), "wb");

    FILE *trace = fopen(argv[3], "w");
    sim->startFrameSteps(SCENARIO_STEPS);
    for (int step = 0; step < SCENARIO_STEPS; ++step) {
        const Controls c = scenario(step);
        sim->m_dyno.m_enabled = c.dyno;
        sim->m_dyno.m_rotationSpeed = units::rpm(c.rpm);
        engine->setSpeedControl(c.throttle);
        sim->simulateStep();
        if (getenv("DUMP") && step == 0) {
            auto *ps = static_cast<PistonEngineSimulator *>(sim);
            for (int i = 0; i < ps->m_system->getRigidBodyCount(); ++i) {
                auto *b = ps->m_system->getRigidBody(i);
                fprintf(stderr, "after body %d p=(%.17g, %.17g) th=%.17g v=(%.17g, %.17g, %.17g)\n",
                    i, b->p_x, b->p_y, b->theta, b->v_x, b->v_y, b->v_theta);
            }
            auto *os = static_cast<atg_scs::OptimizedNsvRigidBodySystem *>(ps->m_system);
            for (int i = 0; i < os->m_iv.lambda.getHeight(); ++i)
                fprintf(stderr, "lambda %d %.17g right %.17g lim %.17g %.17g\n", i, os->m_iv.lambda.get(0, i), os->m_iv.right.get(0, i), os->m_iv.limits.get(0, i), os->m_iv.limits.get(1, i));
        }
        auto *pes = static_cast<PistonEngineSimulator *>(sim);
        fprintf(trace, "%.17g %.17g %.17g %.17g %.17g\n",
            engine->getCrankshaft(0)->m_body.v_theta,
            engine->getCrankshaft(0)->m_body.theta,
            engine->getChamber(0)->m_system.pressure(),
            engine->getChamber(2)->m_exhaustRunnerAndPrimary.pressure(),
            pes->m_exhaustFlowStagingBuffer[0]);
        if (step % 100 == 99) {
            synth.endInputBlock();
            synth.renderPending();
            std::vector<int16_t> pcm(synth.outputAvailable());
            sim->readAudioOutput((int)pcm.size(), pcm.data());
            fwrite(pcm.data(), 2, pcm.size(), audioOut);
        }
    }
    fclose(trace);
    fclose(audioOut);
    return 0;
}
