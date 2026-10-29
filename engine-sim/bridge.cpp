// WebAssembly entry points for engine-sim, run single-threaded from an AudioWorklet.
//
// The game's car simulation stays authoritative: each audio block it supplies the engine
// speed and throttle, the dynamometer (in hold mode) pins the simulated crankshaft to that
// speed, and the simulated engine works against it at that throttle. The sound therefore
// follows load: wide open at low rpm is a strained bark, a closed throttle at speed is
// overrun, and a free rev in neutral is light.

#include "scripting/include/compiler.h"

#include "include/engine.h"
#include "include/simulator.h"
#include "include/constants.h"
#include "include/units.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

Engine *g_engine = nullptr;
Vehicle *g_vehicle = nullptr;
Transmission *g_transmission = nullptr;
Simulator *g_simulator = nullptr;

int g_sampleRate = 44100;
double g_stepDebt = 0; // fractional simulation steps owed to the audio clock
std::vector<int16_t> g_pcm;
std::vector<float> g_out;

double g_targetRpm = 0;
double g_targetThrottle = 0;
bool g_engaged = true;
double g_throttle = 0; // eased towards the target: a foot, not a switch
double g_dynoRpm = 0;  // slewed towards the target while the clutch takes up
double g_overrun = 0;  // see es_overrun

bool readWav(const std::string &path, std::vector<int16_t> *samples) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (data.size() < 12 || std::string(data.data(), 4) != "RIFF") return false;

    int channels = 1, bits = 16;
    for (size_t p = 12; p + 8 <= data.size();) {
        const std::string id(data.data() + p, 4);
        uint32_t size;
        memcpy(&size, data.data() + p + 4, 4);
        const char *body = data.data() + p + 8;
        if (id == "fmt ") {
            uint16_t c, b;
            memcpy(&c, body + 2, 2);
            memcpy(&b, body + 14, 2);
            channels = c;
            bits = b;
        } else if (id == "data") {
            if (bits != 16) return false;
            const size_t frames = std::min<size_t>(size, data.size() - (p + 8)) / (2 * channels);
            samples->resize(frames);
            for (size_t i = 0; i < frames; ++i) {
                memcpy(&(*samples)[i], body + i * 2 * channels, 2); // first channel
            }
            return true;
        }
        p += 8 + size + (size & 1);
    }
    return false;
}

void dumpLog(const char *path) {
    std::ifstream log(path);
    if (log) std::cerr << log.rdbuf();
}

} // namespace

extern "C" {

void es_set_gain(double gain);

// Compiles the engine script and builds the simulator. Returns 0 on success.
int es_init(const char *scriptPath, int simulationFrequency) {
    es_script::Compiler compiler;
    compiler.initialize();
    if (!compiler.compile(scriptPath)) {
        compiler.destroy();
        dumpLog("error_log.log");
        return 1;
    }
    const es_script::Compiler::Output output = compiler.execute();
    compiler.destroy();

    g_engine = output.engine;
    g_vehicle = output.vehicle;
    g_transmission = output.transmission;
    if (g_engine == nullptr) return 2;

    if (g_vehicle == nullptr) {
        Vehicle::Parameters p;
        p.mass = units::mass(800, units::kg);
        p.diffRatio = 4.375;
        p.tireRadius = units::distance(0.315, units::m);
        p.dragCoefficient = 0.48;
        p.crossSectionArea = 1.8;
        p.rollingResistance = 100;
        g_vehicle = new Vehicle;
        g_vehicle->initialize(p);
    }
    if (g_transmission == nullptr) {
        static const double ratios[] = {3.80, 2.06, 1.32, 0.89};
        Transmission::Parameters p;
        p.GearCount = 4;
        p.GearRatios = ratios;
        p.MaxClutchTorque = units::torque(200.0, units::ft_lb);
        g_transmission = new Transmission;
        g_transmission->initialize(p);
    }

    g_simulator = g_engine->createSimulator(g_vehicle, g_transmission);
    g_engine->calculateDisplacement();
    g_simulator->setSimulationFrequency(
        simulationFrequency > 0 ? simulationFrequency : g_engine->getSimulationFrequency());

    Synthesizer &synth = g_simulator->synthesizer();
    Synthesizer::AudioParameters audio = synth.getAudioParameters();
    audio.inputSampleNoise = static_cast<float>(g_engine->getInitialJitter());
    audio.airNoise = static_cast<float>(g_engine->getInitialNoise());
    audio.dF_F_mix = static_cast<float>(g_engine->getInitialHighFrequencyGain());
    synth.setAudioParameters(audio);

    for (int i = 0; i < g_engine->getExhaustSystemCount(); ++i) {
        ImpulseResponse *response = g_engine->getExhaustSystem(i)->getImpulseResponse();
        std::vector<int16_t> samples;
        if (!readWav(response->getFilename(), &samples)) {
            std::cerr << "engine-sim: cannot read " << response->getFilename() << "\n";
            return 3;
        }
        synth.initializeImpulseResponse(
            samples.data(), (unsigned int)samples.size(), (float)response->getVolume(), i);
    }
    synth.clearOutput();
    es_set_gain(0.003); // headroom for full throttle at the redline

    // The dyno holds speed in whichever direction the crank already turns, so start it
    // turning the way the starter would (negative), not from rest.
    for (int i = 0; i < g_engine->getCrankshaftCount(); ++i) {
        g_engine->getCrankshaft(i)->m_body.v_theta = -g_engine->getStarterSpeed();
    }

    g_engine->getIgnitionModule()->m_enabled = true;
    g_simulator->m_dyno.m_enabled = true;
    g_simulator->m_dyno.m_hold = true;
    g_simulator->m_starterMotor.m_enabled = false;
    return 0;
}

int es_sample_rate() { return 44100; }

// Engine speed (rpm) the dyno holds, throttle 0..1, and whether the clutch is engaged.
// With the clutch out (a gear change) the simulated engine runs free and revs down by
// itself; once it is back in, the dyno pulls it to the car's speed.
void es_set_controls(double rpm, double throttle, int engaged) {
    g_targetRpm = rpm;
    g_targetThrottle = throttle;
    g_engaged = engaged != 0;
}

// engine-sim normally levels its output to a constant peak, which would flatten the
// difference between load and overrun; a fixed gain keeps it.
void es_set_gain(double gain) {
    Synthesizer::AudioParameters audio = g_simulator->synthesizer().getAudioParameters();
    audio.levelerMinGain = audio.levelerMaxGain = static_cast<float>(gain);
    g_simulator->synthesizer().setAudioParameters(audio);
}

void es_set_volume(double volume) {
    Synthesizer::AudioParameters audio = g_simulator->synthesizer().getAudioParameters();
    audio.volume = static_cast<float>(volume);
    g_simulator->synthesizer().setAudioParameters(audio);
}

// Simulates enough to produce `frames` samples at 44.1 kHz and returns them as floats.
float *es_render(int frames) {
    if ((int)g_out.size() < frames) g_out.resize(frames);
    if (g_simulator == nullptr) {
        std::fill(g_out.begin(), g_out.end(), 0.0f);
        return g_out.data();
    }

    const double dt = (double)frames / g_sampleRate;

    const double tau = g_targetThrottle < g_throttle ? 0.05 : 0.12;
    g_throttle += (g_targetThrottle - g_throttle) * (1 - std::exp(-dt / tau));
    g_engine->setSpeedControl(g_throttle);

    // The clutch lets go once the foot is (nearly) off, so the revs don't flare; running
    // free, the engine must not sink below the car's (idle) speed.
    const double engineRpm = g_engine->getRpm();
    const bool hold = g_engaged || g_throttle > 0.1 || engineRpm < g_targetRpm;
    if (!hold) {
        g_dynoRpm = engineRpm;
    } else if (!g_engaged) {
        // Clutch out, foot still lifting: hold where it is (not the car's rev-match).
    } else {
        // The clutch takes up over a few tenths of a second; small changes pass at once.
        const double maxStep = 8000.0 * dt;
        g_dynoRpm += std::max(-maxStep, std::min(maxStep, g_targetRpm - g_dynoRpm));
    }
    g_simulator->m_dyno.m_enabled = hold;

    if (hold) {
        // Follows the airflow rather than the pedal: the throttle plate (gamma 2) passes
        // roughly sin(s^2 pi/2) of full flow, so most of the drop comes in the last third.
        const double s = std::min(1.0, std::max(0.0, g_throttle));
        g_overrun = 1 - std::sin(s * s * constants::pi / 2);
    } else {
        // Let go by the clutch: fade, so the change of sound doesn't open a hole.
        g_overrun *= std::exp(-dt / 0.2);
    }
    g_simulator->m_dyno.m_rotationSpeed = units::rpm(std::max(0.0, g_dynoRpm));

    Synthesizer &synth = g_simulator->synthesizer();
    const int frequency = g_simulator->getSimulationFrequency();

    // Simulate the steps this block is worth, then top up if the synthesizer still has
    // too little output (startup, resampling remainders).
    g_stepDebt += dt * frequency;
    int steps = (int)g_stepDebt;
    g_stepDebt -= steps;
    for (int guard = 0; guard < 8; ++guard) {
        if (steps > 0) {
            g_simulator->startFrameSteps(steps);
            while (g_simulator->simulateStep()) {}
            g_simulator->endFrame();
            synth.renderPending();
        }
        if (synth.outputAvailable() >= frames) break;
        steps = std::max(1, (frames - synth.outputAvailable()) * frequency / g_sampleRate + 1);
    }

    if ((int)g_pcm.size() < frames) g_pcm.resize(frames);
    g_simulator->readAudioOutput(frames, g_pcm.data());
    for (int i = 0; i < frames; ++i) g_out[i] = g_pcm[i] / 32768.0f;
    return g_out.data();
}

// How closed the throttle is while the dyno holds the engine (0..1), for the loudness
// stage: held on a closed throttle, the exhaust goes ~20 dB quieter than a real car does.
// Revving down freely in a gear change it stays louder, and needs no lift.
double es_overrun() { return g_overrun; }

double es_engine_rpm() { return g_engine ? g_engine->getRpm() : 0; }
double es_dyno_torque() { return g_simulator ? g_simulator->getFilteredDynoTorque() : 0; }

} // extern "C"
