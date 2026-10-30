//! Steady flow through one pipe of a reference config, to see its pressure drop by section.
//!   pipedrop <reference.json> <pipe name> <inlet pressure bar> <inlet temperature K>
use engine_sim::duct::*;
use engine_sim::spec::DuctSpec;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&a[1]).unwrap()).unwrap();
    let spec: DuctSpec = serde_json::from_value(cfg["ducts"].clone()).unwrap();
    let p = spec.pipes.iter().find(|p| p.name.as_deref() == Some(a[2].as_str())).unwrap();
    let (p0, t0): (f64, f64) = (a[3].parse::<f64>().unwrap() * 1e5, a[4].parse().unwrap());
    let sections: Vec<Section> = p
        .sections
        .iter()
        .map(|s| Section {
            length: s.length,
            diameter: s.diameter,
            diameter_b: s.diameter_b.unwrap_or(s.diameter),
            area: s.area,
            square: s.square,
            wall: s.wall_temperature,
            resistivity: s.flow_resistivity,
            roughness: s.roughness.unwrap_or(STEEL_ROUGHNESS),
        })
        .collect();
    let c = (gamma(p.temperature) * RS * p.temperature).sqrt();
    let pipe = Pipe::with_sections(&sections, c / (3000.0 * 30.0), 1.013e5, p.temperature, 0.0, 0.25);
    let mut net = Network::default();
    let tank = net.add_node(Node::volume(100.0, p0, t0, 0.0, 0.0));
    let atm = net.add_node(Node::atmosphere(1.01325e5, 300.0, 0.0, 0.25));
    let i = net.add_pipe(pipe);
    net.connect(i, Side::A, tank);
    let out = net.connect(i, Side::B, atm);
    for _ in 0..4000 {
        net.step(5e-5);
    }
    let pipe = &net.pipes[i];
    println!("mass flow {:.3} kg/s", net.links[out].flow.m);
    let mut x = 0.0;
    let mut cell = 0;
    for s in &sections {
        let n = (s.length / pipe.dx).round() as usize;
        let (first, last) = (cell, (cell + n).min(pipe.cells()) - 1);
        let (pa, pb) = (pipe.prim[first], pipe.prim[last]);
        println!(
            "{x:5.3}-{:5.3} m d {:.4}{}: p {:.3} -> {:.3} bar, u {:.0} -> {:.0} m/s, T {:.0} -> {:.0} K",
            x + s.length,
            s.diameter,
            if s.area.is_some() { " (channels)" } else { "" },
            pa.p / 1e5,
            pb.p / 1e5,
            pa.u,
            pb.u,
            pa.t,
            pb.t
        );
        x += s.length;
        cell += n;
    }
}
