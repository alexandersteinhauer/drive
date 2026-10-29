import { expect, test } from "bun:test";
import { OUTDOORS, outdoorImpulse } from "./outdoors";

const RATE = 44100;
const energy = (x: Float32Array, from = 0, to = x.length) => {
  let e = 0;
  for (let i = from; i < to; i++) e += x[i]! * x[i]!;
  return e;
};

test("the surroundings answer late, die away and differ in each ear", () => {
  const [left, right] = outdoorImpulse(RATE);
  const first = Math.round(OUTDOORS.firstReflection * RATE);
  // Nothing before the first reflection: no room around the listener.
  expect(energy(left, 0, first)).toBe(0);
  // Down ~60 dB by the decay time.
  const at = Math.round(OUTDOORS.decay * RATE);
  const late = energy(left, at, at + RATE / 10) / (RATE / 10);
  const early = energy(left, first, first + RATE / 10) / (RATE / 10);
  expect(10 * Math.log10(late / early)).toBeLessThan(-45);
  // Each ear its own surroundings.
  let lr = 0;
  for (let i = 0; i < left.length; i++) lr += left[i]! * right[i]!;
  expect(Math.abs(lr) / Math.sqrt(energy(left) * energy(right))).toBeLessThan(
    0.3,
  );
  // The same surroundings every time.
  expect(outdoorImpulse(RATE)[0]).toEqual(left);
});
