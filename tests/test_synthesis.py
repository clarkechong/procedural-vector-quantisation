import unittest

import numpy as np

from experiments.encoding import synthesized
from experiments.generator_algebra import bitvector
from experiments.synthesis import grammar


class SynthesisTests(unittest.TestCase):
    def test_interpreters_agree_on_unsigned_edges(self):
        seeds = [0, 1, 7, 31, 32, 127, 255, 4095]
        e = np.arange(-4, 4, dtype=np.int8)
        for op in bitvector.OPERATIONS:
            with self.subTest(op=op):
                args = ("r0",) if op in {"NOT", "POPCOUNT"} else ("r0", "i")
                program = (bitvector.Instruction("SUB", ("s", 0x80000001)), bitvector.Instruction(op, args))
                vectors = bitvector.codebook(program, 12, 40)
                payload = synthesized.pack(seeds, e, 12)
                self.assertEqual(16, len(payload))
                self.assertEqual(bytes.fromhex("00c001d007e01ff020007f10ff20ff3f"), payload)
                np.testing.assert_array_equal(
                    synthesized.decode(program, payload, 12, 40), np.ldexp(vectors[seeds].astype(float), e[:, None])
                )

    def test_bit_packing_across_byte_boundaries(self):
        program = (bitvector.Instruction("XOR", ("s", "i")),)
        seeds, exponents = [5, 63, 7, 14, 23], [-8, 7, -1, 0, 4]
        payload = synthesized.pack(seeds, exponents, 6)
        self.assertEqual(7, len(payload))
        arrays = bitvector.codebook(program, 6, 8)
        np.testing.assert_array_equal(synthesized.decode(program, payload, 6, 8, count=5),
                                      np.ldexp(arrays[seeds].astype(float), np.asarray(exponents)[:, None]))
        with self.assertRaises(ValueError):
            synthesized.decode(program, payload[:-1], 6, 8, count=5)
        with self.assertRaises(ValueError):
            synthesized.decode(program, payload[:-1] + bytes([payload[-1] | 0x80]), 6, 8, count=5)

    def test_enumeration_retains_forks_and_computed_literals(self):
        options = tuple(tuple(grammar.choices(n, ("ADD", "MUL", "XOR"), (0, 1, 2))) for n in range(3))
        p0 = bitvector.Instruction("ADD", ("s", "i"))
        p1 = bitvector.Instruction("XOR", ("s", "i"))
        p2 = bitvector.Instruction("MUL", ("r0", "r1"))
        self.assertIn((p0, p1, p2), list(grammar.extensions((p0, p1), options)))
        p1 = bitvector.Instruction("MUL", ("s", "r0"))
        self.assertIn((p0, p1, p2), list(grammar.extensions((p0, p1), options)))
        self.assertTrue(any(all(isinstance(x, int) for x in row.args) for row in options[0]))
        self.assertNotIn(bitvector.Instruction("ADD", ("i", "s")), options[0])
        self.assertFalse(grammar.live_instructions((p0, bitvector.Instruction("ADD", ("s", "i")))))

    def test_twenty_bit_seed_payload_preserves_high_bits(self):
        program = (bitvector.Instruction("SHR", ("s", "i")),)
        seeds = np.array([0, 65536, (1 << 20) - 1], dtype=np.uint32)
        exponents = np.array([-8, 0, 7])
        payload = synthesized.pack(seeds, exponents, 20)
        self.assertEqual(len(payload), 9)
        for row, (seed, exponent) in enumerate(zip(seeds, exponents)):
            word = int.from_bytes(payload[3 * row:3 * row + 3], "little")
            self.assertEqual(word, int(seed) | ((int(exponent) & 15) << 20))
        expected = np.ldexp(bitvector.evaluate(program, seeds[:, None], np.arange(8)),
                            exponents[:, None])
        np.testing.assert_array_equal(synthesized.decode(program, payload, 20, 8), expected)
        with self.assertRaises(ValueError):
            synthesized.pack([1 << 20], [0], 20)

    def test_twenty_four_bit_fields_cross_byte_boundaries(self):
        program = (bitvector.Instruction("SHR", ("s", "i")),)
        seeds = np.array([0, (1 << 20) + 7, (1 << 24) - 1], dtype=np.uint32)
        exponents = np.array([-8, 7, -1], dtype=np.int8)
        payload = synthesized.pack(seeds, exponents, 24)
        self.assertEqual(11, len(payload))
        stream = int.from_bytes(payload, "little")
        for row, (seed, exponent) in enumerate(zip(seeds, exponents)):
            self.assertEqual((stream >> (28 * row)) & ((1 << 28) - 1),
                             int(seed) | ((int(exponent) & 15) << 24))
        expected = np.ldexp(bitvector.evaluate(program, seeds[:, None], np.arange(8)),
                            exponents[:, None])
        np.testing.assert_array_equal(synthesized.decode(program, payload, 24, 8, count=3), expected)
        with self.assertRaises(ValueError):
            synthesized.decode(program, payload[:-1] + bytes([payload[-1] | 0x10]), 24, 8, count=3)
        with self.assertRaises(ValueError):
            synthesized.pack([1 << 24], [0], 24)
        with self.assertRaises(ValueError):
            synthesized.pack([-1], [0], 24)

    def test_fingerprint_matches_are_verified(self):
        first = (bitvector.Instruction("SHL", ("s", "i")),)
        equivalent = (bitvector.Instruction("ROTL", ("s", "i")),)
        different = (bitvector.Instruction("SHR", ("s", "i")),)
        screen = grammar.SemanticScreen(12, 8)
        # Force both distinct non-degenerate outputs to share the same probes.
        screen.probes = np.array([[1], [2]], dtype=np.uint32)
        self.assertTrue(screen.accept(first))
        self.assertFalse(screen.accept(equivalent))
        self.assertEqual(1, screen.audit["collision_checks"])
        self.assertTrue(screen.accept(different))

        # With only the zero probe, all functions collide and appear seed-independent.
        screen = grammar.SemanticScreen(12, 8)
        screen.probes = np.array([[0]], dtype=np.uint32)
        self.assertTrue(screen.accept(first))
        self.assertTrue(screen.accept(different))
        self.assertEqual(1, screen.audit["distinct_collisions"])
        self.assertFalse(screen.accept((bitvector.Instruction("ADD", ("s", 1)),)))

    def test_exact_search_matches_all_encodings(self):
        rng = np.random.default_rng(1001)
        vectors = rng.integers(-128, 128, size=(4096, 8), dtype=np.int16).astype(np.int8)
        vectors[0] = 0
        blocks = np.concatenate((rng.standard_normal(size=(6, 8)), np.zeros((1, 8)),
                                 1000.0 * rng.standard_normal(size=(1, 8))))
        errors, seeds, exponents, reconstruction = synthesized.exact_search(vectors, blocks, seed_batch=256)
        table = np.ldexp(vectors[:, None, :].astype(float), np.arange(-8, 8)[None, :, None])
        brute = np.sum((blocks[:, None, None, :] - table) ** 2, axis=-1)
        np.testing.assert_array_equal(errors, brute.min(axis=(1, 2)))
        np.testing.assert_array_equal(reconstruction, table[seeds, exponents + 8])
        vectors = np.array([[8]], dtype=np.int8)
        blocks = np.array([[12.0], [12.0001], [-2.0]])
        _, _, exponents, _ = synthesized.exact_search(vectors, blocks)
        np.testing.assert_array_equal(exponents, [0, 1, -8])

    def test_streamed_program_reference_matches_materialized_search(self):
        program = (bitvector.Instruction("MUL", ("s", 0x9E3779B9)),
                   bitvector.Instruction("SHR", ("r0", "i")),
                   bitvector.Instruction("SHR", ("r1", "r1")))
        blocks = np.vstack((np.zeros((1, 8)), np.random.default_rng(11).normal(size=(6, 8))))
        expected = synthesized.exact_search(bitvector.codebook(program, 10, 8), blocks)
        actual = synthesized.exact_program_search(program, blocks, 10, seed_batch=71)
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)


if __name__ == "__main__":
    unittest.main()
