// Fused paper-equation scoring for a set of candidate SeedLM seeds.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

#if defined(SEEDLM_USE_CBLAS)
#include <cblas.h>
#elif defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif
#if defined(__APPLE__) || defined(SEEDLM_USE_CBLAS)
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#endif

template <int C, int P>
inline void evaluate_seed(const float* weights, size_t seed, const float* inverse,
                          const float* bases, float& best, uint32_t& best_seed,
                          int8_t* best_codes, int8_t& best_exponent) {
    const float* fitted = inverse + seed * P * C;
    float coefficients[P];
    float peak = 0;
    for (int column = 0; column < P; ++column) {
        float dot = 0;
        for (int i = 0; i < C; ++i)
            dot += weights[i] * fitted[column * C + i];
        coefficients[column] = dot;
        peak = std::max(peak, std::abs(dot));
    }
    const int exponent = std::clamp(int((std::bit_cast<uint32_t>(peak) >> 23) & 255) - 127, -8, 7);
    const float scale = std::bit_cast<float>(uint32_t(127 + exponent) << 23);
    const float inv_scale = std::bit_cast<float>(uint32_t(127 - exponent) << 23);
    float h[P], q[P];
    for (int column = 0; column < P; ++column) {
        q[column] = std::clamp(std::nearbyint(coefficients[column] * inv_scale), -8.0f, 7.0f);
        h[column] = scale * q[column];
    }
    const float* raw = bases + seed * P * C;
    float error = 0;
    for (int i = 0; i < C; ++i) {
        float decoded = 0;
        for (int column = 0; column < P; ++column)
            decoded += raw[column * C + i] * h[column];
        const float residual = weights[i] - decoded;
        error += residual * residual;
    }
    if (error < best || (error == best && seed + 1 < best_seed)) {
        best = error;
        best_seed = uint32_t(seed + 1);
        best_exponent = int8_t(exponent);
        for (int column = 0; column < P; ++column)
            best_codes[column] = int8_t(q[column]);
    }
}

template <int C, int P>
void encode(const float* blocks, size_t count, const float* inverse, const float* bases,
            const uint16_t* index, const int32_t* keys, size_t pool, size_t threads,
            uint32_t* seeds, int8_t* codes, int8_t* exponents) {
    auto chunk = [&](size_t begin, size_t end) {
        for (size_t block = begin; block < end; ++block) {
            const float* weights = blocks + block * C;
            const uint16_t* candidates = index ? index + size_t(keys[block]) * pool : nullptr;
            float best = std::numeric_limits<float>::infinity();
            for (size_t entry = 0; entry < pool; ++entry) {
                const size_t seed = candidates ? candidates[entry] : entry;
                evaluate_seed<C, P>(weights, seed, inverse, bases, best, seeds[block],
                                    codes + block * P, exponents[block]);
            }
        }
    };
    threads = std::max(size_t(1), std::min(count, threads));
    std::vector<std::thread> workers;
    for (size_t i = 1; i < threads; ++i)
        workers.emplace_back(chunk, count * i / threads, count * (i + 1) / threads);
    chunk(0, count / threads);
    for (auto& worker : workers)
        worker.join();
}

extern "C" void paper_seedlm_encode(
    int width, const float* blocks, size_t count, const float* inverse, const float* bases,
    const uint16_t* index, const int32_t* keys, size_t pool, size_t threads,
    uint32_t* seeds, int8_t* codes, int8_t* exponents) {
    if (width == 8)
        encode<8, 3>(blocks, count, inverse, bases, index, keys, pool, threads, seeds, codes, exponents);
    else
        encode<12, 4>(blocks, count, inverse, bases, index, keys, pool, threads, seeds, codes, exponents);
}

#if defined(__APPLE__) || defined(SEEDLM_USE_CBLAS)
// Test the continuous projection bound for every seed, then evaluate the
// paper's integer coefficient rule for seeds that can improve the answer.
template <int C, int P>
void projected_encode(const float* blocks, size_t count, const float* inverse, const float* bases,
                      const float* orthogonal, size_t threads, uint32_t* seeds,
                      int8_t* codes, int8_t* exponents) {
    constexpr int block_tile = 96;
    constexpr int seed_tile = 512;
    constexpr int seed_count = 65535;
    const auto worker = [&](size_t begin, size_t end) {
        std::vector<float> projected(block_tile * seed_tile * P);
        for (size_t offset = begin; offset < end; offset += block_tile) {
            const int block_count = int(std::min(size_t(block_tile), end - offset));
            float best[block_tile], energy[block_tile], threshold[block_tile], guard[block_tile];
            for (int row = 0; row < block_count; ++row) {
                const size_t block = offset + row;
                const float* weights = blocks + block * C;
                best[row] = std::numeric_limits<float>::infinity();
                energy[row] = 0;
                for (int i = 0; i < C; ++i)
                    energy[row] += weights[i] * weights[i];
                // Cover FP32 projection and residual roundoff with a conservative margin.
                guard[row] = energy[row] * 1e-4f + 1e-15f;
                evaluate_seed<C, P>(weights, seeds[block] - 1, inverse, bases, best[row],
                                    seeds[block], codes + block * P, exponents[block]);
                threshold[row] = energy[row] - best[row] - guard[row];
            }
            for (int seed = 0; seed < seed_count; seed += seed_tile) {
                const int seeds_here = std::min(seed_tile, seed_count - seed);
                const int stride = seeds_here * P;
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, block_count, stride, C,
                            1, blocks + offset * C, C, orthogonal + seed * P * C, C, 0,
                            projected.data(), stride);
                for (int row = 0; row < block_count; ++row) {
                    const size_t block = offset + row;
                    const float* values = projected.data() + row * stride;
                    const auto refine = [&](int j, float score) {
                        if (score < threshold[row])
                            return;
                        evaluate_seed<C, P>(blocks + block * C, seed + j, inverse, bases,
                                            best[row], seeds[block], codes + block * P,
                                            exponents[block]);
                        threshold[row] = energy[row] - best[row] - guard[row];
                    };
                    int j = 0;
#ifdef __aarch64__
                    for (; j + 3 < seeds_here; j += 4) {
                        float32x4_t energy;
                        if constexpr (P == 3) {
                            const float32x4x3_t v = vld3q_f32(values + j * P);
                            energy = vmulq_f32(v.val[0], v.val[0]);
                            energy = vfmaq_f32(energy, v.val[1], v.val[1]);
                            energy = vfmaq_f32(energy, v.val[2], v.val[2]);
                        } else {
                            const float32x4x4_t v = vld4q_f32(values + j * P);
                            energy = vmulq_f32(v.val[0], v.val[0]);
                            energy = vfmaq_f32(energy, v.val[1], v.val[1]);
                            energy = vfmaq_f32(energy, v.val[2], v.val[2]);
                            energy = vfmaq_f32(energy, v.val[3], v.val[3]);
                        }
                        if (vmaxvq_f32(energy) >= threshold[row]) {
                            refine(j, vgetq_lane_f32(energy, 0));
                            refine(j + 1, vgetq_lane_f32(energy, 1));
                            refine(j + 2, vgetq_lane_f32(energy, 2));
                            refine(j + 3, vgetq_lane_f32(energy, 3));
                        }
                    }
#endif
                    for (; j < seeds_here; ++j) {
                        float score = 0;
                        for (int k = 0; k < P; ++k)
                            score += values[j * P + k] * values[j * P + k];
                        refine(j, score);
                    }
                }
            }
        }
    };
    // Partition by whole tiles to preserve GEMM dimensions in each worker.
    const size_t tiles = (count + block_tile - 1) / block_tile;
    threads = std::max(size_t(1), std::min(tiles, threads));
    std::vector<std::thread> workers;
    for (size_t i = 1; i < threads; ++i)
        workers.emplace_back(worker, tiles * i / threads * block_tile,
                             std::min(count, tiles * (i + 1) / threads * block_tile));
    worker(0, std::min(count, tiles / threads * block_tile));
    for (auto& thread : workers)
        thread.join();
}

extern "C" void paper_seedlm_project(
    int width, const float* blocks, size_t count, const float* inverse, const float* bases,
    const float* orthogonal, size_t threads, uint32_t* seeds, int8_t* codes, int8_t* exponents) {
    if (width == 8)
        projected_encode<8, 3>(blocks, count, inverse, bases, orthogonal, threads, seeds, codes, exponents);
    else
        projected_encode<12, 4>(blocks, count, inverse, bases, orthogonal, threads, seeds, codes, exponents);
}
#endif
