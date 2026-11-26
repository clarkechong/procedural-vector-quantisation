// Standalone packed 4-bit decoder and fused dot-product benchmark.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#define BOTH __host__ __device__
#else
#define BOTH
#endif

enum Family { lfsr, binary, ternary, xorshift, splitmix };

BOTH int nibble(uint32_t word, int shift) {
    return (int((word >> shift) & 15) ^ 8) - 8;
}

BOTH uint32_t hash32(uint32_t value) {
    value = (value ^ (value >> 16)) * 0x7FEB352Du;
    value = (value ^ (value >> 15)) * 0x846CA68Bu;
    return value ^ (value >> 16);
}

BOTH uint64_t mix64(uint64_t value) {
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

template<Family family>
BOTH auto lfsr_stream(unsigned seed) {
    // Each group of four feedback bits depends only on already available bits.
    using Word = std::conditional_t<family == lfsr, uint64_t, uint32_t>;
    Word stream = seed;
    const int end = family == lfsr ? 24 : (family == binary ? 12 : 16);
    for (int offset = 0; offset < end; offset += 4) {
        const Word word = stream >> offset;
        const Word feedback = (word ^ (word >> 1) ^ (word >> 3) ^ (word >> 12)) & 15;
        stream |= feedback << (16 + offset);
    }
    return stream;
}

template<Family family, bool fused>
BOTH float decode(uint32_t word, float* output, const float* activation) {
    const unsigned seed = word & 65535;
    unsigned state = family == xorshift ? hash32(seed) : seed;
    const auto stream = family == lfsr || family == binary || family == ternary ?
        lfsr_stream<family>(seed) : 0;
    const float scale = ldexpf(1.0f, nibble(word, 16));
    const int coefficients[] = {nibble(word, 20), nibble(word, 24), nibble(word, 28)};
    float dot = 0;
    for (int row = 0; row < 8; ++row) {
        float decoded = 0;
        int integer_sum = 0;
        for (int col = 0; col < 3; ++col) {
            if constexpr (family == xorshift) {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                decoded += (float(state) * 0x1p-31f - 1.0f) * coefficients[col];
            } else if constexpr (family == splitmix) {
                const uint64_t counter = (uint64_t(seed) << 32) | unsigned(row * 3 + col);
                const uint32_t top = mix64(counter + 0x9E3779B97F4A7C15ull) >> 32;
                decoded += (float(top) * 0x1p-31f - 1.0f) * coefficients[col];
            } else {
                const int position = 1 + row * 3 + col;
                int term;
                if constexpr (family == lfsr)
                    term = (int((stream >> position) & 65535) - 32768) * coefficients[col];
                else
                    term = (stream >> position) & 1 ? -coefficients[col] : coefficients[col];
                if constexpr (family == ternary)
                    term &= -int((stream >> (position + 7)) & 1);
                integer_sum += term;
            }
        }
        if constexpr (family == lfsr)
            decoded = float(integer_sum) * (scale / 32767.0f);
        else if constexpr (family == binary || family == ternary)
            decoded = float(integer_sum) * scale;
        else
            decoded *= scale;
        if constexpr (fused)
            dot += decoded * activation[row];
        else
            output[row] = decoded;
    }
    return dot;
}

template<Family family, bool fused>
void cpu_pass(const uint32_t* input, float* output, size_t count, const float* activation) {
    for (size_t i = 0; i < count; ++i) {
        if constexpr (fused)
            output[i] = decode<family, true>(input[i], nullptr, activation);
        else
            decode<family, false>(input[i], output + i * 8, activation);
    }
}

#ifdef __HIPCC__
void check(hipError_t status) {
    if (status != hipSuccess)
        throw std::runtime_error(hipGetErrorString(status));
}

void streaming_copy(const std::vector<uint32_t>& input, int repeats) {
    uint32_t* source = nullptr;
    uint32_t* destination = nullptr;
    const size_t bytes = input.size() * sizeof(uint32_t);
    check(hipMalloc(&source, bytes));
    check(hipMalloc(&destination, bytes));
    check(hipMemcpy(source, input.data(), bytes, hipMemcpyHostToDevice));
    hipEvent_t start, end;
    check(hipEventCreate(&start));
    check(hipEventCreate(&end));
    check(hipMemcpyAsync(destination, source, bytes, hipMemcpyDeviceToDevice));
    std::vector<double> times;
    for (int i = 0; i < repeats; ++i) {
        check(hipEventRecord(start));
        check(hipMemcpyAsync(destination, source, bytes, hipMemcpyDeviceToDevice));
        check(hipEventRecord(end));
        check(hipEventSynchronize(end));
        float elapsed;
        check(hipEventElapsedTime(&elapsed, start, end));
        times.push_back(elapsed / 1000.0);
    }
    uint32_t check_word;
    check(hipMemcpy(&check_word, destination, sizeof(check_word), hipMemcpyDeviceToHost));
    if (check_word != input.front())
        throw std::runtime_error("Device copy disagrees with source");
    check(hipEventDestroy(start));
    check(hipEventDestroy(end));
    check(hipFree(destination));
    check(hipFree(source));
    std::sort(times.begin(), times.end());
    const double elapsed = times[times.size() / 2];
    std::cout << "HIP,raw,device_copy," << input.size() << ',' << repeats << ',' << elapsed
              << ",," << bytes / elapsed / 1e9 << ',' << 2 * bytes / elapsed / 1e9 << ",0,\n";
}

template<Family family, bool fused>
__global__ void decode_kernel(const uint32_t* input, float* output, size_t count) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const float activation[] = {1, 0.5f, -0.25f, -1, -0.75f, 0.25f, 2, -2};
    if constexpr (fused)
        output[index] = decode<family, true>(input[index], nullptr, activation);
    else
        decode<family, false>(input[index], output + index * 8, activation);
}
#endif

template<Family family, bool fused>
void benchmark(const char* name, const std::vector<uint32_t>& input, int repeats) {
    const size_t count = input.size();
    const size_t values = count * (fused ? 1 : 8);
    const float activation[] = {1, 0.5f, -0.25f, -1, -0.75f, 0.25f, 2, -2};
    std::vector<float> output(values);
    std::vector<double> times;
#ifdef __HIPCC__
    uint32_t* device_input = nullptr;
    float* device_output = nullptr;
    check(hipMalloc(&device_input, count * sizeof(uint32_t)));
    check(hipMalloc(&device_output, values * sizeof(float)));
    check(hipMemcpy(device_input, input.data(), count * sizeof(uint32_t), hipMemcpyHostToDevice));
    hipEvent_t start, end;
    check(hipEventCreate(&start));
    check(hipEventCreate(&end));
    auto launch = [&] {
        decode_kernel<family, fused><<<(count + 255) / 256, 256>>>(device_input, device_output, count);
        check(hipGetLastError());
    };
    launch();
    for (int run = 0; run < repeats; ++run) {
        check(hipEventRecord(start));
        launch();
        check(hipEventRecord(end));
        check(hipEventSynchronize(end));
        float elapsed;
        check(hipEventElapsedTime(&elapsed, start, end));
        times.push_back(elapsed / 1000.0);
    }
    check(hipMemcpy(output.data(), device_output, values * sizeof(float), hipMemcpyDeviceToHost));
    check(hipEventDestroy(start));
    check(hipEventDestroy(end));
    check(hipFree(device_output));
    check(hipFree(device_input));
    const char* backend = "HIP";
#else
    cpu_pass<family, fused>(input.data(), output.data(), count, activation);
    for (int run = 0; run < repeats; ++run) {
        const auto start = std::chrono::steady_clock::now();
        cpu_pass<family, fused>(input.data(), output.data(), count, activation);
        const auto end = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double>(end - start).count());
    }
    const char* backend = "CPU";
#endif
    const size_t checked_blocks = std::min<size_t>(count, 128);
    std::vector<float> reference(checked_blocks * (fused ? 1 : 8));
    cpu_pass<family, fused>(input.data(), reference.data(), checked_blocks, activation);
    double mismatch = 0;
    for (size_t i = 0; i < reference.size(); ++i)
        mismatch = std::max(mismatch, double(std::abs(reference[i] - output[i])));
    if (mismatch > 0.005)
        throw std::runtime_error("Device decode disagrees with host reference");
    double checksum = 0;
    for (size_t i = 0; i < values; i += 127)
        checksum += output[i];
    std::sort(times.begin(), times.end());
    const double elapsed = times[times.size() / 2];
    std::cout << backend << ',' << name << ',' << (fused ? "block_dot" : "materialize") << ','
              << count << ',' << repeats << ',' << elapsed << ',' << count * 8 / elapsed / 1e9 << ','
              << count * sizeof(uint32_t) / elapsed / 1e9 << ','
              << (count * sizeof(uint32_t) + values * sizeof(float)) / elapsed / 1e9
              << ',' << mismatch << ',' << checksum << '\n';
}

int main(int argc, char** argv) {
    try {
        const size_t count = argc > 1 ? std::stoull(argv[1]) : size_t(1) << 23;
        const int repeats = argc > 2 ? std::stoi(argv[2]) : 7;
        if (!count || count > size_t(1) << 27 || repeats < 1)
            throw std::invalid_argument("Pass a positive repeat count and up to 2^27 blocks");
        std::vector<uint32_t> input(count);
        for (size_t i = 0; i < count; ++i) {
            input[i] = hash32(uint32_t(i));
            if ((input[i] & 65535) == 0)
                input[i] |= 1;
        }
        std::cout << "backend,family,operation,blocks,repeats,median_seconds,gweights_per_s,"
                     "packed_input_gb_per_s,minimum_total_gb_per_s,max_reference_difference,checksum\n";
#ifdef __HIPCC__
        hipDeviceProp_t device;
        int device_id;
        check(hipGetDevice(&device_id));
        check(hipGetDeviceProperties(&device, device_id));
        std::cerr << device.name << " (" << device.gcnArchName << ")\n";
        streaming_copy(input, repeats);
#endif
        benchmark<lfsr, false>("lfsr", input, repeats);
        benchmark<lfsr, true>("lfsr", input, repeats);
        benchmark<binary, false>("binary", input, repeats);
        benchmark<binary, true>("binary", input, repeats);
        benchmark<ternary, false>("ternary", input, repeats);
        benchmark<ternary, true>("ternary", input, repeats);
        benchmark<xorshift, false>("xorshift", input, repeats);
        benchmark<xorshift, true>("xorshift", input, repeats);
        benchmark<splitmix, false>("splitmix", input, repeats);
        benchmark<splitmix, true>("splitmix", input, repeats);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
