#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

const int N = 1024;

void randomInit(float* data, int size) {
	for (int i = 0; i < size; i++) {
		data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
	}
}

void matrixMulCPU(const float* A, const float* B, float* C, int n) {
	for (int row = 0; row < n; row++) {
		for (int col = 0; col < n; col++) {
			float sum = 0.f;
			for (int k = 0; k < n; k++) {
				sum += A[row * n + k] * B[k * n + col];
			}
			C[row * n + col] = sum;
		}
	}
}

bool verifyResult(const float* gpuRes, const float* cpuRes, int size) {
	const float epsilon = 1e-3f;
	for (int i = 0; i < size; i++) {
		if (std::fabs(gpuRes[i] - cpuRes[i]) > epsilon) {
			std::cerr << "Mismatch at index " << i << ": GPU " << gpuRes[i] << " vs CUP " << cpuRes[i] << std::endl;
			return false;
		}
	}
	return true;
}

int main(void) {
	std::cout << "Step 0: Initializing Infrastructure..." << std::endl;
	std::cout << "Matrix Size: " << N << " x " << N << std::endl;

	size_t bytes = N * N * sizeof(float);

	std::vector<float> h_A(N * N);
	std::vector<float> h_B(N * N);
	std::vector<float> h_C_CPU(N * N);
	std::vector<float> h_C_GPU(N * N);

	randomInit(h_A.data(), N * N);
	randomInit(h_B.data(), N * N);

	std::cout << "Calculating on CPU (Golden Reference)...";
	auto startCPU = std::chrono::high_resolution_clock::now();

	matrixMulCPU(h_A.data(), h_B.data(), h_C_CPU.data(), N);

	auto endCPU = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float, std::milli> durationCPU = endCPU - startCPU;
	std::cout << "Done! (" << durationCPU.count() << " ms)" << std::endl;

	std::cout << "Infrastructure Ready. Waiting for Kernel Implementation." << std::endl;

	return 0;
}