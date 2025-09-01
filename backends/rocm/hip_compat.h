#ifndef HIP_COMPAT_H
#define HIP_COMPAT_H

/*
 * Este é um header de compatibilidade para compilar código CUDA em plataformas ROCm/HIP.
 * Quando o compilador HIP (hipcc) é usado, ele define __HIP_PLATFORM_HCC__.
 * Usamos essa definição para substituir as chamadas da API CUDA por suas equivalentes da API HIP.
 *
 * Para usar: inclua este header no topo dos seus arquivos .cpp e .cu.
 * Exemplo: #include "hip_compat.h"
 *
 * Compile com o compilador hipcc.
 */

#ifdef __HIP_PLATFORM_HCC__
#include <hip/hip_runtime.h>

// Mapeia tipos de erro
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess

// Mapeia funções da API de runtime
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaFree hipFree
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString

// Mapeia tipos de enum para cópia de memória
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost

// Mapeia qualificadores de kernel e tipos de dimensão
// O HIP já define __global__, __host__, __device__, então não precisamos mapeá-los.
// dim3 também é definido no HIP.

#else
// Se não estivermos compilando com HIP, apenas inclua o header padrão do CUDA.
// Isso garante que o código ainda compile normalmente com nvcc.
#include <cuda_runtime.h>

#endif // __HIP_PLATFORM_HCC__

#endif // HIP_COMPAT_H
