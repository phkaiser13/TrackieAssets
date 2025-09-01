#include <cuda_runtime.h>
#include <stdint.h>

/*
 * CUDA Kernel para ajustar o brilho de uma imagem.
 * A imagem é tratada como um array de floats (normalizada entre 0 e 1).
 *
 * Parâmetros:
 * - image: Ponteiro para os dados da imagem na memória da GPU.
 * - width: Largura da imagem.
 * - height: Altura da imagem.
 * - channels: Número de canais da imagem (ex: 3 para RGB).
 * - brightness_factor: Fator de brilho a ser adicionado a cada pixel.
 */
__global__ void brightness_kernel(float* image, int width, int height, int channels, float brightness_factor) {
    // Calcula o índice global do thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Garante que o thread está dentro dos limites da imagem
    if (x < width && y < height) {
        // Itera sobre os canais da imagem (R, G, B)
        for (int c = 0; c < channels; ++c) {
            // Calcula o índice linear para o pixel (y, x) e canal c
            int idx = (y * width + x) * channels + c;

            // Lê o valor original do pixel
            float pixel_value = image[idx];

            // Aplica o fator de brilho
            float new_value = pixel_value + brightness_factor;

            // Garante que o novo valor permaneça no intervalo [0, 1] (clamping)
            image[idx] = fmaxf(0.0f, fminf(1.0f, new_value));
        }
    }
}

/*
 * Função host para lançar o kernel de brilho.
 * Esta função é a interface que será chamada pelo wrapper C++.
 *
 * Parâmetros:
 * - d_image: Ponteiro para a imagem na memória do dispositivo (GPU).
 * - width: Largura da imagem.
 * - height: Altura da imagem.
 * - channels: Número de canais.
 * - brightness_factor: Fator de brilho.
 */
void launch_brightness_kernel(float* d_image, int width, int height, int channels, float brightness_factor) {
    // Define o tamanho dos blocos e da grade
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Lança o kernel
    brightness_kernel<<<numBlocks, threadsPerBlock>>>(d_image, width, height, channels, brightness_factor);

    // Sincroniza para garantir que o kernel terminou antes de retornar
    cudaDeviceSynchronize();
}
