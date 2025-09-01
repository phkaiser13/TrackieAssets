#include <cuda_runtime.h>
#include <iostream>

// Declaração da função que lança o kernel (definida em augment.cu)
// Isso informa ao compilador C++ sobre a existência da função que está em um arquivo .cu
void launch_brightness_kernel(float* d_image, int width, int height, int channels, float brightness_factor);

// Usamos extern "C" para evitar o "name mangling" do C++, garantindo que o nome da função
// seja exatamente "apply_brightness_cuda" no binário compilado, facilitando a chamada via ctypes.
extern "C" {
    /**
     * @brief Aplica um ajuste de brilho a uma imagem usando CUDA.
     *
     * Esta função gerencia todo o ciclo de vida da operação na GPU:
     * 1. Aloca memória na GPU.
     * 2. Copia a imagem do CPU para a GPU.
     * 3. Lança o kernel CUDA para processar a imagem.
     * 4. Copia a imagem processada de volta para o CPU.
     * 5. Libera a memória da GPU.
     *
     * @param h_image_in Ponteiro para os dados da imagem de entrada no host (CPU).
     * @param h_image_out Ponteiro para o buffer de saída no host (CPU) onde a imagem processada será armazenada.
     *                    Este buffer deve ser pré-alocado pelo chamador.
     * @param width A largura da imagem.
     * @param height A altura da imagem.
     * @param channels O número de canais da imagem (ex: 3 para RGB).
     * @param brightness_factor O fator de brilho a ser aplicado.
     */
    void apply_brightness_cuda(float* h_image_in, float* h_image_out, int width, int height, int channels, float brightness_factor) {

        size_t image_size_bytes = width * height * channels * sizeof(float);
        float* d_image = nullptr;

        // 1. Alocar memória na GPU
        cudaError_t alloc_status = cudaMalloc((void**)&d_image, image_size_bytes);
        if (alloc_status != cudaSuccess) {
            std::cerr << "Erro ao alocar memória na GPU: " << cudaGetErrorString(alloc_status) << std::endl;
            return;
        }

        // 2. Copiar imagem do Host (CPU) para o Device (GPU)
        cudaError_t h2d_status = cudaMemcpy(d_image, h_image_in, image_size_bytes, cudaMemcpyHostToDevice);
        if (h2d_status != cudaSuccess) {
            std::cerr << "Erro ao copiar dados do host para o device: " << cudaGetErrorString(h2d_status) << std::endl;
            cudaFree(d_image);
            return;
        }

        // 3. Lançar o kernel CUDA
        launch_brightness_kernel(d_image, width, height, channels, brightness_factor);

        // Verificar se houve erros durante a execução do kernel
        cudaError_t kernel_status = cudaGetLastError();
        if (kernel_status != cudaSuccess) {
            std::cerr << "Erro na execução do kernel CUDA: " << cudaGetErrorString(kernel_status) << std::endl;
            cudaFree(d_image);
            return;
        }

        // 4. Copiar imagem processada do Device (GPU) para o Host (CPU)
        cudaError_t d2h_status = cudaMemcpy(h_image_out, d_image, image_size_bytes, cudaMemcpyDeviceToHost);
        if (d2h_status != cudaSuccess) {
            std.cerr << "Erro ao copiar dados do device para o host: " << cudaGetErrorString(d2h_status) << std::endl;
            cudaFree(d_image);
            return;
        }

        // 5. Liberar a memória da GPU
        cudaFree(d_image);
    }
}
