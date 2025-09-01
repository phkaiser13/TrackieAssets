#import <Metal/Metal.h>

// Estrutura para encapsular os objetos Metal necessários para a computação.
// Isso evita a reinicialização em cada chamada se a função for usada várias vezes.
typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> pipelineState;
} MetalState;

// Inicializador para o estado do Metal. É chamado uma vez.
MetalState* initialize_metal_state() {
    MetalState* state = (MetalState*)malloc(sizeof(MetalState));
    if (!state) return NULL;

    state->device = MTLCreateSystemDefaultDevice();
    if (!state->device) {
        NSLog(@"Erro: Dispositivo Metal não encontrado.");
        free(state);
        return NULL;
    }

    state->commandQueue = [state->device newCommandQueue];

    // O Python script irá fornecer o caminho para a biblioteca compilada.
    // Aqui, assumimos que ela está no mesmo diretório ou em um caminho conhecido.
    // Os engenheiros do usuário serão responsáveis por compilar .metal para .metallib.
    NSError* error = nil;
    // O nome 'default.metallib' é um placeholder. O script Python o encontrará.
    state->library = [state->device newDefaultLibrary];
    if (!state->library) {
        NSLog(@"Erro: Não foi possível carregar a biblioteca Metal padrão. Verifique se o .metallib foi compilado e está acessível.");
        free(state);
        return NULL;
    }

    id<MTLFunction> kernelFunction = [state->library newFunctionWithName:@"brightness_kernel"];
    if (!kernelFunction) {
        NSLog(@"Erro: Não foi possível encontrar a função do kernel 'brightness_kernel'.");
        free(state);
        return NULL;
    }

    state->pipelineState = [state->device newComputePipelineStateWithFunction:kernelFunction error:&error];
    if (!state->pipelineState) {
        NSLog(@"Erro ao criar o estado do pipeline de computação: %@", error);
        free(state);
        return NULL;
    }

    return state;
}

// Singleton para o estado do Metal
static MetalState* g_metal_state = NULL;

// Função C exportada que será chamada pelo Python via ctypes
__attribute__((visibility("default")))
void apply_brightness_metal(
    float* h_image_in,
    float* h_image_out,
    int width,
    int height,
    int channels,
    float brightness_factor
) {
    // Inicializa o estado do Metal na primeira chamada
    if (g_metal_state == NULL) {
        g_metal_state = initialize_metal_state();
        if (g_metal_state == NULL) {
            fprintf(stderr, "Falha ao inicializar o backend do Metal.\n");
            return;
        }
    }

    @autoreleasepool {
        id<MTLDevice> device = g_metal_state->device;
        id<MTLCommandQueue> commandQueue = g_metal_state->commandQueue;
        id<MTLComputePipelineState> pipelineState = g_metal_state->pipelineState;

        // 1. Criar descritores e texturas
        MTLTextureDescriptor* texDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                           width:width
                                                                                          height:height
                                                                                       mipmapped:NO];
        texDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

        id<MTLTexture> inTexture = [device newTextureWithDescriptor:texDesc];
        id<MTLTexture> outTexture = [device newTextureWithDescriptor:texDesc];

        // 2. Copiar dados do host (CPU) para a textura de entrada
        MTLRegion region = MTLRegionMake2D(0, 0, width, height);
        NSUInteger bytesPerRow = width * channels * sizeof(float);
        [inTexture replaceRegion:region mipmapLevel:0 withBytes:h_image_in bytesPerRow:bytesPerRow];

        // 3. Criar buffer de comando e encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> compute_encoder = [commandBuffer computeCommandEncoder];

        // 4. Configurar o pipeline e os recursos
        [compute_encoder setComputePipelineState:pipelineState];
        [compute_encoder setTexture:inTexture atIndex:0];
        [compute_encoder setTexture:outTexture atIndex:1];
        [compute_encoder setBytes:&brightness_factor length:sizeof(float) atIndex:0];

        // 5. Definir o tamanho da grade de computação
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        MTLSize threadgroupCount = MTLSizeMake(
            (width + threadgroupSize.width - 1) / threadgroupSize.width,
            (height + threadgroupSize.height - 1) / threadgroupSize.height,
            1);
        [compute_encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];

        // 6. Finalizar o encoder e comitar o buffer
        [compute_encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted]; // Bloqueia até a conclusão

        // 7. Copiar dados da textura de saída de volta para o host (CPU)
        [outTexture getBytes:h_image_out bytesPerRow:bytesPerRow fromRegion:region mipmapLevel:0];
    }
}
