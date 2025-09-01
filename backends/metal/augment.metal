#include <metal_stdlib>

using namespace metal;

/*
 * Metal Shading Language (MSL) Kernel para ajustar o brilho de uma imagem.
 *
 * Parâmetros:
 * - in_texture [[texture(0)]]: A textura de entrada da qual ler os pixels.
 * - out_texture [[texture(1)]]: A textura de saída na qual escrever os pixels processados.
 * - brightness_factor [[buffer(0)]]: Um buffer contendo o fator de brilho a ser aplicado.
 * - gid [[thread_position_in_grid]]: A coordenada 2D do thread atual na grade de computação,
 *   que corresponde à coordenada do pixel (x, y).
 */
kernel void brightness_kernel(
    texture2d<float, access::read> in_texture [[texture(0)]],
    texture2d<float, access::write> out_texture [[texture(1)]],
    constant float &brightness_factor [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Garante que o thread não está acessando uma coordenada fora da textura
    if (gid.x >= out_texture.get_width() || gid.y >= out_texture.get_height()) {
        return;
    }

    // Lê a cor do pixel da textura de entrada na coordenada atual
    float4 original_color = in_texture.read(gid);

    // Aplica o fator de brilho a cada canal de cor (R, G, B)
    // O canal alpha (A) é mantido inalterado
    float3 brightened_rgb = original_color.rgb + brightness_factor;

    // Garante que os novos valores de cor permaneçam no intervalo [0, 1] (clamping)
    brightened_rgb = clamp(brightened_rgb, float3(0.0), float3(1.0));

    // Monta a nova cor final
    float4 final_color = float4(brightened_rgb, original_color.a);

    // Escreve a cor final na textura de saída na mesma coordenada
    out_texture.write(final_color, gid);
}
