# Z Image Turbo

models/diffusion_models/
    z_image_turbo_bf16.safetensors                       12 GB  (Z-Image Turbo, 6B)

models/text_encoders/
    qwen_3_4b.safetensors                                7.5 GB (for Z-Image)
 

models/vae/
    ae.safetensors                                       320 MB (Z-Image VAE)

# Wan 2.2

models/diffusion_models/
    wan2.2_t2v_high_noise_14B_fp16.safetensors           27 GB  (Wan 2.2 T2V, high-noise half)
    wan2.2_t2v_low_noise_14B_fp16.safetensors            27 GB  (Wan 2.2 T2V, low-noise half)

models/text_encoders/
    umt5_xxl_fp8_e4m3fn_scaled.safetensors               6.3 GB (for Wan 2.2)

models/vae/
    wan_2.1_vae.safetensors                              243 MB (Wan 2.2 VAE)

# LTX 2.3

models/diffusion_models/
    ltx-2.3-22b-dev.safetensors                          43 GB  (LTX-Video 2.3 dev, 22B)

models/text_encoders/
    gemma-3-12b-it-qat-q4_0-unquantized/                 23 GB  (folder, for LTX-2.3)
        - 5 sharded safetensors files
        - tokenizer.json, tokenizer.model, tokenizer_config.json
        - config.json, generation_config.json, model.safetensors.index.json
        - chat_template.json, preprocessor_config.json, processor_config.json
        - special_tokens_map.json, added_tokens.json

models/loras/
    ltx-2.3-22b-distilled-lora-384-1.1.safetensors       7.1 GB (apply to ltx-2.3-22b-dev for fast inference)

models/upscale_models/
    ltx-2.3-spatial-upscaler-x2-1.1.safetensors          950 MB (LTX 2x spatial upscaler)


