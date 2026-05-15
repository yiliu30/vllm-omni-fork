MODEL=/mnt/disk1/yiliu7/Qwen/Qwen-Image/
MODEL=/media/hf_models/Qwen/Qwen-Image
# MODEL=/mnt/disk1/yiliu7/Yi30/Qwen-Image-W4A16
# MODEL=/mnt/disk1/yiliu7/Qwen/Qwen-Image-W4A16-SKIP-IMG-MOD-TXT-MOD
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.29.7


    
# CUDA_VISIBLE_DEVICES=6,7 python examples/offline_inference/text_to_image/text_to_image.py \
#     --model $MODEL \
#     --tensor-parallel-size 2 \
#     --cfg-parallel-size 1 \
#     --prompt "a cup of coffee on the table" \
#     --seed 42 --num-inference-steps 50 \
#     --height 1024 --width 1024 \
#     --guidance-scale 3.5 --cfg-scale 4.0 \
#     --output out/qwen_w4a16.png

# VLLM_MXFP4_USE_MARLIN=1 \
source /home/yiliu7/workspace/venvs/omni/bin/activate
SAGE3_QUANT_FORMAT=nvfp4 \
DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN \
CUDA_VISIBLE_DEVICES=2,3 python examples/offline_inference/text_to_image/text_to_image.py \
  --model $MODEL \
  --tensor-parallel-size 1 \
  --width 256 --height 256 \
  --num-inference-steps 10 \
    --enforce-eager \
  --prompt "a cup of coffee on the table" \
  --output ar_coffee_mxfp4_2nd.png 2>&1 | tee gen_coffee_mxfp4.log
#   --enable-cpu-offload \

    # --quantization mxfp4 \


# Loading requests...
# Prepared 5 requests from vbench dataset.
#   0%|             | 0/5 [00:00<?, ?it/s]Running 1 warmup request(s) with num_inference_steps=1 and warmup_concurrency=1...
# 100%|█████| 5/5 [02:04<00:00, 24.93s/it]

# ================= Serving Benchmark Result =================
# Backend:                                 vllm-omni      
# Model:                                   /mnt/disk1/yiliu7/Qwen/Qwen-Image/
# Dataset:                                 vbench         
# Task:                                    t2i            
# --------------------------------------------------
# Benchmark duration (s):                  120.37         
# Request rate:                            inf            
# Max request concurrency:                 1              
# Successful requests:                     5/5              
# --------------------------------------------------
# Request throughput (req/s):              0.04           
# Latency Mean (s):                                                        .0741        
# Latency Median (s):                      24.1517        
# Latency P99 (s):                         24.2475        
# Latency P95 (s):                         24.2343        
# --------------------------------------------------
# Peak Memory Max (MB):                    30710.00       
# Peak Memory Mean (MB):                   30710.00       
# Peak Memory Median (MB):                 30710.00       
# --------------------------------------------------
# Stage Durations Mean (s):
#   queue_wait_ms:                         0.4429         
#   stage_0_gen_ms:                        23691.3612     

# ============================================================

# Loading requests...
# Prepared 5 requests from vbench dataset.
#   0%|             | 0/5 [00:00<?, ?it/s]Running 1 warmup request(s) with num_inference_steps=1 and warmup_concurrency=1...
# 100%|█████| 5/5 [02:00<00:00, 24.08s/it]

# ================= Serving Benchmark Result =================
# Backend:                                 vllm-omni      
# Model:                                   /mnt/disk1/yiliu7/Yi30/Qwen-Image-W4A16/
# Dataset:                                 vbench         
# Task:                                    t2i            
# --------------------------------------------------
# Benchmark duration (s):                  113.31         
# Request rate:                            inf            
# Max request concurrency:                 1              
# Successful requests:                     5/5              
# --------------------------------------------------
# Request throughput (req/s):              0.04           
# Latency Mean (s):                        22.6625        
# Latency Median (s):                      22.6894        
# Latency P99 (s):                         22.7198        
# Latency P95 (s):                         22.7156        
# --------------------------------------------------
# Peak Memory Max (MB):                    17680.00       
# Peak Memory Mean (MB):                   17680.00       
# Peak Memory Median (MB):                 17680.00       
# --------------------------------------------------
# Stage Durations Mean (s):
#   queue_wait_ms:                         0.2086         
#   stage_0_gen_ms:                        22283.2224     

# ============================================================

# Loading requests...
# Prepared 5 requests from vbench dataset.
#   0%|             | 0/5 [00:00<?, ?it/s]Running 1 warmup request(s) with num_inference_steps=1 and warmup_concurrency=1...
# 100%|█████| 5/5 [01:50<00:00, 22.08s/it]

# ================= Serving Benchmark Result =================
# Backend:                                 vllm-omni      
# Model:                                   /mnt/disk1/yiliu7/Yi30/Qwen-Image-W4A16/
# Dataset:                                 vbench         
# Task:                                    t2i            
# --------------------------------------------------
# Benchmark duration (s):                  103.56         
# Request rate:                            inf            
# Max request concurrency:                 1              
# Successful requests:                     5/5              
# --------------------------------------------------
# Request throughput (req/s):              0.05           
# Latency Mean (s):                        20.7124        
# Latency Median (s):                      20.7311        
# Latency P99 (s):                         20.9843        
# Latency P95 (s):                         20.9444        
# --------------------------------------------------
# Peak Memory Max (MB):                    20800.00       
# Peak Memory Mean (MB):                   20483.20       
# Peak Memory Median (MB):                 20404.00       
# --------------------------------------------------
# Stage Durations Mean (s):
#   queue_wait_ms:                         0.2142         
#   stage_0_gen_ms:                        20331.2406     

# ============================================================