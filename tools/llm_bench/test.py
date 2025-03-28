# import time
# from pathlib import Path
# import openvino_genai as ov_genai
# from tqdm import tqdm
# import huggingface_hub as hf_hub

# draft_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# target_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# draft_model_path = Path(draft_model_id.split("/")[-1])
# target_model_path = Path(target_model_id.split("/")[-1])
# if not draft_model_path.exists():
#     hf_hub.snapshot_download(draft_model_id, local_dir=draft_model_path)
# if not target_model_path.exists():
#     hf_hub.snapshot_download(target_model_id, local_dir=target_model_path)
# # draft_model_path = Path("models/llama-2-7b-chat-int4")
# # target_model_path = Path("models/llama-2-7b-chat-int8")
# # draft_model_path = Path("models/meta-llama-3-8b-instruct-int4")
# # target_model_path = Path("models/deepseek-r1-distill-llama-8b")
# target_tokenizer_path = Path("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
# device = "CPU"
# tokenizer = ov_genai.Tokenizer(target_model_path)
# # Load target model for auto-regressive generation
# pipe = ov_genai.LLMPipeline(target_model_path, tokenizer, device)

# config = ov_genai.GenerationConfig()
# config.max_new_tokens = 128

# prompts = ["<s>\n\ndef prime_fib(n: int):"] * 50  # Simplified dataset

# # Auto-Regressive Generation
# print("Running Auto-Regressive generation...")
# times_auto_regressive = []
# for prompt in tqdm(prompts):
#     start_time = time.perf_counter()
#     result = pipe.generate(prompt, config)
#     end_time = time.perf_counter()
#     times_auto_regressive.append(end_time - start_time)
# print("Done")

# # Cleanup before speculative decoding
# del pipe
# import gc
# gc.collect()

# # Speculative Decoding Setup
# scheduler_config = ov_genai.SchedulerConfig()
# scheduler_config.cache_size = 0
# scheduler_config.num_kv_blocks = 2048 // 8
# scheduler_config.max_num_batched_tokens = 2048

# draft_model = ov_genai.draft_model(draft_model_path, device)
# pipe = ov_genai.LLMPipeline(target_model_path, device, draft_model=draft_model, scheduler_config=scheduler_config)

# config.num_assistant_tokens = 5

# times_speculative_decoding = []
# print("Running Speculative Decoding generation...")
# for prompt in tqdm(prompts):
#     start_time = time.perf_counter()
#     result = pipe.generate(prompt, config)
#     end_time = time.perf_counter()
#     times_speculative_decoding.append(end_time - start_time)
# print("Done")

# # Speedup Calculation
# avg_speedup = sum(x / y for x, y in zip(times_auto_regressive, times_speculative_decoding)) / len(prompts)
# print(f"Average speedup: {avg_speedup:.2f}")
from pathlib import Path
import huggingface_hub as hf_hub
import openvino_genai as ov_genai
import time
import gc
from tqdm import tqdm
from datasets import load_dataset

# # Model IDs
# draft_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# target_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# # Download models if needed
# draft_model_path = Path(draft_model_id.split("/")[-1])
# target_model_path = Path(target_model_id.split("/")[-1])
# if not draft_model_path.exists():
#     hf_hub.snapshot_download(draft_model_id, local_dir=draft_model_path)
# if not target_model_path.exists():
#     hf_hub.snapshot_download(target_model_id, local_dir=target_model_path)
draft_model_path = "models/DeepSeek-R1-Distill-Qwen-1.5B"
target_model_path = "models/DeepSeek-R1-Distill-Qwen-7B"
# Set device (CPU or GPU)
device = "CPU"  # Change to "GPU" if available

# Load test dataset
num_samples_to_select = 50
data_type = "Code"  # Change to "Text" for text summarization

print("Loading dataset...")
if data_type == "Code":
    ds = load_dataset("openai_humaneval", split="test")
    prompts = ds["prompt"]
    prompts = ["<s>" + prompts[i] for i in range(num_samples_to_select)]
else:
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
    prompts = ds["article"]
    prompts = [
        "<|user|> ###\nArticle: " + prompts[i] + "\n\nSummarize the above article in 5 sentence.\n<|end|><|assistant|>" 
        for i in range(num_samples_to_select)
    ]
print("Dataset loaded.")

# Benchmark auto-regressive generation
print("Running auto-regressive generation benchmark...")
pipe = ov_genai.LLMPipeline(target_model_path, device)
config = ov_genai.GenerationConfig()
config.max_new_tokens = 330

times_auto_regressive = []
for prompt in tqdm(prompts):
    start_time = time.perf_counter()
    result = pipe.generate(prompt, config)
    end_time = time.perf_counter()
    times_auto_regressive.append(end_time - start_time)

avg_time_auto = sum(times_auto_regressive) / len(times_auto_regressive)
print(f"Auto-regressive average generation time: {avg_time_auto:.2f}s")

# Clean up
del pipe
gc.collect()

# Benchmark speculative decoding
print("Running speculative decoding benchmark...")
scheduler_config = ov_genai.SchedulerConfig()
scheduler_config.cache_size = 0
scheduler_config.num_kv_blocks = 2048 // 8
scheduler_config.max_num_batched_tokens = 2048

draft_model = ov_genai.draft_model(draft_model_path, device)
pipe = ov_genai.LLMPipeline(target_model_path, device, draft_model=draft_model, scheduler_config=scheduler_config)

config = ov_genai.GenerationConfig()
config.max_new_tokens = 330
config.num_assistant_tokens = 5

times_speculative_decoding = []
for prompt in tqdm(prompts):
    start_time = time.perf_counter()
    result = pipe.generate(prompt, config)
    end_time = time.perf_counter()
    times_speculative_decoding.append(end_time - start_time)

avg_time_spec = sum(times_speculative_decoding) / len(times_speculative_decoding)
print(f"Speculative decoding average generation time: {avg_time_spec:.2f}s")

# Calculate speedup
avg_speedup = sum([x / y for x, y in zip(times_auto_regressive, times_speculative_decoding)]) / len(prompts)
print(f"Average speedup: {avg_speedup:.2f}x")