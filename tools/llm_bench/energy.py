import pandas as pd
import json

with open("/home/hoang-std/openvino_distributed/openvino.genai/tools/llm_bench/data.json", "r") as file:
    data = json.load(file)

start_time = data["start_time"]
end_time = data["end_time"]

df = pd.read_csv("/home/hoang-std/pcm/build/bin/int8_llama-2-7b-chat-hf.csv")
df = df.drop(index=0).reset_index(drop=True)
filtered_df = df[(df["System.1"] > start_time) & (df["System.1"] < end_time)][["Proc Energy (Joules)", "Proc Energy (Joules).1"]]
filtered_df = filtered_df.apply(pd.to_numeric)
total_sum = filtered_df.sum().sum()
power_consumption = total_sum / (len(filtered_df))
print(f"Total energy: {total_sum}")
print(f"Power consumption: {power_consumption}")