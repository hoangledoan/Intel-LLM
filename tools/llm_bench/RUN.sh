# # unset ONEDNN_MAX_CPU_ISA
# # unset ONEDNN_VERBOSE
# # unset OPENVINO_CPU_OPTIMIZE_FOR
export ONEDNN_MAX_CPU_ISA=AVX512_CORE_VNNI
# # export ONEDNN_VERBOSE=1

# Function to run benchmark and energy measurement with separate logs
run_benchmark() {
  local batch_size=$1
  local log_file="log_no_amx_bs${batch_size}_256.txt"
  
  echo "Running benchmark with batch size $batch_size, logging to $log_file..."
  
  # Run benchmark and redirect output to the log file
  python benchmark.py -m models/llama-3-8b-int8/ -n 3 --load_config config.json -ic 128 --input_length 256 -bs $batch_size > $log_file 2>&1
  sleep 10
  # Run energy measurement and append output to the same log file
  python energy.py >> $log_file 2>&1
  
  echo "Completed run with batch size $batch_size"
}

run_benchmark_128() {
  local batch_size=$1
  local log_file="log_no_amx_bs${batch_size}_128.txt"
  
  echo "Running benchmark with batch size $batch_size, logging to $log_file..."
  
  # Run benchmark and redirect output to the log file
  python benchmark.py -m models/llama-3-8b-int8/ -n 3 --load_config config.json -ic 128 --input_length 128 -bs $batch_size > $log_file 2>&1
  sleep 10
  # Run energy measurement and append output to the same log file
  python energy.py >> $log_file 2>&1
  
  echo "Completed run with batch size $batch_size"
}

run_benchmark_int4_128() {
  local batch_size=$1
  local log_file="log_no_amx_int4_bs${batch_size}_128.txt"
  
  echo "Running benchmark with batch size $batch_size, logging to $log_file..."
  
  # Run benchmark and redirect output to the log file
  python benchmark.py -m models/llama-3-8b-int4/ -n 3 --load_config config.json -ic 128 --input_length 128 -bs $batch_size > $log_file 2>&1
  sleep 10
  # Run energy measurement and append output to the same log file
  python energy.py >> $log_file 2>&1
  
  echo "Completed run with batch size $batch_size"
}

run_benchmark_int4_256() {
  local batch_size=$1
  local log_file="log_no_amx_int4_bs${batch_size}_256.txt"
  
  echo "Running benchmark with batch size $batch_size, logging to $log_file..."
  
  # Run benchmark and redirect output to the log file
  python benchmark.py -m models/llama-3-8b-int4/ -n 3 --load_config config.json -ic 128 --input_length 256 -bs $batch_size > $log_file 2>&1
  sleep 10
  # Run energy measurement and append output to the same log file
  python energy.py >> $log_file 2>&1
  
  echo "Completed run with batch size $batch_size"
}

# run_benchmark_int4_512() {
#   local batch_size=$1
#   local log_file="log_no_amx_int4_bs${batch_size}_512.txt"
  
#   echo "Running benchmark with batch size $batch_size, logging to $log_file..."
  
#   # Run benchmark and redirect output to the log file
#   python benchmark.py -m models/llama-2-7b-chat-int4/ -n 3 --load_config config.json -ic 128 --input_length 512 -bs $batch_size > $log_file 2>&1
#   sleep 10
#   # Run energy measurement and append output to the same log file
#   python energy.py >> $log_file 2>&1
  
#   echo "Completed run with batch size $batch_size"
# }

Run for different batch sizes
run_benchmark 1
run_benchmark 8
run_benchmark 64
run_benchmark 256
run_benchmark 512
# run_benchmark 1024

run_benchmark_128 1
run_benchmark_128 8
run_benchmark_128 64
run_benchmark_128 256
run_benchmark_128 512
# run_benchmark_128 1024

run_benchmark_int4_128 1
run_benchmark_int4_128 8
run_benchmark_int4_128 64
run_benchmark_int4_128 256
run_benchmark_int4_128 512
# run_benchmark_int4_128 1024

run_benchmark_int4_256 1
run_benchmark_int4_256 8
run_benchmark_int4_256 64
run_benchmark_int4_256 256
run_benchmark_int4_256 512
# run_benchmark_int4_256 1024

# run_benchmark_int4_512 1
# run_benchmark_int4_512 8
# run_benchmark_int4_512 64
# run_benchmark_int4_512 256
# run_benchmark_int4_512 512
# run_benchmark_int4_512 1024
# echo "All benchmark runs completed!"



export ONEDNN_MAX_CPU_ISA=ONEDNN_MAX_CPU_ISA=AVX10_1_512_AMX

# Function to run benchmark and energy measurement with separate logs
run_benchmark_amx() {
  local batch_size=$1
  local log_file="log_amx_bs${batch_size}_256.txt"
  
  echo "Running benchmark with batch size $batch_size, logging to $log_file..."
  
  # Run benchmark and redirect output to the log file
  python benchmark.py -m models/llama-3-8b-int8/ -n 3 --load_config config.json -ic 128 --input_length 256 -bs $batch_size > $log_file 2>&1
  sleep 10
  # Run energy measurement and append output to the same log file
  python energy.py >> $log_file 2>&1
  
  echo "Completed run with batch size $batch_size"
}

run_benchmark_128_amx() {
  local batch_size=$1
  local log_file="log_amx_bs${batch_size}_128.txt"
  
  echo "Running benchmark with batch size $batch_size, logging to $log_file..."
  
  # Run benchmark and redirect output to the log file
  python benchmark.py -m models/llama-3-8b-int8/ -n 3 --load_config config.json -ic 128 --input_length 128 -bs $batch_size > $log_file 2>&1
  sleep 10
  # Run energy measurement and append output to the same log file
  python energy.py >> $log_file 2>&1
  
  echo "Completed run with batch size $batch_size"
}


# Run for different batch sizes
run_benchmark_amx 1
run_benchmark_amx 8
run_benchmark_amx 64
run_benchmark_amx 256
run_benchmark_amx 512
# run_benchmark_amx 1024

run_benchmark_128_amx 1
run_benchmark_128_amx 8
run_benchmark_128_amx 64
run_benchmark_128_amx 256
run_benchmark_128_amx 512
# run_benchmark_128_amx 1024


run_benchmark_int4_128_amx() {
  local batch_size=$1
  local log_file="log_amx_int4_bs${batch_size}_128.txt"
  
  echo "Running benchmark with batch size $batch_size, logging to $log_file..."
  
  # Run benchmark and redirect output to the log file
  python benchmark.py -m models/llama-3-8b-int4/ -n 3 --load_config config.json -ic 128 --input_length 128 -bs $batch_size > $log_file 2>&1
  sleep 10
  # Run energy measurement and append output to the same log file
  python energy.py >> $log_file 2>&1
  
  echo "Completed run with batch size $batch_size"
}

run_benchmark_int4_256_amx() {
  local batch_size=$1
  local log_file="log_amx_int4_bs${batch_size}_256.txt"
  
  echo "Running benchmark with batch size $batch_size, logging to $log_file..."
  
  # Run benchmark and redirect output to the log file
  python benchmark.py -m models/llama-3-8b-int4/ -n 3 --load_config config.json -ic 128 --input_length 256 -bs $batch_size > $log_file 2>&1
  sleep 10
  # Run energy measurement and append output to the same log file
  python energy.py >> $log_file 2>&1
  
  echo "Completed run with batch size $batch_size"
}

run_benchmark_int4_512_amx() {
  local batch_size=$1
  local log_file="log_amx_int4_bs${batch_size}_512.txt"
  
  echo "Running benchmark with batch size $batch_size, logging to $log_file..."
  
  # Run benchmark and redirect output to the log file
  python benchmark.py -m models/llama-2-7b-chat-int4/ -n 3 --load_config config.json -ic 128 --input_length 512 -bs $batch_size > $log_file 2>&1
  sleep 10
  # Run energy measurement and append output to the same log file
  python energy.py >> $log_file 2>&1
  
  echo "Completed run with batch size $batch_size"
}

Run INT4 benchmarks with AMX
run_benchmark_int4_128_amx 1
run_benchmark_int4_128_amx 8
run_benchmark_int4_128_amx 64
run_benchmark_int4_128_amx 256
run_benchmark_int4_128_amx 512
# run_benchmark_int4_128_amx 1024

run_benchmark_int4_256_amx 1
run_benchmark_int4_256_amx 8
run_benchmark_int4_256_amx 64
run_benchmark_int4_256_amx 256
run_benchmark_int4_256_amx 512
# run_benchmark_int4_256_amx 1024

# run_benchmark_int4_512_amx 1
# run_benchmark_int4_512_amx 8
# run_benchmark_int4_512_amx 64
# run_benchmark_int4_512_amx 256
# run_benchmark_int4_512_amx 512
# run_benchmark_int4_512_amx 1024

echo "All benchmark runs completed!"