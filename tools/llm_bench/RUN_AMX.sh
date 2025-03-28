unset ONEDNN_MAX_CPU_ISA
unset ONEDNN_VERBOSE
unset OPENVINO_CPU_OPTIMIZE_FOR

export ONEDNN_MAX_CPU_ISA=AVX10_1_512_AMX
# export ONEDNN_VERBOSE=1
# export OPENVINO_CPU_OPTIMIZE_FOR=AMX



python benchmark.py -m models/llama-2-7b-chat-int8/ -n 1 --load_config config.json -ic 128 --input_length 128 -bs 1