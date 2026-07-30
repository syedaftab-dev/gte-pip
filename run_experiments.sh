#!/bin/bash
set -e

# Function to run training and evaluation for a given mode
run_exp() {
    local mode=$1
    echo "=========================================================="
    echo "STARTING EXPERIMENT: $mode"
    echo "=========================================================="
    
    # Run training
    ./venv/bin/python train.py --fusion_mode "$mode"
    
    # Find the latest model directory for this mode
    local latest_log_dir=$(ls -td Log/fusion_${mode}_d128_* 2>/dev/null | head -n 1)
    if [ -z "$latest_log_dir" ]; then
        echo "Error: Could not find log directory for mode $mode"
        exit 1
    fi
    local model_dir="${latest_log_dir}/model"
    
    echo "Training completed. Checkpoints saved in: ${model_dir}"
    echo "Running evaluation..."
    
    # Run testing and save results to the log directory
    ./venv/bin/python test.py --fusion_mode "$mode" --model_dir "$model_dir" > "${latest_log_dir}/evaluation_results.txt" 2>&1
    
    echo "Evaluation completed. Results saved to: ${latest_log_dir}/evaluation_results.txt"
    cat "${latest_log_dir}/evaluation_results.txt"
}

# Wait for the current none baseline process (PID 23211) to finish first!
echo "Waiting for active baseline training (PID 23211) to finish..."
while ps -p 23211 > /dev/null; do
    sleep 30
done

echo "Active baseline training finished. Finding its model directory..."
latest_none_dir=$(ls -td Log/fusion_none_d128_* 2>/dev/null | head -n 1)
if [ -z "$latest_none_dir" ]; then
    echo "Error: Could not find log directory for none baseline"
else
    none_model_dir="${latest_none_dir}/model"
    echo "Running evaluation for none baseline..."
    ./venv/bin/python test.py --fusion_mode none --model_dir "$none_model_dir" > "${latest_none_dir}/evaluation_results.txt" 2>&1
    echo "Evaluation completed. Results saved to: ${latest_none_dir}/evaluation_results.txt"
    cat "${latest_none_dir}/evaluation_results.txt"
fi

# Now run the other three modes sequentially
run_exp "concat"
run_exp "gated"
run_exp "cross_attn"

echo "All experiments completed successfully!"
