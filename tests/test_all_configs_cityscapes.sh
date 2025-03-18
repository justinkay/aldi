#!/usr/bin/env bash

# clear the output file
OUTPUT_FILE="output.txt"
> "$OUTPUT_FILE"

# get all cityscapes configs
find configs/cityscapes/ -type f -name '*.yaml' | while read -r CONFIG; do
  
  echo "Processing: $CONFIG" >> "$OUTPUT_FILE"
  
  # download checkpoint if applicable
  python tools/download_model_for_config.py --config-file "$CONFIG" \
    >> "$OUTPUT_FILE" 2> temp.err
  if [ $? -ne 0 ]; then
    echo "Error encountered while downloading model for: $CONFIG"
    cat temp.err
    continue
  fi
  
  # smoke test for train_net
  timeout 15s python tools/train_net.py --config "$CONFIG" SOLVER.MAX_ITER 1 \
    >> "$OUTPUT_FILE" 2> temp.err
  RET_CODE=$?
  
  if [ $RET_CODE -eq 124 ]; then
    # 124 is the exit status for a `timeout`-killed process
    echo "Training command timed out after 15 seconds for config: $CONFIG"
    cat temp.err
  elif [ $RET_CODE -ne 0 ]; then
    # Some other error
    echo "Error encountered while training with config: $CONFIG"
    cat temp.err
  fi
  
  # clean up
  rm -f temp.err

done

echo "All done! Logs can be found in '$OUTPUT_FILE'."
