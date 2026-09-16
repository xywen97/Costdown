#!/bin/bash

trap 'echo "Interrupted!"; exit 130' SIGINT

while IFS=\| read -r out_name benchmark arg_json
do
  echo "=== name=$out_name benchmark=$benchmark $arg_json"

  if [[ "$out_name" = "" ]]; then
    echo NOTHING.
    continue
  fi

  if [[ "$out_name" =~ ^# ]]; then
    echo SKIP.
    continue
  fi

  TRAJ_ANALYSIS="$arg_json" \
  python3 swebench_main.py \
    --benchmark "$benchmark" \
    --log_path "../out/$out_name/log" \
    --patches_path "../out/$out_name/patch" \
    --output_path "../out/$out_name/output"

done < main_args-arbiteros_hybrid.txt

echo "=== FINISHED"
date