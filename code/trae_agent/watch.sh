watch -n 2 '
OUT=/home/ubuntu/project/agentdiet/code/out/design_space/paper_gpt_5_6_terra_gpt_5_6_luna
echo "logs(done-ish): $(ls "$OUT/log"/*.json 2>/dev/null | wc -l)  containers: $(docker ps -q | wc -l)"
echo
ls -lt "$OUT/output"/task_*.log 2>/dev/null | awk "\$5>0" | head -3
echo
for f in $(ls -t "$OUT/output"/task_*.log 2>/dev/null); do
  [ -s "$f" ] || continue
  line=$(grep -E "Expert\\] turn|PASS|FAIL|NOPATCH|Pulling|validating|Image .* removed" "$f" | tail -1)
  [ -n "$line" ] && echo "$(basename "$f"): $line"
done | head -3
'