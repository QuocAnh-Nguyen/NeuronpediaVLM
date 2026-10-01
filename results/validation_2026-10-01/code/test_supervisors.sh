#!/bin/sh
# Regression test for the shipped supervisors (run_campaign.sh, run_x6_retry.sh,
# reclaim_and_launch.sh) with stubbed externals: no GPU, no python, no real fits.
# Run on the server (reads the deployed code dir):  bash results/validation_2026-10-01/code/test_supervisors.sh
# Each scenario gets its own fake $HOME so the scripts' $REPO/$RESULTS/$CODE resolve to it.
set -u
REAL=/home/nvidia-lab/ai4life/phuongnh/vlm-lens/results/validation_2026-10-01/code
T=/tmp/svtest
rm -rf $T; mkdir -p $T/bin

printf '#!/bin/sh\nexit 0\n' > $T/bin/sleep
printf '#!/bin/sh\nif [ -s %s/gpu ]; then l=$(sed -n 1p %s/gpu); sed -i 1d %s/gpu; echo "$l"; else echo 60000; fi\n' "$T" "$T" "$T" > $T/bin/nvidia-smi
chmod +x $T/bin/sleep $T/bin/nvidia-smi

mkhome() { # $1 = scenario dir; sets H, CODE, LOGS
    H=$1/home
    CODE=$H/ai4life/phuongnh/vlm-lens/results/validation_2026-10-01/code
    LOGS=$H/ai4life/phuongnh/vlm-lens/results/validation_2026-10-01/logs
    mkdir -p "$CODE" "$LOGS" "$H/miniconda3/envs/vlm_truth_py313/bin"
    printf '#!/bin/sh\nexit 0\n' > "$H/miniconda3/envs/vlm_truth_py313/bin/python"
    chmod +x "$H/miniconda3/envs/vlm_truth_py313/bin/python"
}

cat > $T/stub.sh <<'EOS'
#!/bin/sh
M="$1"; C="$2"
n=$(cat "$C" 2>/dev/null || echo 0); n=$((n+1)); echo "$n" > "$C"
line=$(sed -n "${n}p" "$M"); [ -z "$line" ] && line="0"
rc=$(echo "$line" | cut -d: -f1); marker=$(echo "$line" | cut -d: -f2 -s)
echo "stub step run $n rc=$rc marker=$marker"
case "$marker" in
  OOM)  echo "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 222.00 MiB";;
  KILL) echo "$0: line 40: 12345 Killed  python scripts/fit_llava.py";;
  CUDA) echo "RuntimeError: CUDA error: an illegal memory access was encountered";;
esac
exit "$rc"
EOS
chmod +x $T/stub.sh

PASS=0; FAIL=0
check() { got=$(grep -c -- "$3" "$2" 2>/dev/null || true)
    if [ "${got:-0}" = "$4" ]; then echo "PASS: $1 (found $got)"; PASS=$((PASS+1));
    else echo "FAIL: $1 (want $4 of '$3', got ${got:-0})"; FAIL=$((FAIL+1)); fi; }
check_rc() { if [ "$2" = "$3" ]; then echo "PASS: $1 (rc=$2)"; PASS=$((PASS+1));
    else echo "FAIL: $1 (want rc=$3, got $2)"; FAIL=$((FAIL+1)); fi; }

# ============ A) x6 retry: skip,skip,OOM,KILL,success ==========================
A=$T/A; mkhome $A; echo 60000 > $T/gpu
cp $REAL/run_x6_retry.sh $A/run_x6_retry.sh
printf '3\n3\n1:OOM\n1:KILL\n0\n' > $A/modes
printf '#!/bin/sh\nexec %s/stub.sh %s/modes %s/count\n' "$T" "$A" "$A" > $CODE/run_x6_fp32.sh
chmod +x $CODE/run_x6_fp32.sh
HOME=$A/home PATH=$T/bin:$PATH RESULTS=$A/home/ai4life/phuongnh/vlm-lens/results/validation_2026-10-01 \
    bash $A/run_x6_retry.sh > $A/out.txt 2>&1
check_rc "A exit 0 after recovery" $? 0
check "A two window skips"     $A/out.txt "no window" 2
check "A OOM classified"       $A/out.txt "reason=OOM" 1
check "A SIGKILL classified"   $A/out.txt "reason=sigkill" 1
check "A success on attempt 5" $A/out.txt "leg OK on attempt 5" 1
check "A stub ran 5x"          $A/count "5" 1

# ============ B) x6 retry: 3 consecutive 'other' -> give up ====================
B=$T/B; mkhome $B; echo 60000 > $T/gpu
cp $REAL/run_x6_retry.sh $B/run_x6_retry.sh
printf '1:x\n1:x\n1:x\n0\n' > $B/modes
printf '#!/bin/sh\nexec %s/stub.sh %s/modes %s/count\n' "$T" "$B" "$B" > $CODE/run_x6_fp32.sh
chmod +x $CODE/run_x6_fp32.sh
HOME=$B/home PATH=$T/bin:$PATH RESULTS=$B/home/ai4life/phuongnh/vlm-lens/results/validation_2026-10-01 \
    bash $B/run_x6_retry.sh > $B/out.txt 2>&1
check_rc "B exit 1 on give-up" $? 1
check "B other x3 then stop"   $B/out.txt "reason=other" 3
check "B give-up message"      $B/out.txt "3 consecutive non-transient failures" 1
check "B stub ran 3x"          $B/count "3" 1

# ============ C) campaign: s1 KILL,KILL,ok -> s2/x1/x7x9 ok -> done ============
C=$T/C; mkhome $C; echo 60000 > $T/gpu
cp $REAL/run_campaign.sh $C/run_campaign.sh
printf '1:KILL\n1:KILL\n0\n' > $C/s1modes; printf '0\n' > $C/okmodes
printf '#!/bin/sh\nexec %s/stub.sh %s/s1modes %s/s1count\n' "$T" "$C" "$C" > $CODE/run_step2.sh
for s in run_step3.sh run_step4.sh; do printf '#!/bin/sh\nexec %s/stub.sh %s/okmodes %s/%s\n' "$T" "$C" "$C" "$s" > $CODE/$s; done
chmod +x $CODE/run_step2.sh $CODE/run_step3.sh $CODE/run_step4.sh
HOME=$C/home PATH=$T/bin:$PATH bash $C/run_campaign.sh > $C/out.txt 2>&1
check_rc "C exit 0" $? 0
check "C s1 sigkill classified" $C/out.txt "reason=sigkill" 2
check "C s1 ok on attempt 3"    $C/out.txt "ok (attempt 3)" 1
check "C s2 ran"                $C/out.txt "\[s2\] ok" 1
check "C x1 ran"                $C/out.txt "\[x1\] ok" 1
check "C x7x9 ran"              $C/out.txt "\[x7x9\] ok" 1
check "C campaign done"         $C/out.txt "campaign done" 1
check "C s1 stub ran 3x"        $C/s1count "3" 1

# ============ D) campaign: s1 'other' x3 -> CAMPAIGN_S1_FAILED, exit 1 ========
D=$T/D; mkhome $D; echo 60000 > $T/gpu
cp $REAL/run_campaign.sh $D/run_campaign.sh
printf '1:x\n1:x\n1:x\n0\n' > $D/s1modes
printf '#!/bin/sh\nexec %s/stub.sh %s/s1modes %s/s1count\n' "$T" "$D" "$D" > $CODE/run_step2.sh
chmod +x $CODE/run_step2.sh
HOME=$D/home PATH=$T/bin:$PATH bash $D/run_campaign.sh > $D/out.txt 2>&1
check_rc "D exit 1" $? 1
check "D other classified x3"      $D/out.txt "reason=other" 3
check "D give-up on 3 non-transient" $D/out.txt "giving up: 3 consecutive non-transient failures" 1
check "D CAMPAIGN_S1_FAILED"       $D/out.txt "CAMPAIGN_S1_FAILED" 1
check "D s1 stub ran 3x"           $D/s1count "3" 1

# ============ E) campaign: low free-memory window satisfied on later poll ======
E=$T/E; mkhome $E
cp $REAL/run_campaign.sh $E/run_campaign.sh
printf '5000\n2000\n99999\n' > $T/gpu
printf '0\n' > $E/okmodes
printf '#!/bin/sh\nexec %s/stub.sh %s/okmodes %s/c1\n' "$T" "$E" "$E" > $CODE/run_step2.sh
printf '#!/bin/sh\nexec %s/stub.sh %s/okmodes %s/c2\n' "$T" "$E" "$E" > $CODE/run_step3.sh
printf '#!/bin/sh\nexec %s/stub.sh %s/okmodes %s/c3\n' "$T" "$E" "$E" > $CODE/run_step4.sh
chmod +x $CODE/run_step2.sh $CODE/run_step3.sh $CODE/run_step4.sh
HOME=$E/home PATH=$T/bin:$PATH bash $E/run_campaign.sh > $E/out.txt 2>&1
check_rc "E exit 0" $? 0
check "E waited for the window" $E/out.txt "free=99999MiB >= 28000MiB - starting" 1
check "E campaign done"         $E/out.txt "campaign done" 1

# ============ F) campaign: 5 SIGKILLs then success (transient cap removed) =====
F=$T/F; mkhome $F; echo 60000 > $T/gpu
cp $REAL/run_campaign.sh $F/run_campaign.sh
printf '1:KILL\n1:KILL\n1:KILL\n1:KILL\n1:KILL\n0\n' > $F/s1modes
printf '#!/bin/sh\nexec %s/stub.sh %s/s1modes %s/s1count\n' "$T" "$F" "$F" > $CODE/run_step2.sh
printf '#!/bin/sh\nexec %s/stub.sh %s/s1modes %s/scount\n' "$T" "$F" "$F" > $CODE/run_step3.sh
printf '#!/bin/sh\nexec %s/stub.sh %s/s1modes %s/xcount\n' "$T" "$F" "$F" > $CODE/run_step4.sh
chmod +x $CODE/run_step2.sh $CODE/run_step3.sh $CODE/run_step4.sh
HOME=$F/home PATH=$T/bin:$PATH bash $F/run_campaign.sh > $F/out.txt 2>&1
check_rc "F exit 0 after 5 kills" $? 0
check "F s1 ok on attempt 6"      $F/out.txt "\[s1\] ok (attempt 6)" 1
check "F no give-up"              $F/out.txt "giving up" 0
check "F campaign done"           $F/out.txt "campaign done" 1

# ============ G) reclaim guard: refuse with a live supervisor, FORCE kills =====
G=$T/G; mkhome $G
cp $REAL/reclaim_and_launch.sh $G/reclaim.sh
printf '#!/bin/sh\nexit 0\n' > $CODE/run_campaign.sh; chmod +x $CODE/run_campaign.sh
mkdir -p $T/fake; cp /bin/sleep $T/fake/run_campaign.sh
$T/fake/run_campaign.sh 25 &
FAKE=$!
sleep 0.3
PAT='fit_llav[a].py|run_res[t].sh|run_ste[p][0-9].sh|run_campaign[.]sh|s1_scor[e].py|s2_ev[a]l.py|split_halve[s].py|x1_comp[a]re.py|x6_comp[a]re.py|x7_x9_interve[n]tions.py'
others=$(pgrep -f "$PAT" 2>/dev/null | grep -v "^$FAKE$" | wc -l)
echo "G: other PAT matches besides the fake: $others"
HOME=$G/home PATH=$T/bin:$PATH bash $G/reclaim.sh > $G/refuse.txt 2>&1
check_rc "G refuses without FORCE" $? 1
check "G refusal message" $G/refuse.txt "refusing to reclaim" 1
if kill -0 $FAKE 2>/dev/null; then echo "PASS: G live supervisor untouched"; PASS=$((PASS+1));
else echo "FAIL: G fake supervisor was killed despite refusal"; FAIL=$((FAIL+1)); fi
if [ "$others" = "0" ]; then
    HOME=$G/home PATH=$T/bin:$PATH FORCE=1 bash $G/reclaim.sh > $G/force.txt 2>&1
    check_rc "G FORCE exits 0" $? 0
    check "G FORCE killed leftovers" $G/force.txt "leftovers to kill" 1
    sleep 0.2
    if kill -0 $FAKE 2>/dev/null; then echo "FAIL: G FORCE did not kill the fake supervisor"; FAIL=$((FAIL+1));
    else echo "PASS: G FORCE killed the fake supervisor"; PASS=$((PASS+1)); fi
else
    echo "SKIP: G FORCE path (unexpected PAT matches on the box: $others)"
fi
kill -9 $FAKE 2>/dev/null

echo
echo "=== SUMMARY: PASS=$PASS FAIL=$FAIL ==="
