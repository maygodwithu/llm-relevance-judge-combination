#!/bin/bash
# Independent calibration for ESCI Llama-3.1-8B T1-T7 (scoring only: no training, no PO, no optimizer step).
# Scores the FROZEN independent calibration pool of the earlier T2 check (200 query / 3,939 rows; rows file SHA-256 checked below) with the stored final (625-update) checkpoints:
#   original text T{k}: Base + CE+soft kappa seeds 0-2 for T1,T3,T4,T5,T6,T7 (T2: the 8 score files of run393.sh, which must have been run before analyze410.py);  selected PO text T{k}_APOCF: T3,T5,T7 (T2 reused; T1,T4,T6 keep the original text, so their original scores are reused).
# New scoring jobs: 36. Queue = 18 calls of 2 jobs (Base first inside a call that contains Base), two workers (one per physical GPU 0 / 1), claim by mkdir. A failing gate / call / truncation check stops the queue (no retry).
# Stage 0 gate (fixed before the run): the same evaluator re-scores the first 20 stored selection queries (593 rows) for five jobs; margins must equal the stored ones (max |diff| <= 1e-4).
#   G0=<uuid> G1=<uuid> setsid nohup bash run410.sh > run410_stdout.txt 2>&1 < /dev/null &      (from the project root; device ids are never written to a log)
set -u
PY=${PY:-python}
export HF_HUB_OFFLINE=1
OUT=cpmr_experiment/esci410
CAL=cpmr_experiment/esci/audit/calib_rows.jsonl      # restored by scripts/restore_data.py esci
CAL_SHA=bcfe73aa09cdd2ed20ec88567c270ba720185699bed085d35da3ecf3e2b3a928
GATE_ROWS=$OUT/gate/sel_first20_rows.jsonl
mkdir -p $OUT/logs $OUT/eval $OUT/claims $OUT/gate $OUT/ckpt_maps
log () { echo "$(date -u +%FT%T) $*" >> $OUT/scheduler.log; }
man () { echo cpmr_experiment/esci_llama/p1_T$1/esci_apocf_manifest.json; }
log "instance start"
[ "$(sha256sum $CAL | cut -d' ' -f1)" = "$CAL_SHA" ] || { echo "calibration pool hash mismatch" > $OUT/ABORT.txt; log "ABORT pool hash"; exit 1; }
$PY - <<PYEOF
import json
rows = [json.loads(l) for l in open("cpmr_experiment/esci/audit/sel_rows.jsonl", encoding="utf-8")]
q = []
for r in rows:
    if r["nq"] not in q: q.append(r["nq"])
keep = set(q[:20]); out = [r for r in rows if r["nq"] in keep]
open("$GATE_ROWS", "w", encoding="utf-8", newline="\n").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
PYEOF
[ "$(wc -l < $GATE_ROWS)" = "593" ] || { echo "gate rows not 593" > $OUT/ABORT.txt; log "ABORT gate rows"; exit 1; }
# one checkpoint map for every job name used here (checkpoints written by experiments/commands/llama_esci.sh; paths checked to exist)
CK=${CK:-cpmr_experiment/esci_llama/train/checkpoints}
$PY - <<PYEOF
import json, os
m = {}
for T in (1, 3, 4, 5, 6, 7):
    for s in range(3):
        m[f"ELQ_mix_T{T}_s{s}_step625"] = f"$CK/ELQ_mix_T{T}_s{s}/ckpt_step625.pt"
for T in (3, 5, 7):
    for s in range(3):
        m[f"ELQ_mix_T{T}_APOCF_s{s}_step625"] = f"$CK/ELQ_mix_T{T}_APOCF_s{s}/ckpt_step625.pt"
missing = [v for v in m.values() if not os.path.exists(v)]
assert not missing, missing
json.dump(m, open("$OUT/ckpt_maps/map_all.json", "w"), indent=1); print(len(m), "checkpoints mapped")
PYEOF
[ $? -eq 0 ] || { echo "ckpt map failed" > $OUT/ABORT.txt; exit 1; }
MAP=$OUT/ckpt_maps/map_all.json
# --- stage 0 gate
log "gate start"
( CUDA_VISIBLE_DEVICES=$G0 $PY -u methods/esci_llama_eval.py --jobs Base:T5 Base:T5_APOCF ELQ_mix_T5_APOCF_s0_step625:T5_APOCF --instr_file $(man 5) --rows $GATE_ROWS --tag gate --ckpt_map $MAP --out_dir $OUT/gate --local_files_only > $OUT/logs/gate_T5.out 2>&1; echo "g5 rc=$?" >> $OUT/gate/status.txt ) &
( CUDA_VISIBLE_DEVICES=$G1 $PY -u methods/esci_llama_eval.py --jobs ELQ_mix_T3_APOCF_s0_step625:T3_APOCF --instr_file $(man 3) --rows $GATE_ROWS --tag gate --ckpt_map $MAP --out_dir $OUT/gate --local_files_only > $OUT/logs/gate_T3.out 2>&1; echo "g3 rc=$?" >> $OUT/gate/status.txt
  CUDA_VISIBLE_DEVICES=$G1 $PY -u methods/esci_llama_eval.py --jobs ELQ_mix_T7_s0_step625:T7 --instr_file $(man 7) --rows $GATE_ROWS --tag gate --ckpt_map $MAP --out_dir $OUT/gate --local_files_only > $OUT/logs/gate_T7.out 2>&1; echo "g7 rc=$?" >> $OUT/gate/status.txt ) &
wait
log "gate eval done: $(tr '\n' ' ' < $OUT/gate/status.txt)"
grep -q "rc=[1-9]" $OUT/gate/status.txt && { echo "gate eval failed" > $OUT/ABORT.txt; exit 1; }
$PY cpmr_experiment/esci410/gate_compare410.py || { echo "gate comparison failed" > $OUT/ABORT.txt; log "ABORT gate comparison"; exit 1; }
log "gate passed"
# --- stage 1 tasks: <var>_T<k>_<a|b>   a = [Base, seed 0], b = [seed 1, seed 2];  var o = original text, p = selected PO text (T3, T5, T7)
TASKS=""
for T in 1 3 4 5 6 7; do for part in a b; do TASKS="$TASKS o_T${T}_$part"; done; done
for T in 3 5 7; do for part in a b; do TASKS="$TASKS p_T${T}_$part"; done; done
run_task () {   # $1 task, $2 gpu, $3 worker label
  t=$1; gpu=$2; w=$3; var=${t%%_*}; r=${t#*_T}; T=${r%%_*}; part=${t##*_}
  if [ $var = o ]; then ins=T$T; el=ELQ_mix_T${T}; else ins=T${T}_APOCF; el=ELQ_mix_T${T}_APOCF; fi
  if [ $part = a ]; then jobs="Base:$ins ${el}_s0_step625:$ins"; else jobs="${el}_s1_step625:$ins ${el}_s2_step625:$ins"; fi
  log "start $t ($jobs) on worker $w"; t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$gpu $PY -u methods/esci_llama_eval.py --jobs $jobs --instr_file $(man $T) --rows $CAL --tag calib410 --ckpt_map $MAP --out_dir $OUT/eval --local_files_only > $OUT/logs/eval_$t.out 2>&1
  rc=$?; log "$t finished rc=$rc $(( $(date +%s) - t0 ))s"
  [ $rc -eq 0 ] || { echo "$t rc=$rc" >> $OUT/failed.txt; return 1; }
  $PY - <<PYEOF || { echo "$t truncation/row check failed" >> $OUT/failed.txt; log "$t truncation/row check FAILED"; return 1; }
import glob, gzip, json, sys
bad = []
for j in "$jobs".split():
    name, ins_ = j.split(":")
    f = "$OUT/eval/esciLlamaEval_%s__%s__calib410.json.gz" % (name, ins_)
    d = json.load(gzip.open(f, "rt", encoding="utf-8")); mt = d["meta"]
    n = sum(len(v) for v in d["raw"].values())
    sh = [v for k, v in mt.items() if "shorten" in k]
    if n != 3939 or any(int(x) != 0 for x in sh): bad.append((j, n, sh))
print("check", "$t", "ok" if not bad else bad); sys.exit(1 if bad else 0)
PYEOF
}
worker () {
  while true; do
    [ -f $OUT/failed.txt ] && { log "worker $2 stops: failed.txt exists"; return; }
    claimed=0
    for t in $TASKS; do
      mkdir "$OUT/claims/$t" 2>/dev/null || continue
      claimed=1; run_task $t $1 $2 || return; break
    done
    [ $claimed = 0 ] && { log "worker $2 has no more tasks"; return; }
  done
}
worker "$G0" w0 &
sleep 20
worker "$G1" w1 &
wait
log "ALL DONE"
