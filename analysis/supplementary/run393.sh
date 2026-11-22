#!/bin/bash
# ESCI Llama-3.1-8B, instruction T2 (original) and T2_APOCF (selected PO text); 8 scoring jobs on the frozen independent calibration pool.
# Process A (GPU 0): Base:T2, then ELQ_mix_T2_s0/1/2.   Process B (GPU 1): Base:T2_APOCF, then ELQ_mix_T2_APOCF_s0/1/2.   Base first within each process (adapter must be zero).
# Stage 0 (gate, fixed before the run): the same evaluator re-scores the first 20 stored selection queries for Base:T2 and ELQ_mix_T2_s0 and the margins must equal the stored ones (max |diff| <= 1e-4).
# No retraining, no PO, no optimizer step. Any failure stops the run (no retry). Device ids are never written to a log.
#   G0=<uuid> G1=<uuid> bash run393.sh   (from the project root)
set -u
PY=${PY:-python}
export HF_HUB_OFFLINE=1
OUT=cpmr_experiment/esci393
MAN=cpmr_experiment/esci_llama/p1_T2/esci_apocf_manifest.json
CAL=$OUT/calib/calib_rows.jsonl
CK=cpmr_experiment/esci_204/train/checkpoints
mkdir -p $OUT/eval $OUT/logs $OUT/gate $OUT/ckpt_maps
log () { echo "$(date -u +%FT%T) $*" >> $OUT/scheduler.log; }
mapA="{"; mapB="{"; sepA=""; sepB=""
for s in 0 1 2; do
  mapA="$mapA$sepA\"ELQ_mix_T2_s${s}_step625\": \"$CK/ELQ_mix_T2_s$s/ckpt_step625.pt\""; sepA=", "
  mapB="$mapB$sepB\"ELQ_mix_T2_APOCF_s${s}_step625\": \"$CK/ELQ_mix_T2_APOCF_s$s/ckpt_step625.pt\""; sepB=", "
done
echo "$mapA}" > $OUT/ckpt_maps/map_orig.json; echo "$mapB}" > $OUT/ckpt_maps/map_po.json
# --- stage 0 gate rows: first 20 stored selection queries
$PY - <<PYEOF
import json
rows=[json.loads(l) for l in open("cpmr_experiment/esci/audit/sel_rows.jsonl",encoding="utf-8")]
q=[]; 
for r in rows:
    if r["nq"] not in q: q.append(r["nq"])
keep=set(q[:20]); out=[r for r in rows if r["nq"] in keep]
open("$OUT/gate/sel_first20_rows.jsonl","w",encoding="utf-8",newline="\n").write("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in out)); print(len(out))
PYEOF
log "gate start"
CUDA_VISIBLE_DEVICES=$G0 $PY -u methods/esci_llama_eval.py --jobs Base:T2 ELQ_mix_T2_s0_step625:T2 --instr_file $MAN --rows $OUT/gate/sel_first20_rows.jsonl --tag gate --ckpt_map $OUT/ckpt_maps/map_orig.json --out_dir $OUT/gate --local_files_only > $OUT/logs/gate.out 2>&1
rc=$?; log "gate eval rc=$rc"; [ $rc -eq 0 ] || { echo "gate eval failed" > $OUT/ABORT.txt; exit 1; }
log "scoring start"
( CUDA_VISIBLE_DEVICES=$G0 $PY -u methods/esci_llama_eval.py --jobs Base:T2 ELQ_mix_T2_s0_step625:T2 ELQ_mix_T2_s1_step625:T2 ELQ_mix_T2_s2_step625:T2 --instr_file $MAN --rows $CAL --tag calib393 --ckpt_map $OUT/ckpt_maps/map_orig.json --out_dir $OUT/eval --local_files_only > $OUT/logs/eval_A.out 2>&1; echo "A rc=$?" >> $OUT/status.txt ) &
sleep 20
( CUDA_VISIBLE_DEVICES=$G1 $PY -u methods/esci_llama_eval.py --jobs Base:T2_APOCF ELQ_mix_T2_APOCF_s0_step625:T2_APOCF ELQ_mix_T2_APOCF_s1_step625:T2_APOCF ELQ_mix_T2_APOCF_s2_step625:T2_APOCF --instr_file $MAN --rows $CAL --tag calib393 --ckpt_map $OUT/ckpt_maps/map_po.json --out_dir $OUT/eval --local_files_only > $OUT/logs/eval_B.out 2>&1; echo "B rc=$?" >> $OUT/status.txt ) &
wait
log "ALL DONE"
