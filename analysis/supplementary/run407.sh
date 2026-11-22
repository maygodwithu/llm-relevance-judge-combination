#!/bin/bash
# BCE / MAE controls of ESCI Qwen3-4B T2-T7 (extension of the T1 controls). Two workers, one per physical GPU 0 / 1, claim queue (mkdir). 36 training runs (T2-T7 x seeds 0-2 x {mae, bce}), each followed by its own final validation ('sel') and test300 evaluation:
#   mae_s0 mae_s1 mae_s2 : --loss mixmae --lam 2 (CE + 2 x mean|y - sigmoid(m)|)          bce_s1 bce_s2 : --loss mixbce --lam 1 (CE + mean BCEWithLogits(m, y)), seed 0 is the stored 245 run
# lr 2e-5 (config), 625 updates, final checkpoint only. A failing task writes failed.txt and stops the queue (no retry). Device ids are never written to a log.
#   G0=<uuid> G1=<uuid> setsid nohup bash run407.sh > run384_stdout.txt 2>&1 < /dev/null &      (from the project root)
set -u
PY=${PY:-python}
export HF_HUB_OFFLINE=1
OUT=cpmr_experiment/esci407
MAN=cpmr_experiment/esci_t/esci_t_manifest.json
AUD=cpmr_experiment/esci/audit
mkdir -p $OUT/logs $OUT/eval $OUT/claims $OUT/train $OUT/ckpt_maps
TASKS=""
for T in 2 3 4 5 6 7; do for s in 0 1 2; do for k in mae bce; do TASKS="$TASKS ${k}_T${T}_s${s}"; done; done; done
log () { echo "$(date -u +%FT%T) $*" >> $OUT/scheduler.log; }
run_task () {   # $1 task, $2 gpu, $3 worker label
  t=$1; gpu=$2; w=$3; s=${t##*_s}; k=${t%%_T*}; T=${t#*_T}; T=${T%%_s*}
  if [ $k = mae ]; then loss=mixmae; lam=2; pre=Q407M; else loss=mixbce; lam=1; pre=Q407B; fi
  tag=${pre}_${loss}_T${T}_s$s
  log "start train $t ($tag) on worker $w"; t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$gpu $PY -u methods/esci_loss_train_384.py --loss $loss --lam $lam --tag_prefix $pre --instr T$T --seed $s --instr_file $MAN --out_root $OUT/train --final_only --local_files_only > $OUT/logs/train_$t.out 2>&1
  rc=$?; log "train $t finished rc=$rc $(( $(date +%s) - t0 ))s"
  [ $rc -eq 0 ] || { echo "train $t rc=$rc" >> $OUT/failed.txt; return 1; }
  echo "{\"${tag}_step625\": \"$OUT/train/checkpoints/$tag/ckpt_step625.pt\"}" > $OUT/ckpt_maps/ckpt_map_$t.json
  for pair in "sel:sel_rows.jsonl" "test300:test300_rows.jsonl"; do
    sname=${pair%%:*}; rows=${pair#*:}
    log "start eval $t $sname on worker $w"; t1=$(date +%s)
    CUDA_VISIBLE_DEVICES=$gpu $PY -u methods/esci_eval.py --jobs ${tag}_step625:T$T --instr_file $MAN --rows $AUD/$rows --tag $sname --ckpt_map $OUT/ckpt_maps/ckpt_map_$t.json --out_dir $OUT/eval --local_files_only > $OUT/logs/eval_${t}_$sname.out 2>&1
    rc=$?; log "eval $t $sname finished rc=$rc $(( $(date +%s) - t1 ))s"
    [ $rc -eq 0 ] || { echo "eval $t $sname rc=$rc" >> $OUT/failed.txt; return 1; }
  done
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
log "instance start"
worker "$G0" w0 &
sleep 20
worker "$G1" w1 &
wait
log "ALL DONE"
