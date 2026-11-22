#!/bin/bash
# Long-duration CE and learning-rate runs of the training-budget analysis (Qwen3-4B, ESCI T1). Two workers, one per physical GPU (0 and 1), claim queue (mkdir). Tasks:
#   train_s{0,1,2}  : Qwen3-4B ESCI T1 CE-only, lr 2e-5, N = 2254 updates (frozen in selection/budget_manifest.json before any evaluation), seeds 0-2, final checkpoint only
#   curve_E4_s{k}, curve_EQ_s{k} : validation ('sel') evaluation of the STORED lr 2e-5 / 625 checkpoints at steps 63,157,313,469 (seed 0 additionally step 625 as a reproduction gate)
#   ev_sel_s{k}, ev_test_s{k} : final-checkpoint validation / test300 evaluation of the new long CE runs (claimed only after the training finished)
# Any failing task writes failed.txt and stops the queue (no retry). Device ids are not written to any log.
#   G0=<uuid> G1=<uuid> setsid nohup bash run380.sh > run380_stdout.txt 2>&1 < /dev/null &      (from the project root)
set -u
PY=${PY:-python}
export HF_HUB_OFFLINE=1
OUT=cpmr_experiment/applsci380
MAN=cpmr_experiment/esci_t/esci_t_manifest.json
AUD=cpmr_experiment/esci/audit
N=$($PY -c "import json;print(json.load(open('$OUT/selection/budget_manifest.json'))['N_updates'])")
mkdir -p $OUT/logs $OUT/eval $OUT/curves $OUT/claims $OUT/ckpt_maps
TASKS="train_s0 train_s1 train_s2 curve_E4_s0 curve_EQ_s0 curve_E4_s1 curve_EQ_s1 curve_E4_s2 curve_EQ_s2 ev_sel_s0 ev_sel_s1 ev_sel_s2 ev_test_s0 ev_test_s1 ev_test_s2"
log () { echo "$(date -u +%FT%T) $*" >> $OUT/scheduler.log; }
ready () {   # prerequisite of a task
  case $1 in
    ev_*_s*) s=${1##*_s}; [ -f $OUT/train/checkpoints/E380_T1_s$s/done.json ];;
    *) true;;
  esac
}
run_task () {   # $1 task, $2 gpu uuid, $3 worker label
  t=$1; gpu=$2; w=$3; log "start $t on worker $w"; t0=$(date +%s); rc=0
  case $t in
    train_s*) s=${t##*_s}
      CUDA_VISIBLE_DEVICES=$gpu $PY -u methods/esci_ce_train_380.py --instr T1 --seed $s --instr_file $MAN --rows $AUD/sft2_rows.jsonl --out_root $OUT/train --local_files_only --updates $N --tag_prefix E380 > $OUT/logs/train_s$s.out 2>&1; rc=$?;;
    curve_E4_s*|curve_EQ_s*) s=${t##*_s}; k=${t#curve_}; k=${k%%_s*}
      if [ $k = E4 ]; then M=E4_T1_s$s; D=esci_t; else M=EQ_mix_T1_s$s; D=esci_mix; fi
      steps="63 157 313 469"; [ $s = 0 ] && steps="$steps 625"
      jobs=""; map="{"; sep=""
      for n in $steps; do jobs="$jobs ${M}_step$n:T1"; map="$map$sep\"${M}_step$n\": \"cpmr_experiment/$D/train/checkpoints/$M/ckpt_step$n.pt\""; sep=", "; done
      echo "$map}" > $OUT/ckpt_maps/ckpt_map_$t.json
      CUDA_VISIBLE_DEVICES=$gpu $PY -u methods/esci_eval.py --jobs $jobs --instr_file $MAN --rows $AUD/sel_rows.jsonl --tag sel --ckpt_map $OUT/ckpt_maps/ckpt_map_$t.json --out_dir $OUT/curves --local_files_only > $OUT/logs/$t.out 2>&1; rc=$?;;
    ev_sel_s*|ev_test_s*) s=${t##*_s}; set=${t#ev_}; set=${set%%_s*}
      M=E380_T1_s$s; echo "{\"${M}_step$N\": \"$OUT/train/checkpoints/$M/ckpt_step$N.pt\"}" > $OUT/ckpt_maps/ckpt_map_$t.json
      if [ $set = sel ]; then R=sel_rows.jsonl; tg=sel; else R=test300_rows.jsonl; tg=test300; fi
      CUDA_VISIBLE_DEVICES=$gpu $PY -u methods/esci_eval.py --jobs ${M}_step$N:T1 --instr_file $MAN --rows $AUD/$R --tag $tg --ckpt_map $OUT/ckpt_maps/ckpt_map_$t.json --out_dir $OUT/eval --local_files_only > $OUT/logs/$t.out 2>&1; rc=$?;;
  esac
  log "finished $t rc=$rc $(( $(date +%s) - t0 ))s"
  [ $rc -eq 0 ] || { echo "$t rc=$rc" >> $OUT/failed.txt; return 1; }
}
worker () {   # $1 gpu uuid, $2 label
  while true; do
    [ -f $OUT/failed.txt ] && { log "worker $2 stops: failed.txt exists"; return; }
    left=0; did=0
    for t in $TASKS; do
      [ -d $OUT/claims/$t ] && continue
      left=1
      ready $t || continue
      mkdir $OUT/claims/$t 2>/dev/null || continue
      run_task $t $1 $2 || return; did=1; break
    done
    [ $left = 0 ] && { log "worker $2 has no more tasks"; return; }
    [ $did = 0 ] && sleep 30
  done
}
log "instance start (N=$N)"
worker "$G0" w0 & sleep 15; worker "$G1" w1 & wait
log "ALL DONE"
