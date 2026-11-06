#!/bin/bash
set -e
# Appendix: FP32 re-evaluation of the representative Gemma-3-4B-it checkpoints (ESCI and MS MARCO, T1 and T7, seed 0, CE / soft / Any); the manifest lists the checkpoints and their stored thresholds
# edit the 'path' entries of cpmr_experiment/gemma231/sample_manifest.json if the checkpoints were trained elsewhere
CUDA_VISIBLE_DEVICES=0 python methods/gemma231_audit.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds ESCI --T 1 --loss CE --out cpmr_experiment/gemma231/out
CUDA_VISIBLE_DEVICES=0 python methods/gemma234_fp32_eval.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds ESCI --T 1 --loss CE --mode full --out cpmr_experiment/gemma234/out
CUDA_VISIBLE_DEVICES=0 python methods/gemma231_audit.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds ESCI --T 1 --loss SOFT --out cpmr_experiment/gemma231/out
CUDA_VISIBLE_DEVICES=0 python methods/gemma234_fp32_eval.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds ESCI --T 1 --loss SOFT --mode full --out cpmr_experiment/gemma234/out
CUDA_VISIBLE_DEVICES=0 python methods/gemma231_audit.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds ESCI --T 1 --loss ANY --out cpmr_experiment/gemma231/out
CUDA_VISIBLE_DEVICES=0 python methods/gemma234_fp32_eval.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds ESCI --T 1 --loss ANY --mode full --out cpmr_experiment/gemma234/out
CUDA_VISIBLE_DEVICES=0 python methods/gemma231_audit.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds ESCI --T 7 --loss CE --out cpmr_experiment/gemma231/out
CUDA_VISIBLE_DEVICES=0 python methods/gemma234_fp32_eval.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds ESCI --T 7 --loss CE --mode full --out cpmr_experiment/gemma234/out
CUDA_VISIBLE_DEVICES=0 python methods/gemma231_audit.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds ESCI --T 7 --loss SOFT --out cpmr_experiment/gemma231/out
CUDA_VISIBLE_DEVICES=0 python methods/gemma234_fp32_eval.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds ESCI --T 7 --loss SOFT --mode full --out cpmr_experiment/gemma234/out
CUDA_VISIBLE_DEVICES=0 python methods/gemma231_audit.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds ESCI --T 7 --loss ANY --out cpmr_experiment/gemma231/out
CUDA_VISIBLE_DEVICES=0 python methods/gemma234_fp32_eval.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds ESCI --T 7 --loss ANY --mode full --out cpmr_experiment/gemma234/out
CUDA_VISIBLE_DEVICES=0 python methods/gemma231_audit.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds MS --T 1 --loss CE --out cpmr_experiment/gemma231/out
CUDA_VISIBLE_DEVICES=0 python methods/gemma234_fp32_eval.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds MS --T 1 --loss CE --mode full --out cpmr_experiment/gemma234/out
CUDA_VISIBLE_DEVICES=0 python methods/gemma231_audit.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds MS --T 1 --loss SOFT --out cpmr_experiment/gemma231/out
CUDA_VISIBLE_DEVICES=0 python methods/gemma234_fp32_eval.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds MS --T 1 --loss SOFT --mode full --out cpmr_experiment/gemma234/out
CUDA_VISIBLE_DEVICES=0 python methods/gemma231_audit.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds MS --T 1 --loss ANY --out cpmr_experiment/gemma231/out
CUDA_VISIBLE_DEVICES=0 python methods/gemma234_fp32_eval.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds MS --T 1 --loss ANY --mode full --out cpmr_experiment/gemma234/out
CUDA_VISIBLE_DEVICES=0 python methods/gemma231_audit.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds MS --T 7 --loss CE --out cpmr_experiment/gemma231/out
CUDA_VISIBLE_DEVICES=0 python methods/gemma234_fp32_eval.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds MS --T 7 --loss CE --mode full --out cpmr_experiment/gemma234/out
CUDA_VISIBLE_DEVICES=0 python methods/gemma231_audit.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds MS --T 7 --loss SOFT --out cpmr_experiment/gemma231/out
CUDA_VISIBLE_DEVICES=0 python methods/gemma234_fp32_eval.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds MS --T 7 --loss SOFT --mode full --out cpmr_experiment/gemma234/out
CUDA_VISIBLE_DEVICES=0 python methods/gemma231_audit.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds MS --T 7 --loss ANY --out cpmr_experiment/gemma231/out
CUDA_VISIBLE_DEVICES=0 python methods/gemma234_fp32_eval.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds MS --T 7 --loss ANY --mode full --out cpmr_experiment/gemma234/out
