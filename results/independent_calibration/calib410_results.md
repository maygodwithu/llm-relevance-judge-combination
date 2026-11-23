## Overall (SFT: per seed the mean over instructions, then mean ± SD over 3 seeds; Base: mean over instructions)
| variant | objective | t* shared | t* independent | test k(0) | test k(shared t*) | test k(independent t*) | AUC |
|---|---|---|---|---|---|---|---|
| original | Base | +3.628 | +2.828 | 0.2230 | 0.2976 | 0.2898 | 0.6973 |
| original | CE+soft kappa | -0.097 | -0.523 | 0.4032 ± 0.0019 | 0.3996 ± 0.0031 | 0.3959 ± 0.0022 | 0.7623 |
| PO | Base | +4.602 | +3.995 | 0.2430 | 0.3095 | 0.3051 | 0.7050 |
| PO | CE+soft kappa | -0.258 | -0.478 | 0.4090 ± 0.0008 | 0.4026 ± 0.0024 | 0.3997 ± 0.0039 | 0.7650 |

## PO - original (all instructions; T1, T4, T6 kept the original text so their difference is exactly 0)
| regime | objective | PO - original | seed diffs | positive seeds | 95% CI |
|---|---|---|---|---|---|
| t=0 | Base | +0.0200 | +0.0200 | 1/1 | [+0.0114, +0.0287] |
| shared validation t* | Base | +0.0119 | +0.0119 | 1/1 | [+0.0033, +0.0205] |
| independent calibration t* | Base | +0.0153 | +0.0153 | 1/1 | [+0.0075, +0.0230] |
| t=0 | CE+soft kappa | +0.0058 | +0.0043, +0.0058, +0.0073 | 3/3 | [+0.0010, +0.0105] |
| shared validation t* | CE+soft kappa | +0.0031 | +0.0024, +0.0038, +0.0031 | 3/3 | [-0.0015, +0.0078] |
| independent calibration t* | CE+soft kappa | +0.0038 | +0.0058, +0.0036, +0.0021 | 3/3 | [-0.0010, +0.0086] |

## Change of the PO gain: independent t* minus shared t*
| objective | change | seed diffs | positive seeds | 95% CI |
|---|---|---|---|---|
| Base | +0.0034 | +0.0034 | 1/1 | [-0.0018, +0.0087] |
| CE+soft kappa | +0.0007 | +0.0034, -0.0002, -0.0010 | 1/3 | [-0.0030, +0.0044] |

## PO - original, changed-text instructions only (T2,T3,T5,T7)
| regime | objective | PO - original | seed diffs | positive seeds | 95% CI |
|---|---|---|---|---|---|
| t=0 | Base | +0.0349 | +0.0349 | 1/1 | [+0.0200, +0.0502] |
| shared validation t* | Base | +0.0209 | +0.0209 | 1/1 | [+0.0058, +0.0359] |
| independent calibration t* | Base | +0.0268 | +0.0268 | 1/1 | [+0.0131, +0.0403] |
| t=0 | CE+soft kappa | +0.0102 | +0.0076, +0.0101, +0.0128 | 3/3 | [+0.0017, +0.0184] |
| shared validation t* | CE+soft kappa | +0.0054 | +0.0043, +0.0066, +0.0054 | 3/3 | [-0.0026, +0.0136] |
| independent calibration t* | CE+soft kappa | +0.0067 | +0.0102, +0.0063, +0.0036 | 3/3 | [-0.0018, +0.0150] |

## Per instruction, CE+soft kappa PO - original (seed mean; unadjusted 95% CI)
| T | text changed | t=0 | shared t* | independent t* |
|---|---|---|---|---|
| T1 | no | +0.0000 [+0.0000, +0.0000] (0/3) | +0.0000 [+0.0000, +0.0000] (0/3) | +0.0000 [+0.0000, +0.0000] (0/3) |
| T2 | yes | +0.0274 [+0.0093, +0.0458] (3/3) | +0.0119 [-0.0064, +0.0309] (3/3) | +0.0175 [-0.0003, +0.0355] (3/3) |
| T3 | yes | -0.0065 [-0.0206, +0.0071] (0/3) | -0.0114 [-0.0269, +0.0037] (0/3) | -0.0054 [-0.0183, +0.0074] (1/3) |
| T4 | no | +0.0000 [+0.0000, +0.0000] (0/3) | +0.0000 [+0.0000, +0.0000] (0/3) | +0.0000 [+0.0000, +0.0000] (0/3) |
| T5 | yes | +0.0215 [+0.0079, +0.0362] (3/3) | +0.0178 [+0.0034, +0.0322] (3/3) | +0.0222 [+0.0071, +0.0376] (3/3) |
| T6 | no | +0.0000 [+0.0000, +0.0000] (0/3) | +0.0000 [+0.0000, +0.0000] (0/3) | +0.0000 [+0.0000, +0.0000] (0/3) |
| T7 | yes | -0.0017 [-0.0158, +0.0113] (2/3) | +0.0034 [-0.0102, +0.0160] (2/3) | -0.0074 [-0.0211, +0.0051] (1/3) |
