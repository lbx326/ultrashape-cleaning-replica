# Benchmark on HSSD samples

Aggregated from 20 mesh reports in `outputs\benchmark\reports`.

## Headline numbers

| Metric | Mean | Median | Min | Max |
|--------|-----:|-------:|----:|----:|
| Total wall time (s) | 143.769 | 139.970 | 129.987 | 201.667 |
| Stage 1 wall time (s) | 9.794 | 8.416 | 6.105 | 19.547 |
| Stage 2 wall time (s) | 96.401 | 97.118 | 89.419 | 106.179 |
| Stage 3 wall time (s) | 0.474 | 0.289 | 0.140 | 1.702 |
| Stage 4 wall time (s) | 36.562 | 33.224 | 27.679 | 84.772 |
| Stage 1 chamfer vs input | 0.007 | 0.004 | 0.001 | 0.022 |
| Stage 4 ray sign agreement | 0.908 | 0.988 | 0.415 | 1.000 |
| Stage 4 VAE chamfer | 0.075 | 0.059 | 0.008 | 0.196 |

## Binary rates

- Watertight after Stage 1: 20/20 (100.0%)
- Winding consistent after Stage 1: 20/20 (100.0%)
- Stage 2 VLM accept: 9/20 (45.0%)
- Overall pipeline accept: 5/20 (25.0%)

## Per-mesh breakdown

| # | sha256[:10] | V_in | F_in | V_out | F_out | WT | wind | chamf | ray_agree | VAE_chamf | class | qual | accept | reasons |
|---|-------------|-----:|-----:|------:|------:|----|------|------:|----------:|----------:|-------|-----:|:------|:--------|
| 1 | 3a9377616f | 900 | 1468 | 213864 | 427724 | Y | Y | 0.0011 | 0.589 | 0.04803973915323781 | unidentifiable | 3 | N | stage4:ray_sign_disagreement=0.589;stage2:noisy_scan |
| 2 | 22a6d8c07c | 802 | 1600 | 376840 | 753676 | Y | Y | 0.0101 | 0.994 | 0.19069075717709122 | unidentifiable | 2 | N | stage4:thin_shell vol/area=0.0090;stage4:vae_chamfer=0.1907 > 0.15;stage2:primitive |
| 3 | 2a7342c1f2 | 1059 | 1580 | 579392 | 1158796 | Y | Y | 0.0059 | 0.995 | 0.08358777434433823 | chair | 3 | Y | - |
| 4 | c974198c43 | 888 | 780 | 996076 | 1992152 | Y | Y | 0.0020 | 0.946 | 0.11718710856319943 | unidentifiable | 3 | N | stage4:thin_shell vol/area=0.0007;stage2:noisy_scan |
| 5 | bc5370562e | 38371 | 51064 | 610286 | 1220612 | Y | Y | 0.0161 | 0.999 | 0.05717497762979626 | appliance | 4 | Y | - |
| 6 | 0b23217157 | 24790 | 23171 | 628754 | 1257512 | Y | Y | 0.0026 | 0.906 | 0.02284506268744831 | unidentifiable | 3 | N | stage4:ray_sign_disagreement=0.906;stage2:noisy_scan |
| 7 | 865762d484 | 768 | 508 | 396024 | 792044 | Y | Y | 0.0007 | 0.415 | 0.19615187776454057 | unidentifiable | 2 | N | stage4:thin_shell vol/area=0.0022;stage4:ray_sign_disagreement=0.415;stage4:vae_chamfer=0.1962 > 0.15;stage2:primitive;stage2:noisy_scan |
| 8 | 75268284ec | 30065 | 51340 | 598834 | 1197668 | Y | Y | 0.0224 | 0.994 | 0.033506466773715306 | bed | 3 | N | stage2:noisy_scan |
| 9 | d3597960e7 | 34387 | 66304 | 624680 | 1249392 | Y | Y | 0.0012 | 0.975 | 0.012814761356055438 | unidentifiable | 3 | N | stage4:thin_shell vol/area=0.0005;stage2:noisy_scan |
| 10 | 47121f4491 | 3611 | 5925 | 304054 | 608116 | Y | Y | 0.0035 | 0.967 | 0.1493524173684097 | unidentifiable | 3 | N | stage2:noisy_scan |
| 11 | 8b11a5b62b | 31776 | 52384 | 853272 | 1706556 | Y | Y | 0.0222 | 0.995 | 0.06819846497017813 | bed | 4 | Y | - |
| 12 | 99eaf7ba08 | 6360 | 11098 | 1447040 | 2894324 | Y | Y | 0.0018 | 0.971 | 0.013614370225717943 | vase | 3 | N | stage4:thin_shell vol/area=0.0015;stage2:noisy_scan |
| 13 | 3f3ea70681 | 3363 | 6378 | 691800 | 1383596 | Y | Y | 0.0025 | 0.481 | 0.06441280040597216 | unidentifiable | 3 | N | stage4:ray_sign_disagreement=0.481;stage2:noisy_scan |
| 14 | 2a7b76dec1 | 21828 | 36936 | 196046 | 392112 | Y | Y | 0.0025 | 0.983 | 0.007637324202876218 | orchid | 3 | N | stage4:thin_shell vol/area=0.0038 |
| 15 | 22b0de817d | 64 | 36 | 588840 | 1177676 | Y | Y | 0.0039 | 0.950 | 0.15655436204183532 | unidentifiable | 3 | N | stage4:thin_shell vol/area=0.0085;stage4:vae_chamfer=0.1566 > 0.15 |
| 16 | 9f64ac419f | 2440 | 1696 | 580816 | 1161628 | Y | Y | 0.0114 | 1.000 | 0.09055644883362435 | box | 3 | N | stage2:primitive |
| 17 | 9d6086a607 | 19607 | 36499 | 166578 | 333220 | Y | Y | 0.0026 | 0.996 | 0.00976912942398785 | chandelier | 4 | N | stage4:thin_shell vol/area=0.0040 |
| 18 | 4e032b7c33 | 2662 | 1884 | 278542 | 557080 | Y | Y | 0.0108 | 0.999 | 0.05542918767879489 | cabinet | 3 | Y | - |
| 19 | 4fc559ec29 | 3066 | 1492 | 831432 | 1662860 | Y | Y | 0.0090 | 0.999 | 0.06054067618351096 | cabinet | 4 | N | stage4:thin_shell vol/area=0.0094 |
| 20 | 2c8f67e344 | 5506 | 2660 | 348026 | 696048 | Y | Y | 0.0123 | 0.998 | 0.05540597577574228 | unidentifiable | 3 | Y | - |
