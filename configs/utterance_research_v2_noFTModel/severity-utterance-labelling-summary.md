# Utterance Severity Label Summary

- Parakeet model: `/srv/scratch/speechdata/SAPC_Team/parakeet-tdt-1.1b/parakeet-tdt-1.1b.nemo`
- Dataset splits: dev, train
- Total utterances: 383997
- Total speakers: 999
- Empty ASR hypotheses: 2552
- Missing reference transcripts skipped: 7
- Missing reference transcript CSV: `/srv/scratch/speechdata/Jinghao_shared/parakeet_runs/utterance_research_v2_noFTModel/missing-reference-transcripts.csv`
- Threshold low/middle: 0.200000
- Threshold middle/high: 0.500000

## Utterance Labels

| Severity | Utterances |
| --- | --- |
| low | 255667 |
| middle | 53029 |
| high | 75301 |

## CER Bins

| CER Bin | Utterances | Percent |
| --- | --- | --- |
| 0-5% | 217957 | 56.76% |
| 5-10% | 23352 | 6.08% |
| 10-20% | 32282 | 8.41% |
| 20-45% | 45717 | 11.91% |
| 45%+ | 64689 | 16.85% |

## Speaker Labels

| Severity | Speakers |
| --- | --- |
| low | 614 |
| middle | 231 |
| high | 154 |

## Splits

| Split | Utterances |
| --- | --- |
| dev | 47929 |
| train | 336068 |

## Split Metrics

| Split | Mean WER | Median WER | Mean CER | Median CER |
| --- | --- | --- | --- | --- |
| dev | 0.280847 | 0.076923 | 0.198044 | 0.027778 |
| train | 0.254456 | 0.026316 | 0.175913 | 0.010000 |

## Severity Metrics

| Severity | Mean WER | Median WER | Mean CER | Median CER |
| --- | --- | --- | --- | --- |
| low | 0.036790 | 0.000000 | 0.017933 | 0.000000 |
| middle | 0.409292 | 0.400000 | 0.245342 | 0.238095 |
| high | 0.901250 | 1.000000 | 0.677490 | 0.666667 |
