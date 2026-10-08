# Utterance Severity Label Summary

- Parakeet model: `/srv/scratch/speechdata/SAPC_Team/parakeet-tdt-1.1b/parakeet-tdt-1.1b.nemo`
- Dataset splits: dev, train
- Total utterances: 369320
- Total speakers: 962
- Empty ASR hypotheses: 2554
- Threshold low/middle: 0.250000
- Threshold middle/high: 0.500000

## Utterance Labels

| Severity | Utterances |
| --- | --- |
| low | 270618 |
| middle | 34528 |
| high | 64174 |

## Speaker Labels

| Severity | Speakers |
| --- | --- |
| low | 694 |
| middle | 143 |
| high | 125 |

## Splits

| Split | Utterances |
| --- | --- |
| dev | 47833 |
| train | 321487 |

## Split Metrics

| Split | Mean WER | Median WER | Mean CER | Median CER |
| --- | --- | --- | --- | --- |
| dev | 0.238127 | 0.009901 | 0.163751 | 0.003021 |
| train | 0.232995 | 0.000000 | 0.161038 | 0.000000 |

## Severity Metrics

| Severity | Mean WER | Median WER | Mean CER | Median CER |
| --- | --- | --- | --- | --- |
| low | 0.046939 | 0.000000 | 0.023452 | 0.000000 |
| middle | 0.448749 | 0.444444 | 0.268738 | 0.265306 |
| high | 0.905321 | 1.000000 | 0.685304 | 0.684211 |
