# Park_v1 语音转换验证说明

## 任务目的

`Park_v1` 是一个小规模的 Seed-VC 语音转换验证任务，不是 IndexTTS 训练任务。它从 SAPC 的 Parkinson's Disease 数据中选取一名 `low` 严重程度说话人和一名 `high` 严重程度说话人，并使用两人说出的同一句话 `what alarms are set` 进行双向转换。

这里希望将两个因素分开：

- **articulation（构音/发音方式）**：来自 source 音频，预期保留原有的障碍严重程度。
- **timbre（音色/说话人身份）**：来自 reference 音频，预期转换成参考说话人的声音。

因此，这个实验不是直接把 `low` 严重程度变成 `high`，而是交换说话人的音色，同时尽量保留源语音的构音严重程度。

## 音频文件

| 文件 | 含义 |
|---|---|
| `low_original.wav` | low 严重程度说话人的原始语音，speaker CER 为 0.0162，WER 为 0.0417 |
| `high_original.wav` | high 严重程度说话人的原始语音，speaker CER 为 0.6190，WER 为 0.8081 |
| `high_articulation_low_timbre.wav` | 使用 high 原始语音的构音方式，转换成 low 说话人的音色；预期严重程度仍为 high |
| `low_articulation_high_timbre.wav` | 使用 low 原始语音的构音方式，转换成 high 说话人的音色；预期严重程度仍为 low |

可以按照下面的顺序试听和比较：

1. 比较 `high_original.wav` 和 `high_articulation_low_timbre.wav`，检查构音障碍是否保留，同时音色是否接近 low 说话人。
2. 比较 `low_original.wav` 和 `low_articulation_high_timbre.wav`，检查清晰度是否保留，同时音色是否接近 high 说话人。

## 当前验证结果

`debug.json` 中的 speaker similarity 结果为：

| 转换结果 | 与目标音色的相似度 | 与源音色的相似度 | 判断 |
|---|---:|---:|---|
| high articulation → low timbre | 0.654776 | 0.026542 | 更接近目标 low 说话人 |
| low articulation → high timbre | 0.608562 | 0.341343 | 更接近目标 high 说话人 |

两种转换的音色方向都符合预期。生成音频也没有明显的 clipping。

当前检查只验证了音频有效性和**音色迁移**。它尚未验证转换后的 CER/WER，因此还不能仅凭 `debug.json` 确定源语音的 low/high 严重程度是否被完整保留。需要对两段转换音频重新运行 Parakeet ASR 并计算 CER，才能进一步验证严重程度。

## 相关结果文件

- `selection.json`：记录说话人、原始 CER/WER、样本和转换方向。
- `debug.json`：记录音频属性和说话人音色相似度结果。

