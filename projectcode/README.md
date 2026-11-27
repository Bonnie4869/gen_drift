# GenDrift

本目录是 GenDrift（迭代生成漂移）的代码、样本和结果。

一行样本的流程是：

```text
I0 → T1 → I1 → T2 → I2 → T3 → I3 → T4 → I4 → T5 → I5
                                              └→ T6
```

`T1` 到 `T5` 由 Qwen3-VL 写成一段描述，并原样交给 Stable Diffusion 3 生成下一张图。`T6` 是 `I5` 的描述，只保存，不再送进生图。CSV 里的对象和 attribute 会写入结果目录，不会进入 prompt。

下文路径都相对于本目录。

## 换机器时要改的路径

模型目录不通过命令行传入。打开 `run_iterative.py`，改这两行：

```python
VLM_PATH = Path(r"D:\models\Qwen3-VL-4B-Instruct")
SD_PATH = Path(r"D:\models\stable-diffusion-3-medium")
```

分别指向本机的 Qwen3-VL-4B-Instruct 和 stable-diffusion-3-medium 权重目录。

样本表 `all_five_objects_sorted_by_attributes.csv` 与脚本放在同一目录，一般不用改。

## 环境

需要一块 NVIDIA GPU，以及已经装好的 CUDA 版 PyTorch、支持 Qwen3-VL 的 `transformers`。再安装 `requirements.txt`：

```text
pip install -r requirements.txt
```

不要重新安装 PyTorch，避免把 CUDA 版本换成 CPU 版本。8GB 显存上，脚本会把 T5 按 block 轮流放到 GPU，并优先用 768 分辨率；显存不够时降到 512。

## 运行

在本目录执行，把 `python` 换成当前环境的解释器：

```text
python -u run_iterative.py --image-index 0
```

`--image-index` 是 CSV 行号，从 0 开始。`0` 是第一行。当前表有 317 行，合法范围是 0 到 316。省略该参数时默认是 0。

同一行再跑会跳过已经写好的 `T` 和 `I` 文件。输出在 `runs/single_<image_id>/`，其中 `<image_id>` 来自 CSV 第一列。原图下载到 `data/images/`。单次测试的整理稿放在本目录的 `results.md`，不放进 `runs`。

## 评估已有结果（Windows 服务器）

新增脚本只读取已有图文及标注，输出到样本目录的 `evaluation/`。
不需要重新生成图片。默认模型路径：

- CLIP：`D:\bonnie\aigc\models\clip-vit-large-patch14`
- Grounding DINO：`D:\bonnie\aigc\models\grounding-dino-base`

使用现有 CUDA 环境，不要直接升级或重装 torch / transformers。
评估需要已有环境中的 torch、transformers、Pillow；transformers 须支持
`AutoModelForZeroShotObjectDetection` 和 Grounding DINO。脚本记录实际版本，
所有模型均使用 `local_files_only=True`。缺依赖或加载失败时先查看具体报错。

```powershell
cd D:\bonnie\aigc\gendrift
& ..\.venv\Scripts\Activate.ps1
python check_eval_models.py --device cuda
python evaluate_clip.py --device cuda
python evaluate_dino.py --device cuda
```

默认样本为 `runs/single_497948`，可使用 `--run-dir runs/single_<image_id>`
指定其他样本；三个脚本均支持 `--output-dir` 和 `--device auto|cpu|cuda`。
验证脚本支持 `--clip-path`、`--dino-path`、`--image`；两个评估脚本使用
`--model-path` 覆盖对应路径。模型依次加载，减少同时占用显存。

输出：

| 文件 | 内容 |
|---|---|
| `model_check.json` | 两个模型的离线加载及实际推理结果、版本、失败堆栈 |
| `clip_metrics.csv/json` | I0 与各代、相邻代、It 与 Tt 的原始余弦相似度 |
| `dino_metrics.csv/json` | 五类目标的 ORR、类别 Precision/Recall/F1 |
| `dino_detections.json` | 每张图片的类别、分数、原图像素坐标 xyxy 框 |
| `dino_boxes/I*.jpg` | 画框预览，供人工检查漏检、误检 |

CLIP 对投影向量归一化后计算余弦，不使用缩放后的 logits。
文本按模型上限（通常含特殊 token 共 77）截断，记录原始 token 数及截断标志。
I0 没有生成 prompt，图文指标留空；缺失 caption 或上一代图时对应指标留空。
T6 是 I5 的后续描述，不作为 I5 的生成 prompt。

DINO 对 `man / window / tree / building / shirt` 分别查询，避免多类别短语映射混淆。
默认框阈值 0.35、文本阈值 0.25，可通过 `--box-threshold`、`--text-threshold` 调整。
这些阈值是待人工校准的起点；检测框数量不作为实例计数指标。
参考集合读取 meta.json 的 `ground_truth_not_used_in_prompts`，不从 I0 检测结果推导。
因此 I0 的 ORR 也可能低于 1，可用于发现原图漏检。

ORR = 保留的原图类别数 / 原图目标类别数，与这里的类别 Recall 相同。
类别 Precision/F1 只在五类词表内比较生成图与原图类别，不能解释成检测器准确率，
也不能量化词表外新增对象。无定义的比值输出 null（CSV 留空）。
请先人工检查 `dino_boxes`，再据此选择阈值；检测置信分数不是人工验证的概率。

每次运行覆盖对应评估输出。比较不同阈值或模型时请指定独立 `--output-dir`。
这些脚本尚未在服务器模型上验证；本地接口检查不代表已取得评估结果。

API 参考：[CLIP](https://huggingface.co/docs/transformers/model_doc/clip)、
[Grounding DINO](https://huggingface.co/docs/transformers/model_doc/grounding-dino)。
DINO 后处理兼容 `box_threshold` 与较新版本的 `threshold` 参数名。

## 批量生成与评估全部 317 个候选样本

同步 `run_batch.py` 和更新后的 `run_iterative.py`。批量脚本为生成子进程通过
`GENDRIFT_VLM_PATH` / `GENDRIFT_SD_PATH` 指定权重路径；单样本脚本默认路径保持原值。
SD3 的服务器位置尚未确认，因此必须显式传入真实目录：

```powershell
python run_batch.py --sd-path "D:\实际路径\stable-diffusion-3-medium"
```

默认依次执行生成、CLIP、DINO。可用 `--mode generate` 仅生成，
`--mode evaluate` 仅评估完整的已有结果，不需要 SD3 路径。
`--start 0 --stop 3` 表示 CSV 的前三行，stop 不包含在内。
`--dry-run` 只预览任务，不加载模型、不写文件。

完整样本会跳过生成；不完整样本由原生成脚本按已有 T/I 文件续跑。
CLIP/DINO 每次重新评估并覆盖对应输出，因此不同配置比较前应备份 evaluation。
不要在改变生成模型/参数后对已有半成品直接续跑，以免混合配置。
评估不完整样本会记录失败，不输出部分链路指标。
单样本失败后继续其他样本；最终有失败则退出码为 1。
Ctrl+C 可停止批次，重新运行原命令即可继续。
各阶段日志在 `runs/batch/<image_id>_<stage>.log`，
批次历史在 `runs/batch/history.jsonl`（追加记录）。图片与评估仍在各样本目录。

这是候选表全集运行，并非已定义的独立测试集；不包含固定 prompt 对照组或 ImageReward。
模型按样本/阶段重新加载以隔离显存和异常，速度会受加载开销影响。
按此前单样本 24.5 分钟估算，317 张仅生成约 129 小时；实际服务器速度可能不同。
