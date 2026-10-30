# ActionFormer 接入依据与适配边界

## 来源与结论

固定官方仓库 `happyharrycn/actionformer_release`，commit **`61ea7eb9308a568b0cf45e3804830836e30061de`**。归档 SHA256 为 `44bb86164e62a2ddbfd4a59b80f908e5a62add183d40e2e7955efe1e5d98d82a`，与第一次下载的归档相同。官方文件原样保存在 `vendor/actionformer/`，逐文件哈希在 `vendor/SOURCE.json`，加载时核验。

这项实验是 **在 TASLE 的 XCLIP/DeMamba 特征上接入官方 ActionFormer 模块**，不是复现 THUMOS14 成绩。没有适用于任意视觉编码器/数据集的“正确万能配置”。官方模型/求解器可以直接复用，数据维度、时间坐标、标签及评测协议必须依本实验设置。下面的优化超参数是公开配方起点，未宣称已在 TASLE 上调优。

## 官方实现的直接调用

| 内容 | 官方文件 | 当前接入 |
| --- | --- | --- |
| Transformer、时间金字塔、分类/回归头、目标分配 | `libs/modeling/{backbones,necks,meta_archs}.py` | `make_meta_arch('LocPointTransformer', ...)`，直接返回官方类 |
| Focal / DIoU loss | `libs/modeling/losses.py` | 在官方 `PtTransformer.forward/losses` 内计算 |
| 高维输入投影 | `backbones.py` 的原生 list-input `proj` | `input_dim=[D]`、`embd_dim=[512]`；使用官方 `MaskedConv1D(...,1)`，没有自写投影块或额外输入 LayerNorm |
| 优化器及 weight decay 分组 | `libs/utils/train_utils.py:make_optimizer` | 直接调用；不再用 `ndim>=2` 自行分组 |
| Warmup/cosine 调度 | `train_utils.py:make_scheduler`、`lr_schedulers.py` | 直接调用；不再使用自写 LambdaLR |
| EMA | `train_utils.py:ModelEma` | 直接创建并调用 `.update(model)` |
| 时间裁剪增强与 collate | `libs/datasets/data_utils.py` | 直接调用 `truncate_feats`、`trivial_batch_collator` 和 worker seed helper |
| 解码、时间坐标反变换 | `libs/modeling/meta_archs.py` | 直接使用官方 inference/postprocessing |
| Soft-NMS / segment voting | `libs/utils/nms.py`、`csrc/nms_cpu.cpp` | 原样调用官方 Python/C++，没有自写 hard-NMS 替代 |

本项目训练循环只负责独立 CLI、tqdm、保存/恢复 epoch 内进度。每个 batch 一次更新，按官方顺序执行 backward → clip → optimizer → scheduler → EMA。已删除 `accumulation_steps` 参数。默认不用 AMP；用户主动开启 AMP 属于需记录的运行变更。

## 参数：来源及理由

参考 `configs/thumos_i3d.yaml` 与 `libs/core/config.py` 合并后的配置：

| 参数 | 当前三组共同设置 | 依据 |
| --- | --- | --- |
| embedding/FPN/head 宽度 | 512 | 官方默认 |
| backbone_arch | `(2,2,5)`，6 个时间尺度 | 官方默认 |
| 局部注意力窗口 | 19 个特征点 | 官方 THUMOS 配方；时间跨度随本数据 stride 改变，不宣称最优 |
| regression_range | 官方六级配置 | 单位为特征网格，不是秒；需披露其在 2 秒 stride 下的物理跨度 |
| AdamW LR / weight decay | `1e-4 / 0.05` | 官方 THUMOS 配方，作为未调优的初始方案 |
| batch | 默认全局 2；多卡需显式选择能被卡数整除的全局 batch | 保持官方全局 batch 语义；每个 batch 更新一次，无梯度累积 |
| warmup / 后续训练 | 5 + 30 = 35 epoch | 与官方 train.py 对 `epochs` 的定义相同 |
| 梯度裁剪 / loss normalizer 初值 | 1.0 / 100 | 官方 THUMOS train_cfg |
| EMA decay | 0.999 | 官方 ModelEma 默认 |
| NMS | 官方 Soft-NMS，其余 test_cfg 继承官方 | 去掉此前自写硬 NMS 的配置 |
| seed / workers | 42 / 0 | 本项目固定复现与内存/续跑设置，不冒充官方最优参数 |

原 MSLoc-PG 使用 RGB 端到端微调，ActionFormer 这里使用冻结后的缓存特征。两者的优化对象不同，所以不应机械地把 PG 的 `lr=1e-6`、batch=32 等套给定位器。三组 ActionFormer 应使用相同优化配方、数据、特征来源、预算及评测。与原 PG 比较时如需严格预算公平，应另外报告相同训练步数/计算预算的对照，不能仅凭 epoch 相同就声称预算相同。

## 必须按 TASLE 设置的数据接口

- 输入是每个完整视频的 `[D,T]` 特征，而不是 RGB。单路 `D=150528`，拼接 `D=301056`；这属于数据输入适配，不将 THUMOS 的 I3D `D=2048` 写死。
- 类别为 `fake` 一个前景类别（label=0），真实区域为隐式背景；纯真实视频用空目标张量保留，官方损失支持该输入。真实视频调用官方裁剪时设 `has_action=False`，避免要求不存在的正片段。
- 使用原来的 8 fps 帧目录、2 秒窗口、8 张采样帧和 2 秒 stride。官方字段为 `fps=8`、`feat_num_frames=16`、`feat_stride=16`，**不是 THUMOS 的约 30 fps、16 帧 clip / 4 帧 stride**。
- 转换严格对应官方数据加载式：`segment_grid=seconds*fps/feat_stride - 0.5*feat_num_frames/feat_stride`。预测使用官方反变换。
- `max_seq_len` 默认从训练集实际 T 推导，不照搬 THUMOS 的 2304。保留原六尺度、窗口 19 时，官方整除因子是 `32×18=576`；训练最长 T=62 或 248 均向上补到 576。只用训练长度决定 padding。推理仍由官方代码处理动态长度。
- 短 GT 没有网格内部点时可能无法获得正样本，记录在 `supervision_report.json`；这是 2 秒时间分辨率的实际限制，不通过改标签来掩盖。若补充 0.5 秒 stride，三组统一重提并单列该协议。
- NMS 内部 `min_score=0.001` 继承官方候选筛选。最终论文 F1 的 `score_threshold=0.5` 是本项目事先固定的评测规则，**不是官方为 TASLE 推荐的阈值**。两者分开记录；无验证集时不能用测试集选择阈值、轮数或超参数。

不使用验证集，固定训练预算结束后输出最终 EMA。这里还保留本项目的 `F1_Det/F1_Loc/Det_Acc`，不把 THUMOS 的 mAP 评测直接当作本数据集协议。

## 验证范围与运行前提

`verify_official.py` 用官方 `train_utils.train_one_epoch` 作为数值参考，检查原生模型类型、一步更新后的参数/EMA/学习率以及时间坐标对应；模型采用小维度以便 CPU/GPU 接口检查，不代表论文实验的宽度/层数。三组正式配置单独核验同一配方。

当前 Windows 主机可以验证源码、模型训练和数值一致性，不能编译官方要求的 Linux C++ NMS。NMS 测试明确跳过，不能据此宣称服务器端到端已验证。服务器先执行：

```bash
cd MSLoc-AF/vendor/actionformer/libs/utils
python setup.py build_ext --inplace
cd ../../../..
python verify_official.py --device cuda:0
```

该验证命令默认要求官方 NMS，并执行 Soft-NMS 检查；缺扩展直接失败，不会静默切换其他 NMS。`--allow-missing-nms` 只用于本地源码/数学检查，输出 `complete:false`。

此前的 256 维/四尺度/硬 NMS 实验只能视为早期改动版，不能冒充此版本。本版新训练保存在 `results_official_ddp/`，旧训练断点不作为本版多卡实验的续跑起点。**已有 XCLIP/Mamba 特征缓存无需重提**；原 MSLoc-PG/PR 文件不修改。

## 多卡执行适配

通过 `python train.py --device-ids 0,1,2,3` 启动 DDP，入口内部用 PyTorch spawn 为每张选定 GPU 启动一个进程。GPU ID 的顺序直接映射到 rank，不要求用户设置 CUDA_VISIBLE_DEVICES 或运行 torchrun。进程通过临时 FileStore 初始化通信，退出后自动清理。每个进程接收完整视频的 `[D,T]` 特征，避免普通 DataParallel 对 list 内张量的 C 维做 scatter。模型、损失、优化器、调度器与 EMA 仍直接调用官方代码；DDP 是执行层适配，并非声称逐字复用官方 DataParallel 训练入口。

`--batch-size` / `--train-batch-size` 是所有进程的全局 batch，必须不小于进程数且可整除；每卡 batch=全局 batch/进程数。尾部不足一个全局 batch 的样本按官方 drop_last 丢弃，不填充重复视频。`--lr` 是基础学习率，按官方 train.py 乘以卡数；实际学习率、卡数和每卡 batch 写入 run_config.json。`num_workers` 为每进程 worker 数。

每卡保留官方损失 normalizer、独立随机流；DDP 平均各卡梯度。断点包含每个 rank 的随机状态与 normalizer，以及共享模型、EMA、优化器、调度器、更新步数和 batch 游标。续跑要求相同进程数、全局 batch 和训练配置；默认 num_workers=0 支持精确的 epoch 内续跑。仅 rank 0 写文件、显示进度条。推理导出去除并行包装，可用单卡加载。

推理通过 `python eval.py --device-ids 0,1,2,3,4,5,6,7` 启动独立的模型进程，按完整视频分片，无梯度同步、不补齐或重复视频。每个进程原样调用官方前向、解码和 CPU Soft-NMS；每个视频完成后原子保存到 `video_progress/`。主进程显示总进度，待全部完成后按原标注顺序汇总并调用原评测代码一次。待推理视频少于 GPU 数时只启动所需进程。续跑可改变 GPU 数量，保留已完成视频，仅重新分配未完成视频。

官方 NMS 扩展在主进程启动推理前检查；本次并行改动保留 Soft-NMS，不提供自动替代实现。NMS 二进制加载的临时搜索路径会恢复，避免 spawn 子进程导入时官方 `utils/metrics.py` 遮蔽本项目的 `metrics.py`。

特征提取也支持 `python extract_features.py --device-ids 0,1,2,3,4,5,6,7`。每个进程独立加载同一个原始 PG 权重，按完整视频分片，原样执行图像预处理、窗口/帧采样和 Mamba hook 聚合。`--batch-size` 是每卡前向的窗口数，和训练的全局 batch 语义不同；一次前向同时缓存 xclip/mamba，fusion 仍在训练时拼接。提取不依赖 ActionFormer 的 NMS 扩展。

只有主进程写共享 `manifest.json`。工作进程完成一个视频的两份 `.npy` 后才写 `video_commits/` 完成标记并通知主进程；主进程验证配对形状后更新清单。若父进程在清单提交前退出，重跑时会根据完成标记恢复成对缓存；尚未成对完成的视频重新提取。续跑允许更换卡数或每卡窗口 batch，已缓存视频不重写。旧版已知单卡缓存保持原 spec/特征签名，只追加新视频与本次检查记录，兼容原训练模型所用的特征缓存。
