# SR Backbone 改动日志

本文档记录本阶段为了将 `StableDiffusionLatentUpscalePipeline` 接入 `StealthDiffusion` 而做的代码改动、改动原因和当前状态。

## 1. 改动目标

这一阶段只做一件事：

- 将原始 `StealthDiffusion` 的 diffusion backbone 从普通 `StableDiffusionPipeline` 替换为 `StableDiffusionLatentUpscalePipeline`
- 保持攻击主框架尽量不变
- 暂时关闭 `ControlVAE`
- 不改损失函数
- 不改优化器
- 不改迭代次数
- 不做 DiffAttack 对比

核心问题是验证：

- 只换 backbone，是否有价值

## 2. 工作目录

所有改动都在独立目录完成：

- `StealthDiffusion_SR`

原始参考仓库未修改：

- `StealthDiffusion`

## 3. 改动文件

本轮实际修改了以下文件：

- `StealthDiffusion_SR/main.py`
- `StealthDiffusion_SR/diff_latent_attack.py`
- `StealthDiffusion_SR/scripts/test_sr_backbone.py`

新增文档：

- `StealthDiffusion_SR/SR_BACKBONE_CHANGELOG_CN.md`

## 4. main.py 的改动

### 4.1 默认模型路径改为本地 SR backbone

将默认 `pretrained_diffusion_path` 改为：

- `/root/gpufree-data/checkpoints/hf_models/sd-x2-latent-upscaler`

目的：

- 避免每次运行时重新走 Hugging Face 在线下载
- 为后续实验提供固定本地模型入口

### 4.2 默认关闭 ControlVAE

将：

- `is_encoder` 默认值从 `1` 改为 `0`

含义：

- 当前阶段默认运行 `without ControlVAE`

### 4.3 增加 backbone 自动识别和加载逻辑 

新增：

- `_infer_pipeline_class()`
- `load_diffusion_pipeline()`

作用：

- 自动识别当前路径对应的是普通 SD pipeline 还是 SR pipeline
- 如果检测到 `model_index.json` 中的 `_class_name` 是 `StableDiffusionLatentUpscalePipeline`，则按 SR backbone 加载
- 否则退回普通 `StableDiffusionPipeline`

### 4.4 统一替换 scheduler

加载完 pipeline 后，不再直接使用原 scheduler，而是统一改为：

- `DDIMScheduler`
- `DDIMInverseScheduler`

原因：

- 原始代码的 inversion 和 denoising 都是按 DDIM 思路写的
- 直接复用 SR pipeline 自带的 Euler scheduler，和现有攻击逻辑不兼容

### 4.5 针对 SR backbone 修正 prediction_type

如果当前 backbone 是 `StableDiffusionLatentUpscalePipeline`，则额外设置：

- `prediction_type = "sample"`

原因：

- `sd-x2-latent-upscaler` 的 UNet 经过预条件化后，更适合按 `pred_original_sample` 来接入 DDIM / DDIMInverse scheduler
- 不这样改会导致 inverse scheduler 直接报错

### 4.6 移除硬编码代理，改为镜像默认值

移除了代码里固定的：

- `http_proxy`
- `https_proxy`

改为：

- `HF_ENDPOINT=https://hf-mirror.com`

目的：

- 减少对某一台代理机器的硬依赖
- 和当前实验环境保持一致

### 4.7 修复 Python 3.8 类型注解兼容问题

将：

- `list[Path]`

改为：

- `List[Path]`

原因：

- 当前环境为 Python 3.8，原写法会报 `type object is not subscriptable`

## 5. diff_latent_attack.py 的改动

这是本轮最核心的改动文件。

### 5.1 增加 backbone 判别函数

新增：

- `is_sr_backbone(model)`

作用：

- 区分当前 backbone 是普通 SD 还是 SR pipeline
- 后续条件构造、UNet 调用方式、scheduler 对接方式都依赖这个判断

### 5.2 统一读取 VAE scaling factor

新增：

- `get_scaling_factor(model)`

并将多个硬编码的：

- `0.18215`
- `0.1825`

替换为动态读取：

- `model.vae.config.scaling_factor`

原因：

- 避免 backbone 切换后 scaling factor 写死造成隐性错误

### 5.3 重写图像编码入口

新增：

- `encode_image_to_latents(image, model, res=...)`

作用：

- 统一将输入图像编码为 latent
- 同时兼顾 VAE dtype 和 UNet dtype

### 5.4 重写 prompt 编码逻辑

新增：

- `encode_prompt_embeddings(model, prompt)`

原因：

- 普通 SD backbone 的 text encoder 输出接口和 SR latent upscaler 的使用方式不完全相同
- SR backbone 不仅需要 token-level text embedding，还需要 pooled text embedding

当前仍然保持空 prompt：

- `prompt = [""] * 2`

所以这里的 text condition 仍然主要是结构占位，而不是引入真实语言语义

### 5.5 新增 SR 所需的图像条件分支

新增：

- `build_sr_image_condition(image, model, res, batch_size)`

作用：

- 将低分辨率输入图像编码到 latent 空间
- 再构造 SR backbone 所需的 `image_cond`

原因：

- `StableDiffusionLatentUpscalePipeline` 不是只依赖 text condition 的标准 SD
- 它本质上是 image-conditioned diffusion
- 若不构造图像条件，UNet 输入结构不完整

### 5.6 新增统一 context 构造逻辑

新增：

- `build_context(...)`

普通 SD 情况下，context 只需要：

- `encoder_hidden_states`

SR backbone 情况下，还需要补充：

- `timestep_condition`
- `image_cond`
- pooled text features

### 5.7 新增统一的 UNet 预测封装

新增：

- `get_model_prediction(model, latents, context, t, guidance_scale)`

作用：

- 将普通 SD 和 SR backbone 的 UNet 前向封装到一个入口中

其中 SR backbone 的特殊处理包括：

- UNet 输入是 `scaled_latents + image_cond`
- 需要 `timestep_cond`
- UNet 输出通道数为 `5`，最后一个方差通道需要裁掉
- 再按 latent upscaler 的预条件化公式还原为 `pred_original_sample`

这是本次 backbone 替换的核心适配点。

### 5.8 将 DDIM inversion 改为 scheduler 驱动

原先 inversion 是手写公式，默认假设：

- 模型输出是 `epsilon`

现在改为：

- `DDIMInverseScheduler` 驱动

原因：

- SR backbone 的输出和普通 SD 不同
- 继续套原始手写公式容易直接错位

### 5.9 修复 inversion 从 t=0 开始导致的 NaN

将 inversion 循环从：

- `timesteps[:-1]`

调整为：

- `timesteps[1:]`

原因：

- 如果 inverse step 从 `t=0` 开始，会在内部出现无效计算，产生 NaN

### 5.10 修复 init_latent 对通道数的错误假设

将 `init_latent()` 中的扩展维度从：

- `model.unet.in_channels`

改为：

- `model.vae.config.latent_channels`

原因：

- 普通 SD 的 UNet 输入通道是 `4`
- 但 SR backbone 的 UNet 输入通道是 `8`
- 这 8 通道里有 4 通道来自攻击 latent，另外 4 通道来自 image condition
- 如果直接按 `unet.in_channels=8` 去扩展攻击 latent，会把 latent 自己扩成 8 通道，完全错误

### 5.11 为 SR backbone 引入 diffusion_res

新增：

- `diffusion_res`

当前逻辑：

- classifier / eval 仍用 `res=224`
- 但 SR backbone 内部 diffusion 改用 `diffusion_res=256`

原因：

- `224 -> latent 28x28`
- 对 `sd-x2-latent-upscaler` 的 UNet 结构来说，进一步下采样和上采样时会出现 skip feature 尺寸不对齐
- `256 -> latent 32x32` 才能稳定通过网络

因此当前策略是：

- diffusion 在 256 上跑
- loss 和分类器在 224 上算
- 最终保存图像保留 diffusion 输出分辨率

### 5.12 修复分类器输入 dtype 问题

将最终送入 detector 的图像显式转为：

- `float32`

原因：

- 分类器参数是 float32
- diffusion 侧常用 float16
- 不转换会报 half/float 类型冲突

## 6. test_sr_backbone.py 的改动

新增脚本：

- `StealthDiffusion_SR/scripts/test_sr_backbone.py`

用途：

- 单独验证 SR backbone 是否能被加载
- 验证其是否能处理 GenImage 中的图像
- 尽量先把 backbone 问题和攻击问题分开

主要改动：

- 默认模型路径改成本地目录
- 如果传入本地路径，则自动使用 `local_files_only=True`

说明：

- 在当前环境下，官方 `pipe(...)` 直接走 Euler scheduler 仍有兼容问题
- 因此后续正式实验不依赖这个脚本作为主入口
- 正式实验走的是已经适配过的 `main.py + diff_latent_attack.py`

## 7. 运行中遇到并修复的问题

### 问题 1：官方 SR pipeline 直接调用失败

现象：

- 直接 `pipe(...)` 时在 Euler scheduler 的 `set_timesteps` 处报错

处理：

- 正式实验不依赖官方推理入口
- 改为在现有攻击框架内，用 DDIM scheduler 统一接管

### 问题 2：SR backbone 在 224 分辨率下出现尺寸不匹配

现象：

- UNet 内部 skip connection 拼接时报尺寸错误

处理：

- diffusion 内部改为 256
- 保持分类器和指标仍在 224 上比较

### 问题 3：inverse scheduler 报 prediction_type 错误

现象：

- `prediction_type given as original_sample must be one of epsilon/sample/v_prediction`

处理：

- 对 SR backbone 的 DDIM / DDIMInverse scheduler 配置为 `prediction_type="sample"`

### 问题 4：DDIM inversion 产生 NaN

现象：

- inversion 第一轮就出现 NaN latent

处理：

- 不从 `t=0` 开始做 inverse step

### 问题 5：分类器输入类型冲突

现象：

- `torch.cuda.HalfTensor` 和 `torch.cuda.FloatTensor` 不一致

处理：

- 最终送入分类器前统一转 `float32`

## 8. 当前结果状态

目前已经完成：

- 单图 smoke test 跑通
- 小批量自检跑通：每个 fake source 取 1 张图，共 7 张
- 输出目录结构已兼容现有 dataroot 评估流程
- 可计算：
  - detector 成功率相关指标
  - LPIPS
  - Fourier 频谱指标

小批量自检阶段观察到的现象：

- 工程链路已打通
- SR backbone 版本已经可以稳定出图
- 但当前视觉质量明显偏差较大
- 因此全量实验结果大概率会说明“仅替换 backbone 且其余攻击超参数完全不动”时，质量代价较高

这和本阶段目标并不冲突，因为这一阶段本来就是做可行性判断。

## 9. 当前未改动的部分

以下部分保持原样：

- 损失函数
- 优化器
- 迭代次数
- ControlVAE 结构本身
- eval 口径
- surrogate 分类器配置

## 10. 下一步建议

如果继续当前阶段，建议按顺序做：

1. 继续完成全量 `SR backbone, without ControlVAE` 实验
2. 使用和 SD baseline 相同口径整理：
   - ASR
   - LPIPS
   - Fourier 频谱 L2
3. 与你正在跑的 `SD backbone, without ControlVAE` 做首轮对比

如果 SR 版本结果明显更差，再回头判断：

- 是 backbone 本身不适合
- 还是因为“完全不调参直接平移”导致它没有发挥空间

