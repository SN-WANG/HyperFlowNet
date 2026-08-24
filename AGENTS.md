# HyperFlowNet 项目说明(Agent 速览)

## 项目是什么

面向 CMAME 投稿的研究代码库:研究"自回归神经算子预测间断物理场(激波、界面、前沿)时为何被抹平"的机理,并提出路径级流匹配修复 HyperFlowNet。论文规划见 `outline.md`(同仓库目录之外,位于 `/Users/wsn/Documents/hyflow-paper/outline.md`)。

## 一句话定位

抹平根因是训练目标层面的条件均值吸引子:逐点平方损失的最优解把位置不确定的前沿平均成斜坡;HyperFlowNet 在概率路径/耦合层用"前沿搬运"代替"振幅平均",推理为确定性概率流 ODE。

## 仓库结构

```text
HyperFlowNet/
├── main.py                  # generate / train / evaluate / mechanism / benchmark
├── config.yaml              # 全部实验配置(数据/模型/训练/评测/机理/benchmark)
├── trainer.py               # 训练目标 + Sinkhorn 配对 + 路径族 + rollout 评测
├── models/
│   ├── __init__.py          # make_model 注册表
│   ├── blocks.py            # 1D/2D 共用块
│   ├── operators.py         # 基线:FNO DeepONet U-Net ViT WNO UWNO
│   ├── velocity.py          # FlowUNet/FlowNO 条件速度网络(CFM/OT-CFM/HyperFlowNet 共用)
│   ├── pde_refiner.py       # PDE-Refiner 基线
│   └── hyperflownet.py      # HyperFlowNet(五组件)
├── data/
│   ├── synthetic.py         # 机理合成数据(阶跃族/条件期望配对/FM 配对)
│   ├── burgers.py           # 1D Burgers
│   ├── euler.py             # 1D Sod + 2D Euler Riemann
│   ├── neptuna.py           # Neptuna 加载器(bubble/droplet)
│   └── datasets.py          # 注册表、归一化、npz 持久化
└── utils/                   # 来自 wsnet:hue_logger scaler seeder sweeper + metrics mechanism plotting
```

## 代码约定

- 纯 PyTorch,禁用 jax/equinox/optax(重构目标之一,已 grep 验收)。
- 张量布局 channel-first:`(B, C, *S)`,S 为空间维(1D 或 2D);时序数据 `(N, T+1, C, *S)`。
- 模型统一接口:field-to-field,`forward(x) -> x`,输出形状与输入一致。
- 命名:snake_case 函数/变量、PascalCase 类;公共类/函数/method 写 docstring 并标注张量形状。
- 每个 Python 文件两行头:`# 简短描述` + `# Author: Shengning Wang`。
- 布局紧凑,行宽 ≤120 列;不做防御式编程。

## 运行方式

```bash
pip install -r requirements.txt
python main.py generate --config config.yaml    # 生成机理数据到 data/
python main.py train --config config.yaml --model FNO --objective mse
python main.py train --config config.yaml --model HyperFlowNet --objective hyflow
python main.py evaluate --config config.yaml --checkpoint runs/exp/ckpt.pt
python main.py mechanism --config config.yaml   # 机理预言 1/2/4 实验
python main.py benchmark --config config.yaml   # 按 config 的 baselines 清单批量跑
```

开发解释器 `~/pyenv/bin/python`(3.12);训练服务器单卡 RTX 5090,config 里 `training.amp: true` 启用 bf16。

## Baselines(九件)

FNO、DeepONet、U-Net、ViT、WNO、UWNO(operators.py,MSE 训练);CFM、OT-CFM(velocity.py,差异在 trainer 的耦合);PDE-Refiner(pde_refiner.py)。HyperFlowNet 为方法本体。机理跨架构实验(FNO/WNO/UWNO/UNet/ViT)直接复用主表模型。

## 机理预言与路径族

- 预言 1:条件均值(输出=经验条件平均,跨架构)。
- 预言 2:2.56σ 标度律(宽度 ∝ σ、与跳跃高度 J 无关、跨架构)。
- 预言 3:自回归放大(输运误差占比升、宽度沿 rollout 单调不减、漂移 ∝√T)。
- 预言 4:直路径 FM 终点宽度随 σ 按 2.56σ 增长,搬运路径显著更窄。
- 路径族:(a) 直路径独立耦合 = 标准 CFM;(b) 直路径 + OT 耦合 = OT-CFM;(c) 搬运路径 = HyperFlowNet。**无路径族 d,不实现残差修正**。

## 数据

- 机理数据(合成阶跃族、Burgers、Sod、2D Euler Riemann)本地生成,存 `data/<name>_<grid>.npz`,config 用相对路径。
- Neptuna 工程数据:config 绝对路径 `D:/data/Neptuna/multi_bubble` 与 `D:/data/Neptuna/multi_droplet`;目录下直接是 `train.h5`、`test.h5`、`metadata.json`、`parameters.csv`(无 256x256 子文件夹)。命名统一用 bubble / droplet,不用 sabw / sdba。加载逻辑参照 Neptuna 原仓库 `utils/load_data.py`(组名解析条件参数、z-normalization、log-transform 通道)。

## 关键决策记录

- 工程层只用 Neptuna(2D-SABW=bubble、2D-SDBA=droplet),双主评测,256x256 全量;TransportBench 已排除(稳态非时序)。
- 标题路线:"On the smearing of discontinuities in autoregressive neural operators and its mitigation via transport-based flow matching"(residual correction 字样彻底移除)。
- 论文配套文档:`/Users/wsn/Documents/hyflow-paper/outline.md`(论文大纲)、`/Users/wsn/Documents/hyflow-paper/机理与模型设计_详解.md`(设计推演,仅参考)、`/Users/wsn/Documents/hyflow-paper/discontinuous_field_prediction_review.md`(文献综述,Introduction 文献骨架)。

## 评测指标(分层)

逐点层(Rel-L2/L1,次要);间断层(10-90% 前沿宽度、前沿位置误差、TV 比、形状/输运分解);时序层(误差 vs rollout 步数 T、固定时域误差、NFE-精度)。见 `utils/metrics.py`。
