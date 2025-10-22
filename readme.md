
# YuYu Hakusho AI

这个项目使用强化学习训练AI来玩《幽游白书》(YuYu Hakusho)游戏。基于Stable Baselines3和OpenAI Retro实现，能够训练一个AI代理自动游玩Genesis平台上的《幽游白书》游戏。

## 项目结构

```
YuYuHakusho/
├── trained_models/           # 训练好的模型文件
├── logs/                    # 训练日志
├── custYuYuHakushoWrapper.py # 自定义环境包装器
├── test.py                  # 测试模型脚本
├── train.py                 # 训练模型脚本
└── README.md
```

## 环境配置

### 必需依赖

- Python 3.8.20
- OpenAI Retro
- Stable Baselines3
- PyTorch (带CUDA支持更佳)

### 安装步骤

1. 安装OpenAI Retro:
```bash
pip install gym-retro
```

2. 安装Stable Baselines3:
```bash
pip install stable-baselines3
```

3. 导入游戏ROM:
需要获取《幽游白书》Genesis版本的ROM文件，并使用以下命令导入:
```bash
python -m retro.import ROM_PATH
```

## 使用方法

### 训练模型

运行训练脚本开始训练AI:
```bash
python train.py
```

训练参数可在 [train.py](file://d:\Work\PrivateWork\JAPrograms\PythonWork\YuYuHakusho\train.py) 中调整:
- [Total_Timesteps](file://d:\Work\PrivateWork\JAPrograms\PythonWork\YuYuHakusho\train.py#L11-L11): 总训练步数 (默认3,000,000)
- [NUM_ENV](file://d:\Work\PrivateWork\JAPrograms\PythonWork\YuYuHakusho\train.py#L12-L12): 并行环境数量 (默认16)
- 学习率调度器
- PPO超参数

### 测试模型

使用预训练模型测试AI表现:
```bash
python test.py
```

在 `test.py` 中可以配置:
- `MODEL_NAME`: 要加载的模型名称
- `RESET_ROUND`: 是否在每轮结束后重置
- `RENDERING`: 是否渲染游戏画面
- `NUM_EPISODES`: 测试回合数

## 核心组件

### 训练脚本 (train.py)

主要特性:
- 使用PPO算法和CNN策略
- 多进程并行训练 (SubprocVecEnv)
- 线性学习率和裁剪范围调度
- TensorBoard日志记录
- 定期保存检查点模型
- 训练日志输出到文件

关键参数:
- `n_steps`: 512
- `batch_size`: 512
- `n_epochs`: 4
- `gamma`: 0.94

### 测试脚本 (test.py)

功能:
- 加载预训练模型
- 运行多轮游戏测试
- 统计胜率和平均奖励
- 实时渲染游戏画面

### 自定义环境包装器 (custYuYuHakushoWrapper)

提供针对该游戏的定制化环境处理:
- 奖励机制设计
- 回合结束检测
- 观测空间处理
- 动作空间过滤

## 模型说明

训练过程中会生成以下模型文件:
- 检查点模型: 定期保存的中间模型
- 最终模型: [ppo_yuyuhakusho_final.zip](file://d:\Work\PrivateWork\JAPrograms\PythonWork\YuYuHakusho\YuYuHakusho-Genesis\train\ppo_yuyuhakusho_final.zip)

## 训练细节

- 使用 `retro.Actions.FILTERED` 动作空间
- 图像观测空间 (`retro.Observations.IMAGE`)
- 线性学习率调度 (2.5e-4 → 2.5e-6)
- 线性裁剪范围调度 (0.15 → 0.025)
- 每4次检查点间隔保存一次模型

## 故障排除

常见问题及解决方案:

1. **Action spaces do not match** 错误:
   确保训练和测试时使用相同版本的环境包装器和动作空间设置

2. **ROM未找到**:
   确认已正确导入游戏ROM且路径正确

3. **CUDA内存不足**:
   减少并行环境数量([NUM_ENV](file://d:\Work\PrivateWork\JAPrograms\PythonWork\YuYuHakusho\train.py#L12-L12))或降低`batch_size`

4. **训练效果不佳**:
   尝试调整超参数或增加训练时间

## 许可证

本项目仅供学习研究使用。使用者需确保遵守当地法律法规，并自行承担相关法律责任。游戏ROM需要通过合法渠道获取。
```
