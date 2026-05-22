# bert-dialogue-suite 架构说明

## 目录结构

```
.
├── core/                  # 通用训练/推理/数据基础设施
├── rejection/             # 拒识模块（已完成）
├── intent_router/         # 意图识别分流（骨架）
├── visual_detector/       # 视觉问题判断（骨架）
├── scripts/               # 全局脚本
└── docs/                  # 全局文档
```

## 设计原则

1. **core 只放通用逻辑**：训练循环、推理器、数据集类、数据工具
2. **各模块只放任务专属逻辑**：标签映射、数据模板、配置文件
3. **零额外依赖**：core 与模块均不依赖 YAML 解析器等额外包
4. **脚本可直接运行**：每个 `train.py` 和 `generate_data.py` 都可通过 `python xxx.py` 直接执行

## 添加新模块的标准流程

1. 复制 `intent_router/` 目录，改名为新模块名
2. 修改 `config.py` 中的 `LABEL2ID` / `ID2LABEL` / `MODEL_NAME`
3. 在 `generate_data.py` 中填充真实模板和生成函数
4. 运行 `python new_module/train.py --help` 验证 CLI 正常
5. 在 `scripts/train_all.py` 的 `MODULES` 列表中加入新模块名（可选）
