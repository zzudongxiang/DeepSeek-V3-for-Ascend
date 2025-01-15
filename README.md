# DeepSeek-V3 for Ascend

- 关于`DeepSeek-V3`的文档和相关的权重文档请参考: [DeepSeek-V3](./docs/README.md)和[Weights](./docs/README_WEIGHTS.md)

- 关于在Ascend平台上准备运行环境的操作请参考: [PREPARE](./docs/PREPARE.md)

## 1. 运行程序

需要提前根据并行参数执行`python inference/convert.py`转换权重

**单卡环境** : 参考[`scripts/single-gpu-inference.sh`](./scripts/single-gpu-inference.sh)文件中的设置

**多卡环境** : 参考[`scripts/multi-gpu-inference.sh`](./scripts/multi-gpu-inference.sh)文件中的设置

## 2. 参数设置

常规的模型参数参考[`configs`](./inference/configs)目录下的文件

运行时参数可以使用`python inference/generate.py -h`进行查看

如果需要输出模型的算力、显存占用等信息请将[`model.py`](./inference/model.py)文件中的`disable_writer`变量设置为`False`

## 3. 调试程序

已配置VSCode的Debugger，可以参考[`launch.json`](./.vscode/launch.json)文件中的设置进行调试
