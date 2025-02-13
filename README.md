# DeepSeek-V3 for Ascend

- 关于`DeepSeek-V3`的文档和相关的权重文档请参考: [DeepSeek-V3](./docs/README.md)和[Weights](./docs/README_WEIGHTS.md)

## 1 准备Pytorch

### 1.1 准备Conda环境

```bash
# 创建deepseek环境
conda create -n deepseek python=3.9
# 激活环境
conda activate deepseek
# 安装依赖的三方包
pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

如果出现以下`错误`，则可以忽略，Python3.9自带这些模块

```log
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
op-compile-tool 0.1.0 requires getopt, which is not installed.
op-compile-tool 0.1.0 requires inspect, which is not installed.
op-compile-tool 0.1.0 requires multiprocessing, which is not installed.
```

### 1.2 安装`torch_npu`

```bash
# 确认本地安装的pytorch版本
pip list | grep torch
# 到gitee中下载对应版本的torch_npu(以aarch64+py39+torch2.1.0为例)
wget https://gitee.com/ascend/pytorch/releases/download/v6.0.rc2-pytorch2.1.0/torch_npu-2.1.0.post6-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
# 其他版本的`torch_npu`插件下载请参考：https://gitee.com/ascend/pytorch/releases
pip install ./torch_npu-*.whl
```

### 1.3 验证安装结果

```bash
python -c "import torch; import torch_npu; print(torch_npu.__version__)"
```

正常的输出

```log
/home/.../site-packages/torch_npu/utils/collect_env.py:59: UserWarning: Warning: The /usr/local/Ascend/ascend-toolkit/latest owner does not match the current owner.
  warnings.warn(f"Warning: The {path} owner does not match the current owner.")
/home/.../site-packages/torch_npu/utils/collect_env.py:59: UserWarning: Warning: The /usr/local/Ascend/ascend-toolkit/8.0.RC2.10/aarch64-linux/ascend_toolkit_install.info owner does not match the current owner.
  warnings.warn(f"Warning: The {path} owner does not match the current owner.")
2.1.0.post6
```

## 2 安装Apex

### 2.1 获取apex包

```bash
# 建议在独立的路径~/llm中安装以下包
mkdir -p ~/llm
cd ~/llm
git clone https://gitee.com/ascend/apex.git
```

### 2.2 原生apex下载加速（可选）

```bash
# 切换到已经下载好的gitee仓库中
cd ~/llm/apex
# 手动下载apex镜像或者通过ftp等方法上传到服务器的apex路径中
# 上传的路径一般为~/llm/apex/apex
git clone https://github.com/NVIDIA/apex.git
# 修改apex/scripts/build.sh的脚本，注释掉以上内容（102~104行）
```

### 2.3 编译apex包

```bash
# 安装apex加速包
cd ~/llm/apex
bash scripts/build.sh --python=3.9
# 如果中间提示是否覆盖，则需要按照上一步的方式修改apex/scripts/build.sh
# 注释掉：patch -p1 <npu.patch
```

### 2.4 安装apex

```bash
# 切换到apex/dist路径下
cd ~/llm/apex/apex/dist/
# 使用pip安装
pip uninstall apex -y
pip install --upgrade apex-*.whl
```

### 2.5 验证安装结果

```bash
pip show apex
```

正常的输出

```log
Name: apex
Version: 0.1+ascend
Summary: PyTorch Extensions written by NVIDIA
Home-page:
Author:
Author-email:
License:
Location: /home/<USER_NAME>/anaconda3/envs/deepseek/lib/python3.9/site-packages
Requires:
Required-by
```

## 3 安装Megatron-LM

### 3.1 获取Megatron-LM

```bash
# 建议在独立的路径~/llm中安装以下包
mkdir -p ~/llm
cd ~/llm
git clone https://github.com/NVIDIA/Megatron-LM.git
# 检出与后续MindSpeed匹配的分支
cd Megatron-LM
git checkout core_r0.7.0
```

### 3.2 安装Megatron-LM

```bash
pip install -e . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

### 3.3 验证安装结果

```bash
pip show megatron_core
```

正常的输出

```log
Name: megatron_core
Version: 0.7.0
Summary: Megatron Core - a library for efficient and scalable training of transformer based models
Home-page: https://github.com/NVIDIA/Megatron-LM/megatron/core
Author: NVIDIA
Author-email: nemo-toolkit@nvidia.com
License: BSD-3
Location: /home/<USER_NAME>/anaconda3/envs/deepseek/lib/python3.9/site-packages
Editable project location: /home/<USER_NAME>/llm/Megatron-LM
Requires:
Required-by:
```

## 4 安装MindSpeed

### 4.1 获取MindSpeed

```bash
# 建议在独立的路径~/llm中安装以下包
mkdir -p ~/llm
cd ~/llm
git clone https://gitee.com/ascend/MindSpeed.git
# 切换到与Megatron-LM版本一致的分支
cd MindSpeed
git checkout 4045864e6df
```

### 4.2 安装MindSpeed

```bash
# 切换到MindSpeed路径
cd ~/llm/MindSpeed
pip install -e . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
# 每次使用前需要添加以下命令或将其添加至~/.bashrc文件中
export PYTHONPATH=$PYTHONPATH:/home/<USER_NAME>/llm/Megatron-LM
```

### 4.3 验证安装结果

```bash
pip show mindspeed
```

正常的输出

```log
Name: mindspeed
Version: 0.7.0
Summary: MindSpeed for LLMs of Ascend
Home-page: https://gitee.com/ascend/MindSpeed
Author: Ascend
Author-email:
License: See https://gitee.com/ascend/MindSpeed
Location: /home/<USER_NAME>/llm/MindSpeed
Editable project location: /home/<USER_NAME>/llm/MindSpeed
Requires:
Required-by:
```

**注意**：MindSpeed安装后，安装路径对应的文件不能删除，否则可能导致MindSpeed模块不可用
