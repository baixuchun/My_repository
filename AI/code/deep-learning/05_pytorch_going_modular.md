# 05. PyTorch 模块化

本节回答了“如何将我的笔记本代码转换为 Python 脚本？”这个问题。

为此，我们将把 [04. PyTorch 自定义数据集笔记本](https://www.learnpytorch.io/04_pytorch_custom_datasets/) 中最有用的代码单元转换为一系列 Python 脚本，并保存在名为 [`going_modular`](https://github.com/mrdbourke/pytorch-deep-learning/tree/main/going_modular) 的目录中。

## 什么是模块化？

模块化意味着将笔记本中的代码（来自 Jupyter Notebook 或 Google Colab 笔记本）转换为一系列具有相似功能的不同 Python 脚本。

例如，我们可以将笔记本中的代码从一系列单元转换为以下 Python 文件：

* `data_setup.py` - 用于准备和下载数据的文件（如有需要）。
* `engine.py` - 包含各种训练函数的文件。
* `model_builder.py` 或 `model.py` - 用于创建 PyTorch 模型的文件。
* `train.py` - 用于利用其他文件并训练目标 PyTorch 模型的文件。
* `utils.py` - 用于帮助的实用函数的文件。

> **注意：** 上述文件的命名和布局将根据您的使用情况和代码要求而有所不同。Python 脚本和单个笔记本单元一样具有通用性，意味着你几乎可以为任何功能创建一个脚本。

## 为什么要模块化？

笔记本非常适合快速进行实验和迭代探索。

然而，对于更大规模的项目，您可能会发现 Python 脚本更具可重复性且更易于运行。

虽然这是一个有争议的话题，因为像 [Netflix](https://netflixtechblog.com/notebook-innovation-591ee3221233) 这样的公司已经展示了它们如何将笔记本用于生产代码。

**生产代码** 是为了提供服务给某人或某事而运行的代码。

例如，如果您有一个在线运行的应用，其他人可以访问和使用，那么运行该应用的代码就是 **生产代码**。

像 fast.ai 的 [`nb-dev`](https://github.com/fastai/nbdev) （即笔记本开发）这样的库使得您可以通过 Jupyter 笔记本编写整个 Python 库（包括文档）。

### 笔记本与 Python 脚本的优缺点

两者都有各自的优缺点。

以下是一些主要话题的总结：

|               | **优点**                                            | **缺点**                                      |
| ------------- | --------------------------------------------------- | --------------------------------------------- |
| **笔记本**    | 容易进行实验/快速入门                              | 版本控制可能很困难                           |
|               | 容易共享（例如通过 Google Colab 链接）              | 很难只使用特定的部分                        |
|               | 非常可视化                                         | 文字和图形可能会干扰代码                    |

|                    | **优点**                                              | **缺点**                                          |
| ------------------ | ----------------------------------------------------- | ------------------------------------------------- |
| **Python 脚本**    | 可以将代码打包在一起（节省在不同笔记本中重复编写相似代码） | 实验不是那么直观（通常需要运行整个脚本而不是单个单元）|
|                    | 可以使用 Git 进行版本控制                            |                                                   |
|                    | 许多开源项目使用脚本                                  |                                                   |
|                    | 更大的项目可以在云供应商上运行（而笔记本不那么支持） |                                                   |

### 我的工作流程

我通常在 Jupyter/Google Colab 笔记本中启动机器学习项目，以便快速进行实验和可视化。

然后，当我有了可行的代码时，我会将最有用的代码片段移到 Python 脚本中。

<img src="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/05-my-workflow-for-experimenting.png" alt="一种可能的机器学习代码编写工作流程，从 jupyter 或 google colab 笔记本开始，然后在工作正常后转到 Python 脚本。" width=1000/>

*编写机器学习代码有许多可能的工作流程。有些人喜欢从脚本开始，其他人（像我）则喜欢先从笔记本开始，稍后再转到脚本。*

### PyTorch 在实际应用中的情况

在你的学习过程中，你会看到很多基于 PyTorch 的机器学习项目的代码库提供了如何以 Python 脚本形式运行 PyTorch 代码的说明。

例如，你可能会被指示在终端/命令行中运行以下代码来训练模型：

```
python train.py --model MODEL_NAME --batch_size BATCH_SIZE --lr LEARNING_RATE --num_epochs NUM_EPOCHS
```

<img src="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/05-python-train-command-line-annotated.png" alt="用不同超参数训练 PyTorch 模型的命令行调用" width=1000/> 

*在命令行中运行一个 PyTorch `train.py` 脚本，设置不同的超参数。*

在这种情况下，`train.py` 是目标 Python 脚本，它可能包含用于训练 PyTorch 模型的函数。

`--model`、`--batch_size`、`--lr` 和 `--num_epochs` 是所谓的参数标志。

你可以根据需要设置这些标志，如果它们与 `train.py` 兼容，它们将生效，否则会报错。

例如，如果我们想训练笔记本 04 中的 TinyVGG 模型 10 个周期，批大小为 32，学习率为 0.001，可以运行以下命令：

```
python train.py --model tinyvgg --batch_size 32 --lr 0.001 --num_epochs 10
```

你可以在 `train.py` 脚本中设置任意数量的这些参数标志来满足你的需求。

PyTorch 博客中的一篇文章使用这种风格来训练最先进的计算机视觉模型。

<img src="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/05-training-sota-recipe.png" alt="PyTorch 训练最先进计算机视觉模型的命令行脚本" width=800/>

*PyTorch 命令行训练脚本，用于训练最先进的计算机视觉模型，使用 8 个 GPU。来源：[PyTorch 博客](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/#the-training-recipe)。*

## 我们要讲解的内容

本节的主要概念是：**将有用的笔记本代码单元转换为可重用的 Python 文件**。

这样做将帮助我们避免重复编写相同的代码。

本节有两个笔记本：

1. [**05. Going Modular: Part 1 (cell mode)**](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_cell_mode.ipynb) - 这个笔记本作为传统的 Jupyter Notebook/Google Colab 笔记本运行，是 [04 笔记本](https://www.learnpytorch.io/04_pytorch_custom_datasets/) 的精简版本。
2. [**05. Going Modular: Part 2 (script mode)**](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_script_mode.ipynb) - 这个笔记本与第一个相同，但新增了将每个主要部分转换为 Python 脚本（例如，`data_setup.py` 和 `train.py`）的功能。

本文中的文本侧重于代码单元 05. Going Modular: Part 2 (script mode)，即顶部带有 `%%writefile ...` 的单元。

### 为什么有两个部分？

因为有时学习一件事最好的方法是看到它与其他事物的*不同之处*。

如果你将两个笔记本并排运行，你会看到它们的差异，关键的学习点就在那里。

<img src="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/05-notebook-cell-mode-vs-script-mode.png" alt="运行 cell 模式笔记本与 script 模式笔记本的对比" width=1000/>

*将两个笔记本并排运行。你会注意到 **脚本模式笔记本有额外的代码单元**，用于将 cell 模式笔记本中的代码转换为 Python 脚本。*

### 我们的目标

本节结束时，我们希望实现两件事：

1. 能够通过一行命令在命令行中训练我们在笔记本 04 中构建的模型（Food Vision Mini）：`python train.py`。
2. 一个可重用 Python 脚本的目录结构，例如：

```
going_modular/
├── going_modular/
│   ├── data_setup.py
│   ├── engine.py
│   ├── model_builder.py


│   ├── train.py
│   └── utils.py
├── models/
│   ├── 05_going_modular_cell_mode_tinyvgg_model.pth
│   └── 05_going_modular_script_mode_tinyvgg_model.pth
└── data/
    └── pizza_steak_sushi/
        ├── train/
        │   ├── pizza/
        │   │   ├── image01.jpeg
        │   │   └── ...
        │   ├── steak/
        │   └── sushi/
        └── test/
            ├── pizza/
            ├── steak/
            └── sushi/
```

### 需要注意的事项

* **文档字符串** - 编写可重复且易于理解的代码很重要。考虑到这一点，我们将要放入脚本中的每个函数/类都采用了 Google 的 [Python 文档字符串风格](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods)。
* **脚本顶部的导入** - 由于我们将要创建的所有 Python 脚本可以视为一个小程序，因此所有脚本都需要在脚本开头导入所需的模块，例如：

```python
# 导入 train.py 所需的模块
import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms
```

## 0. 单元模式与脚本模式

单元模式笔记本，如 [05. Going Modular Part 1 (cell mode)](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_cell_mode.ipynb)，是正常运行的笔记本，每个单元要么是代码单元，要么是 Markdown 单元。

脚本模式笔记本，如 [05. Going Modular Part 2 (script mode)](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_script_mode.ipynb)，与单元模式笔记本非常相似，然而，其中的许多代码单元可能会被转换成 Python 脚本。

> **注意：** 你不必通过笔记本来创建 Python 脚本，你可以直接通过集成开发环境（IDE），例如 [VS Code](https://code.visualstudio.com/) 来创建它们。将脚本模式笔记本作为本节的一部分，仅是为了展示从笔记本到 Python 脚本的转换方式之一。

## 1. 获取数据

在每个 05 笔记本中，获取数据的过程与 [笔记本 04](https://www.learnpytorch.io/04_pytorch_custom_datasets/#1-get-data) 中的相同。

通过 Python 的 `requests` 模块调用 GitHub 来下载 `.zip` 文件并解压。

```python 
import os
import requests
import zipfile
from pathlib import Path

# 设置数据文件夹路径
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# 如果图像文件夹不存在，则下载并准备它...
if image_path.is_dir():
    print(f"{image_path} 目录已存在。")
else:
    print(f"未找到 {image_path} 目录，正在创建...")
    image_path.mkdir(parents=True, exist_ok=True)
    
# 下载 pizza、steak、sushi 数据
with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print("正在下载 pizza、steak、sushi 数据...")
    f.write(request.content)

# 解压 pizza、steak、sushi 数据
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("正在解压 pizza、steak、sushi 数据...") 
    zip_ref.extractall(image_path)

# 删除 zip 文件
os.remove(data_path / "pizza_steak_sushi.zip")
```

这样我们就会得到一个名为 `data` 的文件夹，其中包含一个名为 `pizza_steak_sushi` 的子目录，里面存放着 pizza、steak 和 sushi 的图像，采用标准的图像分类格式。

```
data/
└── pizza_steak_sushi/
    ├── train/
    │   ├── pizza/
    │   │   ├── train_image01.jpeg
    │   │   ├── test_image02.jpeg
    │   │   └── ...
    │   ├── steak/
    │   │   └── ...
    │   └── sushi/
    │       └── ...
    └── test/
        ├── pizza/
        │   ├── test_image01.jpeg
        │   └── test_image02.jpeg
        ├── steak/
        └── sushi/
```

## 2. 创建数据集和数据加载器 (`data_setup.py`)

获取数据后，我们可以将其转换为 PyTorch `Dataset` 和 `DataLoader`（一个用于训练数据，一个用于测试数据）。

我们将有用的 `Dataset` 和 `DataLoader` 创建代码转换为一个名为 `create_dataloaders()` 的函数。

然后我们使用 `%%writefile going_modular/data_setup.py` 将其写入文件。

```python title="data_setup.py"
%%writefile going_modular/data_setup.py
"""
包含用于创建 PyTorch DataLoader 的功能，
用于图像分类数据。
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  """创建训练和测试 DataLoader。

  接收训练和测试目录路径，将它们转换为 PyTorch Datasets，
  然后转换为 PyTorch DataLoaders。

  参数：
    train_dir: 训练目录的路径。
    test_dir: 测试目录的路径。
    transform: 要应用于训练和测试数据的 torchvision 变换。
    batch_size: 每个 DataLoader 批次的样本数量。
    num_workers: 每个 DataLoader 的工作进程数量。

  返回：
    一个元组 (train_dataloader, test_dataloader, class_names)。
    其中 class_names 是目标类别的列表。
    示例用法：
      train_dataloader, test_dataloader, class_names = \
        create_dataloaders(train_dir=path/to/train_dir,
                           test_dir=path/to/test_dir,
                           transform=some_transform,
                           batch_size=32,
                           num_workers=4)
  """
  # 使用 ImageFolder 创建数据集
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # 获取类别名称
  class_names = train_data.classes

  # 将图像转换为数据加载器
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,  # 测试数据不需要打乱
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names
```

如果我们想创建 `DataLoader`，现在可以通过以下方式使用 `data_setup.py` 中的函数：

```python
# 导入 data_setup.py
from going_modular import data_setup

# 创建训练/测试 dataloader 并获取类别名称列表
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(...)
```

## 3. 创建模型 (`model_builder.py`)

在过去的几个笔记本中（笔记本 03 和 笔记本 04），我们多次构建了 TinyVGG 模型。

因此，将模型放入文件中，以便我们可以反复使用是很有意义的。

让我们将 `TinyVGG()` 模型类放入一个脚本中，并使用 `%%writefile going_modular/model_builder.py` 来写入：

```python title="model_builder.py"
%%writefile going_modular/model_builder.py
"""
包含 PyTorch 模型代码，用于实例化 TinyVGG 模型。
"""
import torch
from torch import nn 

class TinyVGG(nn.Module):
  """创建 TinyVGG 架构。

  在 PyTorch 中复制 CNN explainer 网站上的 TinyVGG 架构。
  查看原始架构：[https://poloclub.github.io/cnn-explainer/](https://poloclub.github.io/cnn-explainer/)
  
  参数：
    input_shape: 一个整数，表示输入通道的数量。
    hidden_units: 一个整数，表示层之间的隐藏单元数量。
    output_shape: 一个整数，表示输出单元的数量。
  """
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
      super().__init__()
      self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape, 
                    out_channels=hidden_units, 
                    kernel_size=3, 
                    stride=1, 
                    padding=0),  
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units, 
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,
                        stride=2)
      )
      self.conv_block_2 = nn.Sequential(
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2)
      )
      self.classifier = nn.Sequential(
          nn.Flatten(),
          # 这个 in_features 的形状是怎么来的？
          # 是因为我们网络中的每一层都会压缩并改变输入数据的形状。
          nn.Linear(in_features=hidden_units*13*13,
                    out_features=output_shape)
      )
    
  def forward(self, x: torch.Tensor):
      x = self.conv_block_1(x)
      x = self.conv_block_2(x)
      x = self.classifier(x)
      return x
      # return self.classifier(self.conv_block_2(self.conv_block_1(x))) #

 <- 利用操作符融合的好处
```

现在，代替每次从头开始编写 TinyVGG 模型，我们可以通过以下方式导入它：

```python
import torch
# 导入 model_builder.py
from going_modular import model_builder
device = "cuda" if torch.cuda.is_available() else "cpu"

# 从 "model_builder.py" 脚本中实例化模型
torch.manual_seed(42)
model = model_builder.TinyVGG(input_shape=3,
                              hidden_units=10, 
                              output_shape=len(class_names)).to(device)
```

## 4. 创建 `train_step()` 和 `test_step()` 函数，以及 `train()` 来将它们组合在一起

我们在[notebook 04](https://www.learnpytorch.io/04_pytorch_custom_datasets/#75-create-train-test-loop-functions)中编写了几个训练函数：

1. `train_step()` - 接受一个模型，一个 `DataLoader`，一个损失函数和一个优化器，并在 `DataLoader` 上训练模型。
2. `test_step()` - 接受一个模型，一个 `DataLoader` 和一个损失函数，并在 `DataLoader` 上评估模型。
3. `train()` - 将 1 和 2 结合在一起，为给定的训练轮次进行训练和测试，并返回结果字典。

由于这些函数将是我们模型训练的“引擎”，我们可以将它们都放入一个名为 `engine.py` 的 Python 脚本中，并使用 `%%writefile going_modular/engine.py` 来保存：

```python title="engine.py"
%%writefile going_modular/engine.py
"""
包含用于训练和测试 PyTorch 模型的函数。
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
  """对 PyTorch 模型进行单轮训练。

  将目标 PyTorch 模型切换到训练模式，并执行所有必需的训练步骤（前向传播、损失计算、优化器步进）。

  参数：
    model: 要训练的 PyTorch 模型。
    dataloader: 用于训练的 DataLoader 实例。
    loss_fn: 用于最小化的 PyTorch 损失函数。
    optimizer: 用于最小化损失函数的 PyTorch 优化器。
    device: 目标计算设备（例如 "cuda" 或 "cpu"）。

  返回：
    训练损失和训练准确率的元组。
    格式为 (train_loss, train_accuracy)，例如：
    
    (0.1112, 0.8743)
  """
  # 将模型设置为训练模式
  model.train()
  
  # 设置训练损失和训练准确率的初始值
  train_loss, train_acc = 0, 0
  
  # 遍历数据加载器中的数据批次
  for batch, (X, y) in enumerate(dataloader):
      # 将数据发送到目标设备
      X, y = X.to(device), y.to(device)

      # 1. 前向传播
      y_pred = model(X)

      # 2. 计算并累计损失
      loss = loss_fn(y_pred, y)
      train_loss += loss.item() 

      # 3. 优化器梯度清零
      optimizer.zero_grad()

      # 4. 损失反向传播
      loss.backward()

      # 5. 优化器步进
      optimizer.step()

      # 计算并累计每个批次的准确率
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  # 调整度量指标，计算每个批次的平均损失和准确率 
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
  """对 PyTorch 模型进行单轮测试。

  将目标 PyTorch 模型切换到“评估”模式，并在测试数据集上执行前向传播。

  参数：
    model: 要测试的 PyTorch 模型。
    dataloader: 用于测试的 DataLoader 实例。
    loss_fn: 用于计算测试数据集损失的 PyTorch 损失函数。
    device: 目标计算设备（例如 "cuda" 或 "cpu"）。

  返回：
    测试损失和测试准确率的元组。
    格式为 (test_loss, test_accuracy)，例如：
    
    (0.0223, 0.8985)
  """
  # 将模型设置为评估模式
  model.eval() 
  
  # 设置测试损失和测试准确率的初始值
  test_loss, test_acc = 0, 0
  
  # 开启推理上下文管理器
  with torch.inference_mode():
      # 遍历 DataLoader 批次
      for batch, (X, y) in enumerate(dataloader):
          # 将数据发送到目标设备
          X, y = X.to(device), y.to(device)
  
          # 1. 前向传播
          test_pred_logits = model(X)

          # 2. 计算并累计损失
          loss = loss_fn(test_pred_logits, y)
          test_loss += loss.item()
          
          # 计算并累计准确率
          test_pred_labels = test_pred_logits.argmax(dim=1)
          test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
          
  # 调整度量指标，计算每个批次的平均损失和准确率 
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
  """训练并测试 PyTorch 模型。

  将目标 PyTorch 模型传递给 train_step() 和 test_step() 函数进行若干轮训练和测试。

  在整个过程中计算、打印并存储评估指标。

  参数：
    model: 要训练和测试的 PyTorch 模型。
    train_dataloader: 用于训练的 DataLoader 实例。
    test_dataloader: 用于测试的 DataLoader 实例。
    optimizer: 用于最小化损失函数的 PyTorch 优化器。
    loss_fn: 用于计算损失的 PyTorch 损失函数。
    epochs: 一个整数，表示训练的轮次。
    device: 目标计算设备（例如 "cuda" 或 "cpu"）。

  返回：
    一个字典，包含训练和测试损失，以及训练和测试准确率。每个度量值对应每一轮的值，格式为：
    {train_loss: [...],
     train_acc: [...],
     test_loss: [...],
     test_acc: [...]} 
    例如，如果训练了 2 轮：
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]} 
  """
  # 创建空的结果字典
  results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }
  
  # 在若干轮次中循环执行训练和测试步骤
  for epoch in tqdm(range(epochs)):
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
      test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)
      
      # 打印训练和测试的结果
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
      )

      # 更新结果字典
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)

  # 在所有轮次结束后返回填充好的结果
  return results
```

现在我们已经有了 `engine.py` 脚本，可以通过以下方式导入函数：

```python
# 导入 engine.py
from going_modular import engine

# 使用 train() 函数
engine.train(...)
```

## 5. 创建保存模型的函数 (`utils.py`)

通常，你会希望在训练过程中或训练后保存一个模型。

由于我们在前几个笔记本中已经多次编写了保存模型的代码，因此将它转化为一个函数并保存到文件中是合理的做法。

通常会将辅助函数保存在名为 `utils.py` 的文件中（`utils` 是 `utilities` 的缩写）。

我们将 `save_model()` 函数保存到一个名为 `utils.py` 的文件中，使用 `%%writefile going_modular/utils.py`：

```python title="utils.py"
%%writefile going_modular/utils.py
"""
包含用于 PyTorch 模型训练和保存的各种工具函数。
"""
import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """将 PyTorch 模型保存到目标目录。

  参数:
    model: 要保存的 PyTorch 模型。
    target_dir: 模型保存的目录。
    model_name: 保存模型的文件名，文件名应包括 ".pth" 或 ".pt" 扩展名。
  
  示例用法:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # 创建目标目录
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)
  
  # 创建模型保存路径
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name 应该以 '.pt' 或 '.pth' 结尾"
  model_save_path = target_dir_path / model_name

  # 保存模型的 state_dict()
  print(f"[INFO] 正在保存模型到: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
```

现在，如果我们想使用 `save_model()` 函数，而不是每次都编写它，我们可以通过以下方式导入并使用它：

```python
# 导入 utils.py
from going_modular import utils

# 保存模型到文件
save_model(model=...
           target_dir=...,
           model_name=...)
```

## 6. 训练、评估并保存模型 (`train.py`)

如前所述，你会经常遇到将所有功能集成在一个 `train.py` 文件中的 PyTorch 仓库。

这个文件的主要功能是“使用可用的数据训练模型”。

在我们的 `train.py` 文件中，我们将结合之前编写的所有 Python 脚本的功能，使用它们来训练一个模型。

通过这种方式，我们可以通过在命令行运行一行代码来训练 PyTorch 模型：

```
python train.py
```

为了创建 `train.py` 文件，我们将经过以下步骤：

1. 导入各种依赖项，主要是 `torch`、`os`、`torchvision.transforms` 和来自 `going_modular` 目录下的所有脚本，包括 `data_setup`、`engine`、`model_builder`、`utils`。
   * **注意：** 由于 `train.py` 文件位于 `going_modular` 目录中，我们可以通过 `import ...` 导入其他模块，而不需要 `from going_modular import ...`。
2. 设置各种超参数，如批量大小、训练轮数、学习率和隐藏单元数（这些可以通过 [Python 的 `argparse`](https://docs.python.org/3/library/argparse.html) 在未来设置）。
3. 设置训练和测试目录。
4. 设置与设备无关的代码。
5. 创建必要的数据转换操作。
6. 使用 `data_setup.py` 创建 DataLoader。
7. 使用 `model_builder.py` 创建模型。
8. 设置损失函数和优化器。
9. 使用 `engine.py` 训练模型。
10. 使用 `utils.py` 保存模型。

然后我们可以通过以下命令在笔记本单元格中创建该文件：

```python title="train.py"
%%writefile going_modular/train.py
"""
使用与设备无关的代码训练 PyTorch 图像分类模型。
"""

import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms

# 设置超参数
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# 设置目录
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# 设置目标设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 创建数据转换操作
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

# 使用 data_setup.py 创建 DataLoader
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# 使用 model_builder.py 创建模型
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# 设置损失函数和优化器
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# 使用 engine.py 开始训练
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# 使用 utils.py 保存模型
utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")
```

太棒了！

现在我们可以通过在命令行运行以下命令来训练 PyTorch 模型：

```
python train.py
```

执行这个命令将会利用我们创建的其他代码脚本。

如果我们愿意，还可以调整 `train.py` 文件，使用 Python 的 `argparse` 模块接收命令行输入的参数，这样可以提供不同的超参数设置，像之前讨论的那样：

```
python train.py --model MODEL_NAME --batch_size BATCH_SIZE --lr LEARNING_RATE --num_epochs NUM_EPOCHS
```

## 练习

**资源：**

* [05 练习模板笔记本](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/exercises/05_pytorch_going_modular_exercise_template.ipynb)
* [05 示例解答笔记本](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/solutions/05_pytorch_going_modular_exercise_solutions.ipynb)
    * [YouTube 上的解答笔记本直播编码演示](https://youtu.be/ijgFhMK3pp4)

**练习：**

1. 将从第 1 部分（获取数据）中获取数据的代码转化为一个 Python 脚本，如 `get_data.py`。
    * 当你运行 `python get_data.py` 时，它应该检查数据是否已经存在，如果存在则跳过下载。
    * 如果数据下载成功，你应该能从 `data` 目录中访问到 `pizza_steak_sushi` 图像。
2. 使用 [Python 的 `argparse` 模块](https://docs.python.org/3/library/argparse.html)，使得可以为训练过程提供自定义的超参数值。
    * 为以下项目添加命令行参数：
        * 训练/测试目录
        * 学习率
        * 批量大小
        * 训练的轮数
        * TinyVGG 模型中的隐藏单元数
    * 将每个参数的默认值保持为目前的设置（如笔记本中的设置）。
    * 例如，你应该能够运行以下命令来训练一个 TinyVGG 模型，使用学习率 0.003、批量大小 64 和 20 个训练轮数：`python train.py --learning_rate 0.003 --batch_size 64 --num_epochs 20`。
    * **注意：** 由于 `train.py` 文件依赖于第 05 部分创建的其他脚本，如 `model_builder.py`、`utils.py` 和 `engine.py`，你需要确保它们也可以使用。你可以在 [课程 GitHub 中找到这些脚本](https://github.com/mrdbourke/pytorch-deep-learning/tree/main/going_modular/going_modular)。
3. 创建一个脚本来对目标图像进行预测（例如 `predict.py`），给定一个文件路径和一个保存的模型。
    * 例如，你应该能够运行命令 `python predict.py some_image.jpeg`，并让训练好的 PyTorch 模型对图像进行预测并返回结果。
    * 要查看预测代码示例，可以参考 [笔记本 04 中的预测部分](https://www.learnpytorch.io/04_pytorch_custom_datasets/#113-putting-custom-image-prediction-together-building-a-function)。
    * 你可能还需要编写代码来加载训练好的模型。

## 额外课程

* 要了解更多关于 Python 项目结构的知识，可以查看 Real Python 的 [Python 应用布局指南](https://real

python.com/python-application-layouts/)。
* 要了解如何为 PyTorch 代码提供风格，参考 [Igor Susmelj 的 PyTorch 风格指南](https://github.com/IgorSusmelj/pytorch-styleguide#recommended-code-structure-for-training-your-model)（本章中的很多风格都参考了该指南和类似的 PyTorch 仓库）。
* 如果想要查看一个完整的 `train.py` 脚本和 PyTorch 团队为训练最先进的图像分类模型编写的各种 PyTorch 脚本，请查看他们的 [classification 仓库](https://github.com/pytorch/vision/tree/main/references/classification)。