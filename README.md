# OnePassExperiment

Test model performance in one-pass setting.

**运行load_data.py来加载MNIST和Fashion MNIST数据集。**线性模型实验示例：

```bash
python linear_acc.py australian --std --lr 0.01
```

神经网络实验示例：

```bash
python mlp_acc.py australian --std --lr 0.01
```

在这里，我们讨论一下基于梯度的算法在只浏览一遍数据集的情形，也就是online setting下的表现。在数据量足够大的情况下，我们往往只需要过一遍数据就可以让模型达到预期的效果。在大规模数据的情况下，我们甚至只能浏览一次所有数据。在这里，我们研究的数据集包括千样本级和万样本级，同时包括二分类和多分类。

使用模型包括：

- 线性模型；
- 单隐层神经网络；

使用的数据集：

- Tabular数据：
  - australian;
  - diabetes;
  - splice;
  - svmguide3;
  - iris;
  - breast cancer;
  - wine;
- Image数据：
  - digits;
  - mnist;
  - fashion mnist.

我们的目标是通过调参获得不同模型在不同数据集上的性能极限。这里的优化方法我们统一为SGD，使用五折交叉验证。

## Dataset

我们给出实验用到的数据集的基本信息

|    Dataset    | n_samples | n_features | n_class |
| :-----------: | :-------: | :--------: | :-----: |
|  australian   |    690    |     14     |    2    |
|   diabetes    |    768    |     8      |    2    |
|    splice     |   1000    |     60     |    2    |
|   svmguide3   |   1243    |     22     |    2    |
|     iris      |    150    |     4      |    3    |
| breast cancer |    569    |     30     |    2    |
|     wine      |    178    |     13     |    3    |
|    digits     |   1797    |     64     |   10    |
|     MNIST     |   60000   |    784     |   10    |
| fashion MNIST |   60000   |    784     |   10    |

## Linear Model(LM)

线性模型的参数实际上只有learning rate，以及是否进行预处理。先看不进行预处理：

|    Dataset    |   lr=2    | lr=1  |  lr=0.5   | lr=0.1 |  lr=0.01  |
| :-----------: | :-------: | :---: | :-------: | :----: | :-------: |
|  australian   | **56.8%** | 56.8% |   56.8%   | 56.8%  |   56.8%   |
|   diabetes    | **62.4%** | 60.0% |   59.6%   | 59.5%  |   49.2%   |
|    splice     |   75.4%   | 74.7% |   73.7%   | 74.1%  | **77.6%** |
|   svmguide3   |   78.5%   | 79.3% | **80.8%** | 65.6%  |   58.3%   |
|     iris      |   65.3%   | 65.3% |   69.3%   | 78.7%  | **79.3%** |
| breast cancer | **66.8%** | 66.8% |   66.8%   | 66.6%  |   66.8%   |
|     wine      | **43.8%** | 43.8% |   43.8%   | 42.2%  |   34.9%   |
|    digits     |   89.9%   | 89.3% |   90.5%   | 90.3%  | **91.5%** |
|     MNIST     | **62.9%** | 62.9% |   62.9%   | 62.9%  |   59.5%   |
| fashion MNIST |   61.4%   | 61.4% |   61.4%   | 62.6%  | **64.1%** |

标准化之后的效果：

|    Dataset    |   lr=2    |   lr=1    | lr=0.5 |  lr=0.1   |  lr=0.01  |
| :-----------: | :-------: | :-------: | :----: | :-------: | :-------: |
|  australian   |   71.2%   |   69.7%   | 71.3%  |   76.8%   | **85.4%** |
|   diabetes    |   63.8%   |   62.2%   | 64.3%  | **74.1%** |   73.6%   |
|    splice     |   72.4%   |   73.4%   | 73.7%  |   73.4%   | **80.0%** |
|   svmguide3   |   78.0%   |   67.2%   | 76.5%  | **78.4%** |   74.8%   |
|     iris      |   62.0%   |   68.0%   | 81.3%  | **88.0%** |   82.7%   |
| breast cancer |   82.2%   |   83.8%   | 83.4%  |   83.8%   | **89.6%** |
|     wine      |   92.1%   |   93.8%   | 95.5%  | **97.2%** |   93.8%   |
|    digits     |   58.4%   |   58.9%   | 60.2%  |   68.7%   | **86.3%** |
|     MNIST     | **19.6%** |   17.4%   | 15.9%  |   17.5%   |   16.9%   |
| fashion MNIST |   39.9%   | **40.5%** | 38.2%  |   38.7%   |   38.3%   |

最大最小化后的效果：

|    Dataset    |   lr=2    | lr=1  |  lr=0.5   |  lr=0.1   |  lr=0.01  |
| :-----------: | :-------: | :---: | :-------: | :-------: | :-------: |
|  australian   |   85.1%   | 85.5% | **85.9%** |   85.4%   |   84.1%   |
|   diabetes    | **74.0%** | 73.8% |   73.4%   |   73.0%   |   65.6%   |
|    splice     |   73.8%   | 74.7% |   74.0%   | **77.2%** |   77.0%   |
|   svmguide3   | **78.8%** | 76.3% |   72.6%   |   63.5%   |   60.7%   |
|     iris      |   86.0%   | 93.3% | **93.3%** |   88.0%   |   60.7%   |
| breast cancer |   85.9%   | 87.0% |   88.9%   |   91.2%   | **91.4%** |
|     wine      |   91.6%   | 93.8% | **94.4%** |   92.2%   |   77.5%   |
|    digits     |   91.9%   | 92.0% | **92.4%** |   92.2%   |   89.9%   |
|     MNIST     |   63.5%   | 62.5% |   62.0%   |   64.4%   | **74.3%** |
| fashion MNIST |   66.0%   | 64.3% |   65.4%   |   65.9%   | **75.9%** |

我们可以总结线性模型在各数据集上的极限性能（std表示标准化，minmax表示min-max归一化）：

|    Dataset    |     Setting     | Accuracy |
| :-----------: | :-------------: | :------: |
|  australian   | minmax, lr=0.5  |  85.9%   |
|   diabetes    |   std, lr=0.1   |  74.1%   |
|    splice     |   std, l=0.01   |  80.0%   |
|   svmguide3   |     lr=0.5      |  80.8%   |
|     iris      | minmax, lr=0.5  |  93.3%   |
| breast cancer | minmax, lr=0.01 |  91.4%   |
|     wine      |   std, lr=0.1   |  97.2%   |
|    digits     | minmax, lr=0.5  |  92.4%   |
|     MNIST     | minmax, lr=0.01 |  74.3%   |
| fashion MNIST | minmax, lr=0.01 |  75.9%   |

## 单隐层神经网络(MLP)

单隐层神经网络除了学习率以外，还需要考虑隐层神经元数目，以及激活函数类型。在这里，我们将隐层神经元数目设置为128，激活函数取Tanh和ReLU。在这里，我们直接展示学习率调参之后的最优结果：

|    Dataset    | std tanh best | std relu best | minmax tanh best | minmax relu best |
| :-----------: | :-----------: | :-----------: | :--------------: | :--------------: |
|  australian   |     85.7%     |     85.7%     |      85.7%       |    **85.8%**     |
|   diabetes    |     72.7%     |     72.5%     |    **73.6%**     |      73.2%       |
|    splice     |   **79.8%**   |     80.9%     |      78.7%       |      78.6%       |
|   svmguide3   |     74.3%     |   **80.4%**   |      62.3%       |      68.6%       |
|     iris      |     89.3%     |   **94.7%**   |      93.3%       |      94.0%       |
| breast cancer |   **97.2%**   |     92.1%     |      96.5%       |      93.3%       |
|     wine      |   **97.2%**   |     96.7%     |      95.0%       |      96.6%       |
|    digits     |     93.1%     |     80.7%     |    **91.2%**     |      90.1%       |
|     MNIST     |   **94.1%**   |     55.9%     |    **94.1%**     |      88.1%       |
| fashion MNIST |   **84.2%**   |     73.1%     |      82.5%       |      76.0%       |

假如我们将单隐层神经网络和线性模型进行比较：

|    Dataset    |  LM best  | MLP best  |
| :-----------: | :-------: | :-------: |
|  australian   | **85.9%** |   85.8%   |
|   diabetes    | **74.1%** |   73.6%   |
|    splice     | **80.0%** |   79.8%   |
|   svmguide3   | **80.8%** |   74.3%   |
|     iris      |   93.3%   | **94.7%** |
| breast cancer |   91.4%   | **97.2%** |
|     wine      | **97.2%** | **97.2%** |
|    digits     | **92.4%** |   91.2%   |
|     MNIST     |   74.3%   | **94.1%** |
| fashion MNIST |   75.9%   | **84.2%** |

我们发现，神经网络由于参数量大收敛慢，而且本身就不适合tabular数据，所以处于在线学习场景下，tabular数据集上的表现甚至会差于线性模型。只有在MNIST这种结构化数据集下，神经网络才能发挥它的功效。
