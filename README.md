# pointnet_triton_train
UCAS2024课程《GPU架构与编程》大作业2，编写pointnet的triton训练和推理程序。~~为什么不写cuda呢？因为我研究了两天发现计算图就够我写两周了，还需要改一改之前的💩山，水平太菜没法在结课前写完。~~  
参考：[attorch](https://github.com/BobMcDear/attorch)包含了大部分需要用到的前向和反向核函数，[EECS 442](https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html)提供了线性层反向传播的基础知识。
### 2024-12-2
实现了线性层和核为1的卷积层的前向与反向传播，预计最终用时是pytorch的两倍或更多。
### 2024-12-3
实现了其它部分层的前向与反向。达到0.75准确率，采样64个点，v100上的训练时间大约30-60s。
## 使用
修改全局变量`mode`为`"train"`或`"test"`进行训练或推理。在`main`方法中修改数据集以及模型路径。  
`TritonKernels`类实现了各个核函数。
`TritonOps`类实现了几个基本运算，包装了一下核函数。
`TritonFunctions`类定义了运算的前向和反向传播。
`TritonLayers`类重写了pytorch对应的模块。
