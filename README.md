# (WORK IN PROGRESS) pointnet_triton_train
UCAS2024课程《GPU架构与编程》大作业2，编写pointnet的triton训练和推理程序。
参考：[attorch](https://github.com/BobMcDear/attorch)包含了大部分需要用到的前向和反向核函数，[EECS 442](https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html)提供了线性层反向传播的基础知识。
## 2024-12-2
实现了线性层和核为1的卷积层的正向与反向传播，预计最终用时是pytorch的两倍或更多。
## 使用
修改全局变量`mode`为`"train"`或`"test"`进行训练或推理。在`main`方法中修改数据集以及模型路径。
