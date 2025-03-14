python环境 一个个手动安装 keras-2.0.9 

```bash
pip install pandas~=1.1.5 -i http://pypi.doubanio.com/simple/ --trusted-host pypi.doubanio.com
```

# RNN

> CNN是包含卷积计算且具有深度结构的前馈神经网络

1. 可以利用它内部的记忆来处理任意时序的输入序列，这让它可以更容易处理如不分段的手写识别、语音识别：

RNN的假设——事物的发展是按照**时间序列**展开的，即前一刻发生的事物会对未来的事情的发展产生影响。所以在处理过程中，每一刻的输出是带着之前输出值加权之后的结果

2. 梯度消失和梯度爆炸

梯度是指前后信息之间相互影响的程度

LSTM 通过异或减少数据量，将重复信息遗忘，将未知信息记录下来，将结果更新之后，再输出。

3. 注意力机制

找到特征中更有用的一部分 

参考开源项目  https://github.com/philipperemy/keras-attention-mechanism

# Python

1. list 和 array 的区别

list和array都可以根据索引来取其中的元素。

list是列表，list中的元素的数据类型可以不一样。array是数组，数组中的元素的数据类型必须一样。

list不可以进行四则运算，array可以进行四则运算。