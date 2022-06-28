# NER_weibo数据挖掘大作业

## 依赖安装

首先转到/mylog目录

```shell
cd mylog
```

```shell
pip install -r requirement
```

安装依赖时请等待

## 模型训练

在/mylog上直接run即可

```shell
python main
```

## 结果可视化

在训练后，在/mylog目录上

```
fitlog log logs 
```

打开所启动的本地网址，即可开启可视化

本文的实验结果已放出，在目录实验可视化里，打开html即可查看

代码默认运行bert，若要运行随机初始化向量或者word2vec可在/mylog/main.py下修改（很简单，注销即可）