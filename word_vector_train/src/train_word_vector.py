#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：transfer_learning
@File        ：train_word_vector.py
@Create at   ：2025/9/8 17:48
@version     ：V1.0
@Author      ：erainm
@Description : 训练词向量，数据来源于英语维基百科的部分网页信息
'''
import fasttext

# 模型训练
# 在训练词向量过程中, 可以设定很多常用超参数来调节我们的模型效果, 如:
# 无监督训练模式: 'skipgram' 或者 'cbow', 默认为'skipgram', 在实践中，skipgram模式在利用子词方面比cbow更好.
# 词嵌入维度dim: 默认为100, 但随着语料库的增大, 词嵌入的维度往往也要更大.
# 数据循环次数epoch: 默认为5, 但当你的数据集足够大, 可能不需要那么多次.
# 学习率lr: 默认为0.05, 根据经验, 建议选择[0.01，1]范围内.
# 使用的线程数thread: 默认为12个线程, 一般建议和你的cpu核数相同.
model = fasttext.train_unsupervised("../data/fil9", model="cbow", dim=300, epoch=1, lr=0.1, thread=8)

# 模型加载
# model = fasttext.load_model("../model/file9.bin")

# 使用save_model保存模型
model.save_model("../model/fil9.bin")

# 模型效果检验
# 检查单词向量质量的一种简单方法就是查看其邻近单词, 通过主观来判断这些邻近单词是否与目标单词相关来粗略评定模型效果好坏.
print(model.get_nearest_neighbors('sports'))
print(model.get_nearest_neighbors('music'))
print(model.get_nearest_neighbors('dog'))