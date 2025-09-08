#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：transfer_learning
@File        ：train_word_vector.py
@Create at   ：2025/9/8 18:37
@version     ：V1.0
@Author      ：erainm
@Description : 训练词向量
'''
import fasttext

model = fasttext.load_model("./model/cc.zh.300.bin")

# 查看词向量
print(model.get_word_vector("周杰伦"))

# 利用邻近词进行效果检验
print(model.get_nearest_neighbors("周杰伦"))
print(model.get_nearest_neighbors("周星驰"))