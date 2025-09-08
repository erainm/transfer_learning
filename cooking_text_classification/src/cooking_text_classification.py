#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：transfer_learning
@File        ：cooking_text_classification.py
@Create at   ：2025/9/8 16:41
@version     ：V1.0
@Author      ：erainm
@Description : 烹饪相关文本分类案例
'''
import fasttext

# 方式一：训练模型
# 设置train_supervised方法中的参数epoch来增加训练轮数, 默认的轮数是5次,增加轮数意味着模型能够有更多机会在有限数据中调整分类规律, 当然这也会增加训练时间
# 设置train_supervised方法中的参数lr来调整学习率, 默认的学习率大小是0.1，增大学习率意味着增大了梯度下降的步长使其在有限的迭代步骤下更接近最优点
# 设置train_supervised方法中的参数wordNgrams来添加n-gram特征, 默认是1, 也就是没有n-gram特征
#   将其设置为2意味着添加2-gram特征, 这些特征帮助模型捕捉前后词汇之间的关联, 更好的提取分类规则用于模型分类(会增加训练时间)
# 为了能够提升fasttext模型的训练效率, 减小训练时间
# 设置train_supervised方法中的参数loss来修改损失计算方式(等效于输出层的结构), 默认是softmax层结构
#   将其设置为'hs', 代表层次softmax结构, 意味着输出层的结构(计算方式)发生了变化, 将以一种更低复杂度的方式来计算损失.
# model = fasttext.train_supervised(input="../data/cooking.pre.train", epoch=25, lr=1.0, wordNgrams=2, loss='hs')

# 方式二：自动参数
# 手动调节和寻找超参数是非常困难的, 因为参数之间可能相关, 并且不同数据集需要的超参数也不同,
# 因此可以使用fasttext的autotuneValidationFile参数进行自动超参数调优.
# autotuneValidationFile参数需要指定验证数据集所在路径, 它将在验证集上使用随机搜索方法寻找可能最优的超参数.
# 使用autotuneDuration参数可以控制随机搜索的时间, 默认是300s, 根据不同的需求, 我们可以延长或缩短时间.
# 验证集路径'cooking.pre.valid', 随机搜索600秒
model = fasttext.train_supervised(input='../data/cooking.pre.train', autotuneValidationFile='../data/cooking.pre.valid', autotuneDuration=600)

# 方式三：针对多标签多分类问题, 使用'softmax'或者'hs'有时并不是最佳选择, 因为最终得到的应该是多个标签, 而softmax却只能最大化一个标签.
# 所以我们往往会选择为每个标签使用独立的二分类器作为输出层结构, 对应的损失计算方式为'ova'表示one vs all.
# 这种输出层的改变意味着我们在统一语料下同时训练多个二分类模型,对于二分类模型来讲, lr不宜过大, 这里我们设置为0.2
# model = fasttext.train_supervised(input="../data/cooking.pre.train", lr=0.2, epoch=25, wordNgrams=2, loss='ova')

# 使用model的save_model方法保存模型到指定目录
# model.save_model("../model/model_cooking_specify_para.bin")
model.save_model("../model/model_cooking_auto_para.bin")
# model.save_model("../model/model_cooking_specify_para2.bin")

# 模型评估
print(model.test("../data/cooking.pre.valid"))