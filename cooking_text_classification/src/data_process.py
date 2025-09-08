#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：transfer_learning
@File        ：data_process.py
@Create at   ：2025/9/8 16:43
@version     ：V1.0
@Author      ：erainm
@Description : 数据处理
'''
import re

def preprocess_text(input_file, output_file):
    """
        预处理文本数据：分离标点符号并转换为小写
        # 相当于命令: " cat cooking.stackexchange.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > cooking.preprocessed.txt "
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用正则表达式在标点符号前后添加空格
    content = re.sub(r"([.!?,'/()])", r" \1 ", content)
    # 转换为小写
    content = content.lower()

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)


def split_data(input_file, train_file, valid_file, train_lines=12404, valid_lines=3000):
    """
        分割数据为训练集和验证集
        相当于命令:
        head -n 12404 cooking.preprocessed.txt > cooking.pre.train
        tail -n 3000 cooking.preprocessed.txt > cooking.pre.valid
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 获取前train_lines行作为训练数据
    train_lines_data = lines[:train_lines]
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines_data)

    # 获取后valid_lines行作为验证数据
    valid_lines_data = lines[-valid_lines:]
    with open(valid_file, 'w', encoding='utf-8') as f:
        f.writelines(valid_lines_data)


# 执行数据预处理
preprocess_text("../data/cooking.stackexchange.txt", "../data/cooking.preprocessed.txt")

# 分割数据集
split_data("../data/cooking.preprocessed.txt", "../data/cooking.pre.train", "../data/cooking.pre.valid")