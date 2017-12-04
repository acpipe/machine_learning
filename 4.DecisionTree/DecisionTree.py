#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2017/11/23
# Author: Acceml

"""
决策树demo.
原理讲解见博客:acceml.github.io
"""

import numpy as np


def load_data():
    data = []
    features = []
    line_num = 0
    with open('watermelon2.0.txt', 'r', encoding='utf8') as i_file:
        for line in i_file:
            line_num += 1
            tokens = line.strip().split(',')
            if tokens[-1] == '是':
                tokens[-1] = 1
            elif tokens[-1] == '否':
                tokens[-1] = 0
            if line_num != 1:
                data.append([token for token in tokens])
            else:
                features.extend([token for token in tokens])
    data = np.array(data)
    # 删除第一列编号，编号不作为feature.
    data = np.delete(data, 0, axis=1)
    features.pop(0)
    print('data :' + str(data))
    print('features :' + str(features))

    return data, features


def calc_info_entropy(data_set):
    """ 计算信息熵.
    :return: 信息熵.
    """
    labels = data_set[:, -1]
    # 计算正负样本个数.
    label_2_count = {}
    for label in labels:
        current_label = label
        if current_label not in label_2_count:
            label_2_count[current_label] = 0
        label_2_count[current_label] += 1

        # 根据正负样本数计算信息熵.
    ent = 0.0
    entry_num = len(labels)
    for key in label_2_count:
        prob = float(label_2_count[key]) / entry_num
        ent -= prob * np.log2(prob)
    return ent


def split_data_set(raw_data_set, feature_index, feature_value):
    """
    将指定特征的特征值等于 value 的行剩下列作为子数据集.
    :param raw_data_set: 原始特征矩阵
    :param feature_index: 划分的特征index
    :param feature_value: 划分特征的值
    :return: 划分之后的数据集.
    """
    # 原来的矩阵不修改.
    raw_data_set = raw_data_set.copy()
    data_set = []
    for features in raw_data_set:
        if features[feature_index] == feature_value:
            data_set.append(features)
    data_set = np.delete(np.array(data_set), feature_index, axis=1)
    return data_set


def choose_best_feature(raw_data_set, features):
    """
    选择信息增益最大的特征.
    :param raw_data_set: 原始数据集.
    :param features: feature集合.
    :return:
    """
    base_entropy = calc_info_entropy(raw_data_set)
    max_info_gain, best_feature_index, best_feature = 0.0, -1, ''
    for i in range(len(features)):
        values = raw_data_set[:, i]
        condition_entropy = 0.0
        unique_value = set(values)
        for value in unique_value:
            sub_data_set = split_data_set(raw_data_set, i, value)
            weight = len(sub_data_set) / float(len(raw_data_set))
            condition_entropy += weight * calc_info_entropy(sub_data_set)
        info_gain = base_entropy - condition_entropy
        print('info_gain(' + features[i] + ')=' + str(info_gain))
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = features[i]
            best_feature_index = i
    return best_feature, best_feature_index


def majority_vote(class_list):
    label_2_count = {}
    for vote in class_list:
        if vote not in label_2_count.keys():
            label_2_count[vote] = 0
            label_2_count[vote] += 1
    sorted_label_2_count = sorted(label_2_count.iteritems(), key=np.operator.itemgetter(1), reverse=True)
    return sorted_label_2_count[0][0]


def create_dt(raw_data_set, features):
    """
    递归创建决策树.
    :param raw_data_set:
    :param features:
    :return:
    """
    print('-' * 30)
    # 都是相同的分类
    labels = [item[-1] for item in raw_data_set]
    if labels.count(labels[0]) == len(labels):
        return labels[0]

    # 没有特征进行划分,进行多数表决.
    if len(features) == 0:
        return majority_vote(labels)

    best_feature, best_feature_index = choose_best_feature(raw_data_set, features)
    dt = {best_feature: {}}

    # 用最优特征进行分类
    feature_values = set(item[best_feature_index] for item in raw_data_set)

    for feature_value in feature_values:
        copied_features = features.copy()
        sub_data_set = split_data_set(raw_data_set, best_feature_index, feature_value)
        copied_features.pop(best_feature_index)
        son_dt = create_dt(sub_data_set, copied_features)
        dt[best_feature][feature_value] = son_dt

    return dt


def main():
    data_set, features = load_data()
    dt = create_dt(data_set, features)
    # print(dt)


if __name__ == '__main__':
    main()
