import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from collections import defaultdict
import math


# 自定义数据集类
class IrisDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# 朴素贝叶斯分类器实现
class GaussianNaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.feature_stats = defaultdict(
            list
        )  # {class: [(mean1, var1), (mean2, var2), ...]}

    def fit(self, X, y):
        # 计算每个类的先验概率
        classes, counts = torch.unique(y, return_counts=True)
        total_samples = len(y)
        self.class_priors = {
            c.item(): count.item() / total_samples for c, count in zip(classes, counts)
        }

        # 对每个类计算每个特征的均值和方差
        for c in classes:
            class_mask = y == c
            class_data = X[class_mask]

            class_feature_stats = []
            for feature_idx in range(X.shape[1]):
                feature_values = class_data[:, feature_idx]
                mean = torch.mean(feature_values).item()
                var = torch.var(feature_values, unbiased=False).item()  # 使用样本方差
                class_feature_stats.append((mean, var))

            self.feature_stats[c.item()] = class_feature_stats

    def predict(self, X):
        predictions = []
        for sample in X:
            class_scores = {}

            # 对每个类计算后验概率
            for c, prior in self.class_priors.items():
                log_prob = math.log(prior)  # 先验概率的对数

                # 计算每个特征的条件概率
                for feature_idx, value in enumerate(sample):
                    mean, var = self.feature_stats[c][feature_idx]

                    # 高斯概率密度函数
                    if var > 0:  # 防止除以零
                        exponent = -((value.item() - mean) ** 2) / (2 * var)
                        normalizer = 1.0 / math.sqrt(2 * math.pi * var)
                        feature_prob = normalizer * math.exp(exponent)
                    else:  # 方差为零的情况（所有值相同）
                        feature_prob = (
                            1.0 if math.isclose(value.item(), mean) else 1e-10
                        )

                    # 使用对数概率防止下溢
                    log_prob += math.log(feature_prob + 1e-10)  # 添加小值防止log(0)

                class_scores[c] = log_prob

            # 选择最高概率的类
            best_class = max(class_scores, key=class_scores.get)
            predictions.append(best_class)

        return torch.tensor(predictions, dtype=torch.long)

    def accuracy(self, X, y):
        preds = self.predict(X)
        correct = (preds == y).sum().item()
        return correct / len(y)


# 主程序
if __name__ == "__main__":
    # 加载数据
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names

    # 创建数据集
    dataset = IrisDataset(X, y)

    # 划分训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    # 提取训练数据
    X_train, y_train = [], []
    for batch in train_loader:
        features, labels = batch
        X_train.append(features)
        y_train.append(labels)
    X_train = torch.cat(X_train)
    y_train = torch.cat(y_train)

    # 提取测试数据
    X_test, y_test = next(iter(test_loader))

    # 训练模型
    print("训练朴素贝叶斯模型...")
    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)

    # 评估模型
    train_acc = model.accuracy(X_train, y_train)
    test_acc = model.accuracy(X_test, y_test)

    print(f"\n训练准确率: {train_acc:.4f}")
    print(f"测试准确率: {test_acc:.4f}")

    # 打印一些预测示例
    print("\n预测示例:")
    for i in range(5):
        sample = X_test[i]
        true_class = class_names[y_test[i].item()]
        pred_class = class_names[model.predict(sample.unsqueeze(0))[0].item()]
        print(f"样本 {i+1}: 真实类别: {true_class:<12} 预测类别: {pred_class}")

    # 打印学习到的参数
    print("\n学习到的参数:")
    for c in model.class_priors:
        print(f"\n类别: {class_names[c]}")
        print(f"先验概率: {model.class_priors[c]:.4f}")
        for i, (mean, var) in enumerate(model.feature_stats[c]):
            print(f"  特征 {feature_names[i]}: 均值={mean:.4f}, 方差={var:.4f}")
