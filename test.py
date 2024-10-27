import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# 读取训练数据集
train_file_path = './data/train/part-0.parquet'
train_data = pd.read_parquet(train_file_path)
print("Training data preview:")
print(train_data.head())

# 划分训练集和验证集
# 提取所有 responder 列
# 特征列：去除所有 responder 列，包括目标列
responders = [col for col in train_data.columns if col.startswith('responder_')]
target = 'responder_6'  # 设置目标列 responder_6
features = [col for col in train_data.columns if col not in responders]


# 加载已保存的模型
model_file_path = "lgbm_model.txt"
loaded_model = lgb.Booster(model_file=model_file_path)
print(f"Model loaded from {model_file_path}")


# 读取测试数据集
test_file_path = './data/test/part-0.parquet'
test_data = pd.read_parquet(test_file_path)
X_test = test_data[features]  # 使用与训练一致的特征

# 预测 test.parquet 数据集的 responder_6
y_test_pred = loaded_model.predict(X_test)

# 保存预测结果
results = pd.DataFrame({'Predicted_responder_6': y_test_pred})
# print("Test predictions preview:")
# print(results.head(10))

results = pd.DataFrame({
    'row_id': range(len(y_test_pred)),
    'responder_6': y_test_pred
})

# 保存预测结果为 .parquet 文件
output_file_path = "submission.parquet"
results.to_parquet(output_file_path, index=False)


print(f"Predictions saved to {output_file_path}")
print(results.head())

# results.to_csv("test_predictions.csv", index=False)
