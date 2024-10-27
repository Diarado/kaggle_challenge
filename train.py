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

# 划分训练集和验证集
X_train = train_data[features].iloc[:-1000]
y_train = train_data[target].iloc[:-1000]
X_val = train_data[features].iloc[-1000:]
y_val = train_data[target].iloc[-1000:]

# 训练
train_dataset = lgb.Dataset(X_train, label=y_train)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'n_estimators': 1000,
    'device_type': 'gpu'
}
model = lgb.train(params, train_dataset)

# 保存训练好的模型
model_file_path = "lgbm_model.txt"
model.save_model(model_file_path)
print(f"Model saved to {model_file_path}")


# 加载已保存的模型
model_file_path = "lgbm_model.txt"
loaded_model = lgb.Booster(model_file=model_file_path)
print(f"Model loaded from {model_file_path}")


# 验证集测试
y_val_pred = loaded_model.predict(X_val)
val_mse = mean_squared_error(y_val, y_val_pred)
val_rmse = val_mse ** 0.5
print(f"Validation RMSE: {val_rmse}")


print("训练完毕")
