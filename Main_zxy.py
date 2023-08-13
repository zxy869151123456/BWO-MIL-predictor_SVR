import pandas as pd
import numpy as np
from SVR import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle

def  RMSE (prediction,  actual):
    return np.sqrt (np.mean (np.square (prediction - actual ) ) )

df = pd.read_csv('train_6.csv')
df_t = pd.read_csv('test_6.csv')

data = df.values
data_t = df_t.values

X = data[:, 3:]
Y = data[:, 2]

X_t = data_t[:, 3:]
Y_t = data_t[:, 2]

svr = SVR({
    'kernel_type': 'rbf',
    'epsilon': 0.09,
#    'cost': 10.0,
    'cost': 87.0,
    'max_opt_times': 10000,
#   'rbf_gamma': 1.0,
    'rbf_gamma': 0.5,
    'coef_lin': 1.0,
    'coef_const': 1.0,
    'poly_degree': 2
})

svr.fit(X, Y)
predicted = svr.predict(X)
predicted_t = svr.predict(X_t)

RMSE_1 = RMSE(Y, predicted)
RMSE_t = RMSE(Y_t, predicted_t)

print(f'R:{np.corrcoef(Y, predicted)[0,1]:.3f}')
print(f'RMSE:{RMSE_1:.3f}')
print(f'R_t:{np.corrcoef(Y_t, predicted_t)[0][1]:.3f}')
print(f'RMSE_t:{RMSE_t:.3f}')
print(predicted_t)

# 测试预报情况
print(svr.predict([[-1.120412, 1.67, 0.335, 0.67, 0.818535277, -0.173925197]]))

# 将模型保存为文件
model_file = open('model.pkl', 'wb')
pickle.dump(svr, model_file)
model_file.close()

model_file_1 = open('SVR.pkl', 'wb')
pickle.dump(SVR, model_file_1)
model_file_1.close( )

# 读取模型文件，再次尝试预报
model_file = open('model.pkl', 'rb')
svr = pickle.load(model_file)
print(svr.predict([[-1.120412, 1.67, 0.335, 0.67, 0.818535277, -0.173925197]]))

# 最后将保存的模型pkl文件复制到网络服务项目的文件夹中