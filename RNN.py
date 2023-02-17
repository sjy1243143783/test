import numpy as np

# 定义模拟数据
X = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 1, 1]])
Y = np.array([[1, 0], [0, 1], [1, 1], [1, 1]])

# 定义参数
input_size = 3
hidden_size = 4
output_size = 2
learning_rate = 0.1

# 初始化权重矩阵
Wxh = np.random.randn(hidden_size, input_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(output_size, hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((output_size, 1))

# 定义前向传播函数
def rnn_forward(X, Wxh, Whh, Why, bh, by):
    hidden_states = []
    hidden_state = np.zeros((hidden_size, 1))
    for i in range(X.shape[0]):
        # 计算隐藏层状态
        hidden_state = np.tanh(np.dot(Wxh, X[i].reshape(-1, 1)) + np.dot(Whh, hidden_state) + bh)
        # 记录隐藏层状态
        hidden_states.append(hidden_state)
    # 计算输出
    y = np.dot(Why, hidden_state) + by
    # 返回输出和隐藏状态
    return y, hidden_states

# 定义反向传播函数
def rnn_backward(X, Y, y_pred, hidden_states, Wxh, Whh, Why, bh, by):
    # 初始化梯度
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dWhy = np.zeros_like(Why)
    dbh = np.zeros_like(bh)
    dby = np.zeros_like(by)
    dhidden_next = np.zeros_like(hidden_states[0])
    for i in reversed(range(len(X))):
        # 计算输出误差和梯度
        output_error = y_pred - Y[i].reshape(-1, 1)
        dWhy += np.dot(output_error, hidden_states[i].T)
        dby += output_error
        # 计算隐藏层误差和梯度
        dhidden = np.dot(Why.T, output_error) + np.dot(Whh.T, dhidden_next)
        dhidden_raw = (1 - hidden_states[i] ** 2) * dhidden
        dbh += dhidden_raw
        dWxh += np.dot(dhidden_raw, X[i].reshape(1, -1))
        dWhh += np.dot(dhidden_raw, hidden_states[i-1].T)
        dhidden_next = dhidden_raw
    # 更新权重
    Wxh -= learning_rate * dWxh
    Whh -= learning_rate * dWhh
    Why -= learning_rate * dWhy
    bh -= learning_rate * dbh
    by -= learning_rate * dby

# 训练模型
for epoch in range(1000):
    # 前向传播
    y_pred, hidden_states = rnn_forward(X, Wxh, Whh, Why, bh, by)
    # 计算损失
    loss = np.sum((y_pred - Y.T) ** 2)
    # 反向传播
    rnn_backward(X, Y.T, y_pred, hidden_states, Wxh, Whh, Why, bh, by)
    # 每隔100个epoch输出一次损失
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
X_new = np.array([[1, 0, 0], [0, 0, 1]])
y_pred, _ = rnn_forward(X_new, Wxh, Whh, Why, bh, by)
print(f"Predictions: {y_pred}")
