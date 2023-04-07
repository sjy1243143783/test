import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, TimeDistributed,Reshape
from tensorflow_addons.layers import CRF
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score

# 定义数据集
class NERDataset():
    def __init__(self, file_path, word_to_ix, tag_to_ix, max_len):
        self.sentences = []
        self.tags = []
        with open(file_path, 'r', encoding='utf-8') as f:
            sentence = []
            tags = []
            for line in f:
                line = line.strip().split()
                if not line:
                    if sentence and tags:
                        self.sentences.append(sentence)
                        self.tags.append(tags)
                        sentence = []
                        tags = []
                else:
                    word = line[0]
                    tag = line[1]
                    if word in word_to_ix:
                        sentence.append(word_to_ix[word])
                        tags.append(tag_to_ix[tag])
            if sentence and tags:
                self.sentences.append(sentence)
                self.tags.append(tags)

        self.sentences = pad_sequences(self.sentences, maxlen=max_len, padding='post', value=0)
        self.tags = pad_sequences(self.tags, maxlen=max_len, padding='post', value=0)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.tags[idx]



def create_model(max_len, word_vocab_size, tag_vocab_size, embedding_dim, hidden_dim):
    # 定义输入
    inputs = Input(shape=(max_len,))

    # 定义嵌入层
    embeddings = Embedding(input_dim=word_vocab_size, output_dim=embedding_dim, input_length=max_len)(inputs)

    # 定义双向LSTM层
    lstm = Bidirectional(LSTM(units=hidden_dim, return_sequences=True))(embeddings)

    # 为 Dense 层调整形状
    reshape = Reshape((-1, 2 * hidden_dim))(lstm)

    # 定义全连接层
    fc = TimeDistributed(Dense(units=tag_vocab_size))(reshape)

   # 定义CRF层
    crf = CRF(units=tag_vocab_size)
    outputs = crf(fc)

    # 定义模型
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


# 训练模型
def train_model(train_dataset, val_dataset, word_vocab_size, tag_vocab_size, embedding_dim, hidden_dim, batch_size, epochs):
    max_len = train_dataset.sentences.shape[1]
    model = create_model(max_len, word_vocab_size, tag_vocab_size, embedding_dim, hidden_dim)
    early_stop = EarlyStopping(monitor='val_loss', patience=3)

    model.fit(train_dataset.sentences, train_dataset.tags, validation_data=(val_dataset.sentences, val_dataset.tags), batch_size=batch_size, epochs=epochs, callbacks=[early_stop])

    return model

# 预测标签序列
def predict_labels(model, X_test):
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1)
    return y_pred

# 定义标签到索引的映射
tag_to_ix = {'O': 0, 'B-LOC': 1, 'I-LOC': 2, 'B-PER': 3, 'I-PER': 4, 'B-ORG': 5, 'I-ORG': 6}

# 定义单词到索引的映射
word_to_ix = {}
with open('msra/train.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip().split()
        if line:
            word = line[0]
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

# 定义数据集
train_dataset = NERDataset('msra/train.txt', word_to_ix, tag_to_ix, max_len=200)
val_dataset = NERDataset('msra/val.txt', word_to_ix, tag_to_ix, max_len=200)

# 训练模型
model = train_model(train_dataset, val_dataset, word_vocab_size=len(word_to_ix), tag_vocab_size=len(tag_to_ix), embedding_dim=128, hidden_dim=128, batch_size=32, epochs=10)

# 测试模型
test_dataset = NERDataset('msra/test.txt', word_to_ix, tag_to_ix, max_len=200)
X_test, y_test = test_dataset.sentences, test_dataset.tags
y_pred = predict_labels(model, X_test)
y_true = np.reshape(y_test, (-1,))
y_pred = np.reshape(y_pred, (-1,))
f1 = f1_score(y_true, y_pred, average='weighted')
print('Test F1 score:', f1)

