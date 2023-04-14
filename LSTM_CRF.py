import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, TimeDistributed,Reshape
from tf2crf import CRF, ModelWithCRFLoss
# from tf2crf import CRF, ModelWithCRFLoss
# 
# from keras_contrib.layers import CRF
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score
from tensorflow.keras.optimizers import Adam


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
        X = tf.keras.utils.to_categorical(self.sentences[idx], num_classes=len(word_to_ix))
        y = tf.keras.utils.to_categorical(self.tags[idx], num_classes=len(tag_to_ix))
        return X, y




def create_model(max_len, word_vocab_size, tag_vocab_size, embedding_dim, hidden_dim):
    inputs = Input(shape=(max_len,))
    embeddings = Embedding(input_dim=word_vocab_size, output_dim=embedding_dim, input_length=max_len)(inputs)
    lstm = Bidirectional(LSTM(units=hidden_dim, return_sequences=True))(embeddings)
    dense = TimeDistributed(Dense(tag_vocab_size))(lstm)

    crf = CRF(dtype=tf.float32)
    output = crf(dense)

    base_model = Model(inputs, output)
    model = ModelWithCRFLoss(base_model, sparse_target=True)
    model.compile(optimizer=Adam(learning_rate=0.001))
    return model


def train_model(train_dataset, val_dataset, word_vocab_size, tag_vocab_size, embedding_dim, hidden_dim, batch_size, epochs):
    max_len = train_dataset.sentences.shape[1]
    model = create_model(max_len, word_vocab_size, tag_vocab_size, embedding_dim, hidden_dim)
    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    X_train = train_dataset.sentences
    y_train = train_dataset.tags
    X_val = val_dataset.sentences
    y_val = val_dataset.tags

    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs, callbacks=[early_stop])

    return model

# 预测标签序列
def predict_labels(model, X_test):
    y_pred = []
    for x in X_test:
        x_cat = np.expand_dims(tf.keras.utils.to_categorical(x, num_classes=len(word_to_ix)), axis=0)
        y = model.predict(x_cat)
        y_pred.append(np.argmax(y, axis=-1))
    y_pred = np.squeeze(np.array(y_pred), axis=-1)
    return y_pred



# 定义标签到索引的映射
tag_to_ix = {'O': 0, 'B-LOC': 1, 'I-LOC': 2, 'B-PER': 3, 'I-PER': 4, 'B-ORG': 5, 'I-ORG': 6}
num_tags = len(tag_to_ix)
optimizer = Adam(learning_rate=0.001)
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

train_dataset=NERDataset('msra/test.txt',word_to_ix,tag_to_ix,max_len=200)


