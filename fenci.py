# 前向最大匹配分词
def forward_maximum_matching(text, dictionary, max_len):
    words = []  # 分词结果
    text_len = len(text)
    index = 0  # 扫描位置
    while index < text_len:
        for i in range(max_len, 0, -1):
            if index + i > text_len:
                continue
            word = text[index:(index + i)]
            if word in dictionary:
                words.append(word)
                index += i
                break
        else:
            words.append(text[index])
            index += 1
    return words

# 后向最大匹配分词
def backward_maximum_matching(text, dictionary, max_len):
    words = []  # 分词结果
    index = len(text)  # 扫描位置
    while index > 0:
        for i in range(max_len, 0, -1):
            if index - i < 0:
                continue
            word = text[(index - i):index]
            if word in dictionary:
                words.insert(0, word)
                index -= i
                break
        else:
            index -= 1
    return words

# 双向最大匹配分词
def bidirectional_maximum_matching(text, dictionary, max_len):
    words = []  # 分词结果
    text_len = len(text)
    index = 0  # 扫描位置
    while index < text_len:
        for i in range(max_len, 0, -1):
            if index + i > text_len:
                continue
            word = text[index:(index + i)]
            if word in dictionary:
                words.append(word)
                index += i
                break
        else:
            words.append(text[index])
            index += 1
    return words




dictionary = set(['中国', '国家', '主席', '习近平', '发表', '新年', '贺词','快乐'])
text = '习近平主席发表了新年贺词，祝愿全国人民新年快乐！'
result = bidirectional_maximum_matching(text, dictionary, 4)
print(result)



import jieba

text = '习近平主席发表了新年贺词，祝愿全国人民新年快乐！'
result=jieba.lcut(text,cut_all=False)
print(result)




