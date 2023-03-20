def bm25(query, documents, k1=1.5, b=0.75):
    """
    BM25算法实现
    :param query: 查询字符串
    :param documents: 文档列表，每个元素为一个字符串
    :param k1: 调节因子，一般取值为1.2-2.0之间
    :param b: 调节因子，一般取值为0.75
    :return: 包含每个文档得分的列表
    """
    # 文档数量
    N = len(documents)
    # 文档平均长度
    avgdl = sum(len(d) for d in documents) / N
    # 单词频率
    freqs = []
    for d in documents:
        freq = {}
        for word in d.split():
            freq[word] = freq.get(word, 0) + 1
        freqs.append(freq)
    # 查询单词频率
    qfreq = {}
    for word in query.split():
        qfreq[word] = qfreq.get(word, 0) + 1
    # 计算每个文档的得分
    scores = []
    for i in range(N):
        score = 0
        for word in query.split():
            if word not in freqs[i]:
                continue
            idf = math.log((N - len([d for d in freqs if word in d]) + 0.5) / (len([d for d in freqs if word in d]) + 0.5))
            tf = freqs[i][word]
            score += idf * ((k1 + 1) * tf) / (k1 * ((1 - b) + b * (len(documents[i].split()) / avgdl)) + tf)
        scores.append(score)
    return scores
