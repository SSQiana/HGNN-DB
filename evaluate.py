import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score


def svm_test(X, y, test_sizes=(0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99), repeat=10):
    random_states = [182318 + i for i in range(repeat)]
    result_macro_f1_list = []
    result_micro_f1_list = []
    for test_size in test_sizes:
        macro_f1_list = []
        micro_f1_list = []
        for i in range(repeat):#遍历重复次数
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_states[i])#按照test_sizes进行分割测试集
            svm = LinearSVC(dual=False)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            print(i, macro_f1,micro_f1)
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)
        result_macro_f1_list.append((np.mean(macro_f1_list), np.std(macro_f1_list)))#记录重复repeat次数下的均值和标准差
        result_micro_f1_list.append((np.mean(micro_f1_list), np.std(micro_f1_list)))
    return result_macro_f1_list, result_micro_f1_list#返回的是均值和标准差的列表

def kmeans_test(X, y, n_clusters, repeat=10):#聚类数，也就是分成3类
    nmi_list = []
    ari_list = []
    for _ in range(repeat):#迭代进行训练次数
        kmeans = KMeans(n_clusters=n_clusters)#进行KMeans聚类
        y_pred = kmeans.fit_predict(X)#预测
        nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')#计算标准互信息值，NMI【0,1】
        ari_score = adjusted_rand_score(y, y_pred)#调整的兰德系数，ARI【-1,1】
        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
    return np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)#返回NMI和ARI的均值及标准差


def evaluate_results_nc(embeddings, labels, num_classes):#输入的是测试集的嵌入，测试集的标签，类别的数量
    repeat = 20
    print('SVM test')
    svm_macro_f1_list, svm_micro_f1_list = svm_test(embeddings, labels, repeat=repeat)#经过SVM进行测试20次，返回每次的评价列表

    print('Macro-F1: ' + ', '.join(['{:.4f}~{:.4f}({:.2f})'.format(macro_f1_mean, macro_f1_std, train_size) for
                                    (macro_f1_mean, macro_f1_std), train_size in
                                    zip(svm_macro_f1_list, [0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01])]))
    print('Micro-F1: ' + ', '.join(['{:.4f}~{:.4f}({:.2f})'.format(micro_f1_mean, micro_f1_std, train_size) for
                                    (micro_f1_mean, micro_f1_std), train_size in
                                    zip(svm_micro_f1_list, [0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01])]))
    print('\nK-means test')#得到的嵌入进行K-means测试
    nmi_mean, nmi_std, ari_mean, ari_std = kmeans_test(embeddings, labels, num_classes, repeat=repeat)
    print('NMI: {:.6f}~{:.6f}'.format(nmi_mean, nmi_std))
    print('ARI: {:.6f}~{:.6f}'.format(ari_mean, ari_std))

    macro_mean = [x for (x, y) in svm_macro_f1_list]
    micro_mean = [x for (x, y) in svm_micro_f1_list]
    # nmi_mean, ari_mean = 0, 0
    return np.array(macro_mean), np.array(micro_mean), nmi_mean, ari_mean#返回各参数的均值
