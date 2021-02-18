import itertools
import matplotlib.pyplot as plt
import numpy as np
# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Confusion_Matrix_' + '.png')
    plt.close()

cls_num=5
labels=[1,1,3,4,1,2,1,0,4,2,2,1,3,4,1,1,1,0,4,2,0,1,3,4,1,4,1,0,4,2,0,1,3,4,1,3,1,0,4,2]
predicted=[0,1,3,4,1,1,1,0,2,3,0,1,3,4,1,1,1,0,2,3,0,1,3,4,1,1,1,0,2,3,0,1,3,4,1,1,1,0,2,3]
# 第一步：创建混淆矩阵
# 获取类别数，创建 N*N 的零矩阵
conf_mat = np.zeros([cls_num, cls_num])
# 第二步：获取真实标签和预测标签
# labels 为真实标签，通常为一个 batch 的标签
# predicted 为预测类别，与 labels 同长度
# 第三步：依据标签为混淆矩阵计数
for i in range(len(labels)):
	true_i = np.array(labels[i])
	pre_i = np.array(predicted[i])
	conf_mat[true_i, pre_i] += 1.0

attack_types = ['bird1', 'bird2', 'bird3', 'bird4', 'bird5']

plot_confusion_matrix(conf_mat, classes=attack_types, normalize=True, title='Normalized confusion matrix')