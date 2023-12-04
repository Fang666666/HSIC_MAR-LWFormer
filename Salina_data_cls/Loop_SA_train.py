

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from skimage.feature import local_binary_pattern
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv

import mar_lwformer
import time

from util import print_results


def resolve_dict(hp):
    return hp['run_times'], hp['class_num'], hp['patch_size']

def loadData():
    # 读入数据
    ##### 本机数据集
    # data = sio.loadmat('../../../../../data/Salinas_corrected.mat')['salinas_corrected']
    # labels = sio.loadmat('../../../../../data/Salinas_gt.mat')['salinas_gt']
    ##### ssh数据集
    data = sio.loadmat('/home/data/fy/Datasets/hsi_data/Salinas_corrected.mat')['salinas_corrected']
    labels = sio.loadmat('/home/data/fy/Datasets/hsi_data/Salinas_gt.mat')['salinas_gt']

    return data, labels

# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX

# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):

    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX

# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):

    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1

    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        # test_size=testRatio,
                                                        train_size= testRatio,
                                                        random_state=randomState,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test

BATCH_SIZE_TRAIN = 64

def create_data_loader(patch_size):
    # 地物类别
    # class_num = 16
    # 读入数据
    X, y0 = loadData()
    # 用于测试样本的比例
    # train_ratio = 3921
    # test_ratio = 0.95
    train_ratio = 0.005
    # 每个像素周围提取 patch 的尺寸
    patch_size = patch_size
    # # 使用 PCA 降维，得到主成分的数量
    pca_components = 30
    # bands = 103

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y0.shape)

    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)
    # X_emap = torch.from_numpy(torch.squeeze(applyPCA(torch.from_numpy(sio.loadmat('../generateEMAP/Feature_E_Salinas.mat')['Feature_E']), numComponents=1)))
    mm = sio.loadmat('../generateEMAP/Feature_E_Salinas.mat')['Feature_E']
    print("mm", mm.shape)
    mmm = applyPCA(mm, numComponents=1)
    print("mmm", mmm.shape)
    mmmm = torch.squeeze(torch.from_numpy(mmm))
    X_emap = mmmm
    X_lbp = torch.from_numpy(local_binary_pattern(image=torch.squeeze(torch.from_numpy(applyPCA(X, numComponents=1))), P=8, R=1))
    X_emap = X_emap.detach().unsqueeze(dim=2).repeat(1, 1, 30)
    X_lbp = X_lbp.detach().unsqueeze(dim=2).repeat(1, 1, 30)
    print('\n... ... create data cubes ... ...')
    X_pca, y = createImageCubes(X_pca, y0, windowSize=patch_size)
    X_emap, y_emap = createImageCubes(X_emap, y0, windowSize=patch_size)
    X_lbp, y_lbp = createImageCubes(X_lbp, y0, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube X shape: ', X_emap.shape)
    print('Data cube X shape: ', X_lbp.shape)
    print('Data cube y shape: ', y.shape)

    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y, train_ratio)
    Xtrain_emap, Xtest_emap, ytrain_emap, ytest_emap = splitTrainTestSet(X_emap, y_emap, train_ratio)
    Xtrain_lbp, Xtest_lbp, ytrain_lbp, ytest_lbp = splitTrainTestSet(X_lbp, y_lbp, train_ratio)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtrain_emap shape: ', Xtrain_emap.shape)
    print('Xtrain_lbp shape: ', Xtrain_lbp.shape)
    print('Xtest  shape: ', Xtest.shape)

    # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
    X = X_pca.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    X_emap = X_emap.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtrain_emap = Xtrain_emap.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest_emap = Xtest_emap.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain_emap shape: ', Xtrain_emap.shape)
    print('before transpose: Xtest_emap  shape: ', Xtest_emap.shape)


    X_lbp = X_lbp.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtrain_lbp = Xtrain_lbp.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest_lbp = Xtest_lbp.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain_lbp shape: ', Xtrain_lbp.shape)
    print('before transpose: Xtest_lbp  shape: ', Xtest_lbp.shape)

    # 为了适应 pytorch 结构，数据要做 transpose
    # X = X_pca.transpose(0, 3, 1, 2)
    # Xtrain = Xtrain.transpose(0, 3, 1, 2)
    # Xtest = Xtest.transpose(0, 3, 1, 2)
    X = X.transpose(0, 4, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    X_emap = X_emap.transpose(0, 4, 3, 1, 2)
    Xtrain_emap = Xtrain_emap.transpose(0, 4, 3, 1, 2)
    Xtest_emap = Xtest_emap.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain_emap shape: ', Xtrain_emap.shape)
    print('after transpose: Xtest_emap  shape: ', Xtest_emap.shape)

    X_lbp = X_lbp.transpose(0, 4, 3, 1, 2)
    Xtrain_lbp = Xtrain_lbp.transpose(0, 4, 3, 1, 2)
    Xtest_lbp = Xtest_lbp.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain_lbp shape: ', Xtrain_lbp.shape)
    print('after transpose: Xtest_lbp  shape: ', Xtest_lbp.shape)
    X = np.concatenate((X, X_emap, X_lbp), axis=1)
    Xtrain = np.concatenate((Xtrain, Xtrain_emap, Xtrain_lbp), axis=1)
    Xtest = np.concatenate((Xtest, Xtest_emap, Xtest_lbp), axis=1)


    # X = torch.cat((torch.from_numpy(X), torch.from_numpy(X_lbp)), dim=1)
    # Xtrain = torch.cat((torch.from_numpy(Xtrain), torch.from_numpy(Xtrain_lbp)), dim=1)
    # Xtest = torch.cat((torch.from_numpy(Xtest), torch.from_numpy(Xtest_lbp)), dim=1)
    print('after transpose: X shape: ', X.shape)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)






    # 创建train_loader和 test_loader
    X = TestDS(X, y)
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True,
                                               drop_last=True
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=False,
                                               num_workers=0,
                                               drop_last=True
                                              )
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                batch_size=BATCH_SIZE_TRAIN,
                                                shuffle=False,
                                                num_workers=0,
                                                drop_last=True
                                              )

    return train_loader, test_loader, all_data_loader, y

""" Training dataset"""

class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):

        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):

        # 返回文件数据的数目
        return self.len

""" Testing dataset"""

class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):

        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):

        # 返回文件数据的数目
        return self.len

def train(train_loader, epochs):

    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # 网络放到GPU上
    net = mar_lwformer.Mar_LWFormer().to(device)
    # summary(net, input_size=(3, 19, 19))
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 初始化优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # 开始训练
    total_loss = 0
    for epoch in range(epochs):
        net.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # 正向传播 +　反向传播 + 优化
            # 通过输入得到预测的输出
            outputs = net(data)
            # 计算损失函数
            loss = criterion(outputs, target)
            # 优化器梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1,
                                                                         total_loss / (epoch + 1),
                                                                         loss.item()))
        if total_loss / (epoch + 1) < 0.415:
            break
    print('Finished Training')

    return net, device

def test(device, net, test_loader):
    count = 0
    # 模型测试
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test

def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):

    target_names = ['1', '2', '3', '4', '5', '6', '7',
                    '8', '9', '10', '11', '12', '13', '14', '15', '16']
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa, confusion, each_acc, aa, kappa

def get_classification_map_labels(y_pred, y):

    height = y.shape[0]
    width = y.shape[1]
    k = 0
    cls_labels = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            target = int(y[i, j])
            if target == 0:
                continue
            else:
                cls_labels[i][j] = y_pred[k]+1
                k += 1

    return  cls_labels

def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 1101:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 0:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 1:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 2:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 3:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 4:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 5:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 6:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 7:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 8:
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 9:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 10:
            y[index] = np.array([128, 0, 128]) / 255.
        if item == 11:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 12:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 13:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 14:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 15:
            y[index] = np.array([0, 168, 132]) / 255.

    return y

def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1]*2.0/dpi, ground_truth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0


def SA_loop_train_test(hyper_parameter):
    run_times, class_num, patch_size = resolve_dict(hyper_parameter)

    train_loader, test_loader, all_data_loader, y_all = create_data_loader(patch_size)


    OA = []
    AA = []
    KAPPA = []
    TRAINING_TIME = []
    TESTING_TIME = []
    ELEMENT_ACC = np.zeros((run_times, class_num))

    for run_i in range(0, run_times):
        print('round:', run_i + 1)
        print('>' * 10, "Start Training", '<' * 10)
        tic1 = time.perf_counter()
        net, device = train(train_loader, epochs=400)
        # # 只保存模型参数
        # torch.save(net.state_dict(),'Sa_params.pth')
        #
        # params = list(net.parameters())  # 所有参数放在params里
        # k = 0
        # for i in params:
        #     l = 1
        #     for j in i.size():
        #         l *= j  # 每层的参数存入l，这里也可以print 每层的参数
        #     k = k + l  # 各层参数相加
        # print("all params:" + str(k))  # 输出总的参数
        #
        # # input = torch.randn(64, 3, 30, 13, 13).to(device)
        # # flops, params = profile(net, inputs=(input,))
        # # flops, params = clever_format([flops, params], "%.3f")
        # # print(flops, params)

        toc1 = time.perf_counter()
        tic2 = time.perf_counter()
        print('>' * 10, "Start Testing", '<' * 10)
        y_pred_test, y_test = test(device, net, test_loader)
        toc2 = time.perf_counter()
        # 评价指标
        classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
        print('MSTNet')
        print('OA: ', oa)
        print('AA: ', aa)
        print('Kappa: ', kappa)
        classification = str(classification)
        Training_Time = toc1 - tic1
        Test_time = toc2 - tic2

        OA.append(oa)
        AA.append(aa)
        KAPPA.append(kappa)
        TRAINING_TIME.append(toc1 - tic1)
        TESTING_TIME.append(toc2 - tic2)
        ELEMENT_ACC[run_i, :] = each_acc

        # file_name = "cls_result/0.005-10-UP_classification_report.txt"
        # with open(file_name, 'w') as x_file:
        #     x_file.write('{} Training_Time (s)'.format(Training_Time))
        #     x_file.write('\n')
        #     x_file.write('{} Test_time (s)'.format(Test_time))
        #     x_file.write('\n')
        #     x_file.write('{} Overall accuracy (%)'.format(oa))
        #     x_file.write('\n')
        #     x_file.write('{} Average accuracy (%)'.format(aa))
        #     x_file.write('\n')
        #     x_file.write('{} Kappa accuracy (%)'.format(kappa))
        #     x_file.write('\n')
        #     x_file.write('{} Each accuracy (%)'.format(each_acc))
        #     x_file.write('\n')
        #     x_file.write('{}'.format(classification))
        #     x_file.write('\n')
        #     x_file.write('{}'.format(confusion))


        # print('-------Save the result in mat format--------')
        #
        # y_pred, y_new = test(device, net, all_data_loader)
        # y_part = np.array(y_all[42560:])
        # y_pred = np.append(y_pred,y_part)
        # X, y = loadData()
        # cls_labels = get_classification_map_labels(y_pred, y)
        # sio.savemat('SA_TSNE1.mat', {'cls_labels': cls_labels})
        # sio.savemat('SA_TSNE2.mat', {'y': y})
        #
        # x = np.ravel(cls_labels)
        # for i in range(len(x)):
        #     if x[i] == 0:
        #         x[i] = 17
        # x = x[:] - 1
        #
        # gt = y.flatten()
        # for i in range(len(gt)):
        #     if gt[i] == 0:
        #         gt[i] = 17
        # gt = gt[:] - 1
        #
        # y_list = list_to_colormap(x)
        # y_gt = list_to_colormap(gt)
        #
        # y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))
        # gt_re = np.reshape(y_gt, (y.shape[0],y.shape[1], 3))
        # classification_map(y_re, y, 300,
        #                    'classification_maps/' + 'SA-JFMAFormer_0.005.eps')
        # classification_map(y_re, y, 300,
        #                    'classification_maps/'+'SA-JFMAFormer_0.005.png')
        # classification_map(gt_re, y, 300,
        #                    'classification_maps/' + 'SA_gt.png')
        # print('------Get classification maps successful-------')


    print_results(class_num, np.array(OA), np.array(AA), np.array(KAPPA), np.array(ELEMENT_ACC),
                  np.array(TRAINING_TIME), np.array(TESTING_TIME))
