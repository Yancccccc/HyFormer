import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat
from scipy.io import savemat
from torch import optim
from torch.autograd import Variable
from vit_pytorch import ViT
from sklearn.metrics import confusion_matrix
from skimage import io
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import time
import os
from random_select_point import creat_dataset
from vit_CNN import ViT_CNN
from vit_only_CA import ViT_CA
from vit_CA_CNN import ViT_CA_CNN
from CNN_vit_2_res import cnn_vit_2_res
from vit_MLP import ViT_MLP
from MLP_CNN import mlpcnn
parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['changxing','2022','2020','2018'], default='changxing', help='dataset to use')
parser.add_argument('--flag_test', choices=['test', 'train'], default='test', help='testing mark')
parser.add_argument('--mode', choices=['ViT', 'CAF','ViT_CNN','ViT_MLP','MLPCNN','cnn_vit_2_res'],
                    default='cnn_vit_2_res', help='mode choice')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=32, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
parser.add_argument('--patches', type=int, default=1, help='number of patches')
parser.add_argument('--band_patches', type=int, default=2, help='number of related band')
parser.add_argument('--epoches', type=int, default=300, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--num_classes', type=int, default=9, help='num_classes')
parser.add_argument('--num_band', type=int, default=4, help='num_band')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# 边界拓展：镜像  # 逐像素就不用扩展，默认pach=1
def mirror_hsi(height, width, band, input_normalize, patch=1):
    padding = patch//2
    mirror_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)
    # 中心区域
    mirror_hsi[padding:(padding + height), padding:(padding + width), :] = input_normalize
    # 左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height + padding), i, :] = input_normalize[:, padding - i - 1, :]
    # 右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height + padding), width + padding + i, :] = input_normalize[:, width - 1 - i, :]
    # 上边镜像
    for i in range(padding):
        mirror_hsi[i, :, :] = mirror_hsi[padding * 2 - i - 1, :, :]
    # 下边镜像
    for i in range(padding):
        mirror_hsi[height + padding + i, :, :] = mirror_hsi[height + padding - 1 - i, :, :]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0], mirror_hsi.shape[1], mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi
# -------------------------------------------------------------------------------
# 获取patch的图像数据
def gain_neighborhood_pixel(mirror_image, point, i, patch=1):
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]  #(5,5,:)
    return temp_image

def gain_neighborhood_band(x_train, band, band_patch, patch=1):
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float)
    # 中心区域
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape
    #左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]
    #右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band
# -------------------------------------------------------------------------------
# 汇总训练数据和测试数据
def train_and_test_data(mirror_image, band, train_point, test_point, val_point,true_point, patch=1, band_patch=2):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    x_val = np.zeros((val_point.shape[0], patch, patch, band), dtype=float)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i, :, :, :] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j, :, :, :] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    for l in range(val_point.shape[0]):
        x_val[l, :, :, :] = gain_neighborhood_pixel(mirror_image, val_point, l, patch)
    for k in range(true_point.shape[0]):
        x_true[k,:, :, :] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape, x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape, x_test.dtype))
    print("x_val  shape = {}, type = {}".format(x_val.shape, x_val.dtype))
    print("x_true  shape = {}, type = {}".format(x_true.shape, x_test.dtype))
    print("**************************************************")
    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    x_val_band = gain_neighborhood_band(x_val, band, band_patch, patch)
    x_true_band = gain_neighborhood_band(x_true, band, band_patch, patch)
    print("x_train_band shape = {}, type = {}".format(x_train_band.shape, x_train_band.dtype))
    print("x_test_band  shape = {}, type = {}".format(x_test_band.shape, x_test_band.dtype))
    print("x_val_band  shape = {}, type = {}".format(x_val_band.shape, x_val_band.dtype))
    print("x_true_band  shape = {}, type = {}".format(x_true_band.shape, x_true_band.dtype))
    print("**************************************************")
    return x_train_band, x_test_band, x_val_band, x_true_band

# -------------------------------------------------------------------------------
# 标签y_train, y_test
def train_and_test_label(number_train, number_test, number_val,number_true, num_classes):
    y_train = []
    y_test = []
    y_val = []
    y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
        for l in range(number_val[l]):
            y_val.append(l)
    for i in range(num_classes + 1):
        for j in range(number_true[i]):
            y_true.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_val = np.array(y_val)
    y_true = np.array(y_true)
    print("y_train: shape = {} ,type = {}".format(y_train.shape, y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape, y_test.dtype))
    print("y_val: shape = {} ,type = {}".format(y_val.shape, y_val.dtype))
    print("y_true: shape = {} ,type = {}".format(y_true.shape, y_true.dtype))
    print("**************************************************")
    return y_train, y_test, y_val, y_true

# -------------------------------------------------------------------------------
class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

# -------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()

# -------------------------------------------------------------------------------
# train model
def train_epoch(model, train_loader, criterion, optimizer, epochs=1400):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        optimizer.zero_grad()
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return top1.avg, objs.avg, tar, pre, model, epochs
# -------------------------------------------------------------------------------
# validate model
def valid_epoch(model, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        batch_pred = model(batch_data)

        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return tar, pre

def tesdepoch(model, test_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        batch_pred = model(batch_data)

        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre

#-------------------------------------------------------------------------------
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA,matrix

#-------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA
#------------------------------------------------------------------------

def label2image(prelabel,colormap):
    #预测的标签转化为图像，针对一个标签图
    h,w = prelabel.shape
    prelabel = prelabel.reshape(h*w,-1)
    image = np.zeros((h*w,3),dtype="int32")
    for i in range(len(colormap)):
        index = np.where(prelabel == i)
        image[index,:] = colormap[i]
    return image.reshape(h, w, 3)
#---------------------------------------------------------
# Parameter Setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False
# prepare data
if args.dataset == 'changxing':
    data = io.imread('data/changxing/2021/changxing_img_pixel_255.tif')
elif args.dataset == '2018':
    data = io.imread('data/changxing/2018/20180509_pixel_255.tif')
   # data = data.transpose(1, 2, 0)
elif args.dataset == '2020':
    data = io.imread('data/changxing/2020/20200801_pixel_255.tif')
   # data = data.transpose(1, 2, 0)
elif args.dataset == '2022':
    data = io.imread('data/changxing/2022/20220727_pixel_255.tif')
  #  data = data.transpose(1, 2, 0)
else:
    raise ValueError("Unkknow dataset")
colormap = [[255, 255, 255], [216, 191, 216], [0, 0, 255], [0, 139, 0], [255, 0, 255], [255, 255, 0], [255, 165, 0], [0, 255, 0], [255, 70, 5]]
class_names = ['background', 'bulidup', 'water', 'tree', 'pond', 'rice', 'othercrop', 'platation', 'greenhouse']

# data size
height, width, band  = data.shape
print("height={0},width={1},band={2}".format(height, width, band))
#-----------------------------------------------------------------------------
if args.dataset == 'changxing':
    txt_path = 'xy/changxing/2021/2022_8_28_1.txt'
    train_xy_correct = np.load('xy/changxing/2021/train_xy.npy')
    test_xy_correct = np.load('xy/changxing/2021/test_xy.npy')
    val_xy_correct = np.load('xy/changxing/2021/val_xy.npy')
    train_xy_correct = train_xy_correct.astype(int)
    test_xy_correct = test_xy_correct.astype(int)
    val_xy_correct = val_xy_correct.astype(int)
    train_label = np.load('xy/changxing/2021/train_label.npy')
    test_label = np.load('xy/changxing/2021/test_label.npy')
    val_label = np.load('xy/changxing/2021/val_label.npy')

elif args.dataset == 'changxing_NDVI_NDWI':
    txt_path = 'xy/changxing_NDVI_NDWI/NDVI_NDWI_10M4.txt'
    train_xy_correct = np.load('xy/changxing/2021/train_xy.npy')
    test_xy_correct = np.load('xy/changxing/2021/test_xy.npy')
    val_xy_correct = np.load('xy/changxing/2021/val_xy.npy')
    train_xy_correct = train_xy_correct.astype(int)
    test_xy_correct = test_xy_correct.astype(int)
    val_xy_correct = val_xy_correct.astype(int)
    train_label = np.load('xy/changxing/2021/train_label.npy')
    test_label = np.load('xy/changxing/2021/test_label.npy')
    val_label = np.load('xy/changxing/2021/val_label.npy')

elif args.dataset == '2018':
    txt_path = 'xy/changxing/2018/20180618_2.txt'
    train_xy_correct = np.load('xy/changxing/2018/train_xy_2.npy')
    test_xy_correct = np.load('xy/changxing/2018/test_xy_2.npy')
    val_xy_correct = np.load('xy/changxing/2018/val_xy_2.npy')
    train_xy_correct = train_xy_correct.astype(int)
    test_xy_correct = test_xy_correct.astype(int)
    val_xy_correct = val_xy_correct.astype(int)
    train_label = np.load('xy/changxing/2018/train_label_2.npy')
    test_label = np.load('xy/changxing/2018/test_label_2.npy')
    val_label = np.load('xy/changxing/2018/val_label_2.npy')

elif args.dataset == '2020':
    txt_path = 'xy/changxing/2020/20200801_3.txt'
    train_xy_correct = np.load('xy/changxing/2020/train_xy_2.npy')
    test_xy_correct = np.load('xy/changxing/2020/test_xy_2.npy')
    val_xy_correct = np.load('xy/changxing/2020/val_xy_2.npy')
    train_xy_correct = train_xy_correct.astype(int)
    test_xy_correct = test_xy_correct.astype(int)
    val_xy_correct = val_xy_correct.astype(int)
    train_label = np.load('xy/changxing/2020/train_label_2.npy')
    test_label = np.load('xy/changxing/2020/test_label_2.npy')
    val_label = np.load('xy/changxing/2020/val_label_2.npy')
elif args.dataset == '2022':
    txt_path = 'xy/changxing/2022/20220727_2.txt.txt'
    train_xy_correct = np.load('xy/changxing/2022/train_xy_2.npy')
    test_xy_correct = np.load('xy/changxing/2022/test_xy_2.npy')
    val_xy_correct = np.load('xy/changxing/2022/val_xy_2.npy')
    train_xy_correct = train_xy_correct.astype(int)
    test_xy_correct = test_xy_correct.astype(int)
    val_xy_correct = val_xy_correct.astype(int)
    train_label = np.load('xy/changxing/2022/train_label_2.npy')
    test_label = np.load('xy/changxing/2022/test_label_2.npy')
    val_label = np.load('xy/changxing/2022/val_label_2.npy')
#train_xy_correct, train_label, test_xy_correct, test_label, val_xy_correct, val_label = creat_dataset(txt_path) #先在命令窗口运行该代码，保存坐标及标签

# train_xy_correct = np.load('xy/changxing/2021/train_xy.npy')
# test_xy_correct = np.load('xy/changxing/2021/test_xy.npy')
# val_xy_correct = np.load('xy/changxing/2021/val_xy.npy')
# train_xy_correct = train_xy_correct.astype(int)
# test_xy_correct = test_xy_correct.astype(int)
# val_xy_correct = val_xy_correct.astype(int)


true_xy = np.zeros([data.shape[0]*data.shape[1],2])

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        true_xy[i*data.shape[1]+j, 0] = i
        true_xy[i*data.shape[1]+j, 1] = j

true_xy = true_xy.astype(int)
# train_label = np.load('xy/changxing/2021/train_label.npy')
# test_label = np.load('xy/changxing/2021/test_label.npy')
# val_label = np.load('xy/changxing/2021/val_label.npy')

train_label = train_label.astype(int)
test_label = test_label.astype(int)
val_label = val_label.astype(int)

mirror_image = mirror_hsi(height, width, band, data, patch=args.patches)
for i in range(mirror_image.shape[2]):
    input_max = np.max(mirror_image[:, :,i])
    input_min = np.min(mirror_image[:, :,i])
    mirror_image[:, :,i] = (mirror_image[:, :,i] - input_min) / (input_max - input_min)

x_train_band, x_test_band, x_val_band, x_true_band = train_and_test_data(mirror_image=mirror_image, band=band, train_point=train_xy_correct,
                                                             test_point=test_xy_correct, true_point=true_xy, val_point=val_xy_correct,patch=args.patches, band_patch=args.band_patches)

y_train = train_label
y_test = test_label
y_val = val_label
y_true = np.ones([x_true_band.shape[0]]).astype(int) # y_true随便设置，只要尺寸和图片大小一样就行，不影响。
# -------------------------------------------------------------------------------
# load data
x_train = torch.from_numpy(x_train_band.transpose(0, 2, 1)).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)
Label_train = Data.TensorDataset(x_train, y_train)

x_test = torch.from_numpy(x_test_band.transpose(0, 2, 1)).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)
Label_test = Data.TensorDataset(x_test, y_test)

x_val = torch.from_numpy(x_val_band.transpose(0, 2, 1)).type(torch.FloatTensor)
y_val = torch.from_numpy(y_val).type(torch.LongTensor)
Label_val = Data.TensorDataset(x_val, y_val)

x_true=torch.from_numpy(x_true_band.transpose(0,2,1)).type(torch.FloatTensor)
y_true = torch.from_numpy(y_true).type(torch.LongTensor)
Label_true = Data.TensorDataset(x_true, y_true)

label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
label_val_loader = Data.DataLoader(Label_val, batch_size=args.batch_size, shuffle=True, num_workers=0)
label_test_loader = Data.DataLoader(Label_test, batch_size=args.batch_size, shuffle=True, num_workers=0)
label_true_loader = Data.DataLoader(Label_true, batch_size=60, shuffle=False, num_workers=0) # shuffle=False 不然像素点的顺序都打乱了
# -------------------------------------------------------------------------------
# create model
if args.mode == 'ViT' or args.mode == 'CAF':
    model = ViT(
        image_size=args.patches,
        near_band=args.band_patches,
        num_patches=band,
        num_classes=args.num_classes,
        dim=64,
        depth=5,
        heads=4,
        mlp_dim=8,
        dropout=0.1,
        emb_dropout=0.1,
        mode=args.mode
    )
elif args.mode == 'ViT_CNN':
    model = ViT_CNN(
        image_size=args.patches,
        near_band=args.band_patches,
        num_patches=band,
        num_classes=args.num_classes,
        dim=64,
        depth=5,
        heads=4,
        mlp_dim=8,
        dropout=0.1,
        emb_dropout=0.1,
        mode='ViT',
        channels=args.num_band,
        num_features=args.band_patches*args.num_band
    )

elif args.mode == 'ViT_MLP':
    model = ViT_MLP(
        image_size=args.patches,
        near_band=args.band_patches,
        num_patches=band,
        num_classes=args.num_classes,
        dim=64,
        depth=5,
        heads=4,
        mlp_dim=8,
        dropout=0.1,
        emb_dropout=0.1,
        mode='ViT',
        channels=args.num_band,
    )
  
elif args.mode == 'cnn_vit_2_res':
    model = cnn_vit_2_res(
        image_size=args.patches,
        near_band=args.band_patches,
        num_patches=band,
        num_classes=args.num_classes,
        dim=64,
        depth=5,
        heads=4,
        mlp_dim=8,
        dropout=0.1,
        emb_dropout=0.1,
        mode='ViT',
        channels=args.num_band
)
   
elif args.mode == 'MLPCNN':
    model = mlpcnn(num_features=args.band_patches * args.num_band)
model = model.cuda()
# criterion
criterion = nn.CrossEntropyLoss().cuda()
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma)
# -------------------------------------------------------------------------------
model_name = args.mode
#os.makedirs('checpoint/{}'.format(model_name), exist_ok=True)
os.makedirs('checpoint/{}/{}'.format(args.dataset,model_name), exist_ok=True)
os.makedirs('out/{}/{}'.format(args.dataset,model_name), exist_ok=True)
pre_path = 'out/{}/{}'.format(args.dataset,model_name)
if args.flag_test == 'test':
    if args.mode == 'ViT':
        model.load_state_dict(torch.load('checpoint/{}/ViT/best_model_bandpatches{}.pth'.format(args.dataset,args.band_patches)))
    elif (args.mode == 'CAF') & (args.patches == 1):
        model.load_state_dict(torch.load('checpoint/{}/CAF/best_model_bandpatches{}.pth'.format(args.dataset,args.band_patches)))
    elif (args.mode == 'CAF') & (args.patches == 7):
        model.load_state_dict(torch.load('./SpectralFormer_patch.pt'))
    elif args.mode == 'ViT_CNN':
        model.load_state_dict(torch.load('checpoint/{}/ViT_CNN/best_model_bandpatches{}.pth'.format(args.dataset,args.band_patches)))
    elif args.mode == 'ViT_MLP':
        model.load_state_dict(torch.load('checpoint/{}/ViT_MLP/best_model_bandpatches{}.pth'.format(args.dataset, args.band_patches)))
    elif args.mode == 'MLPCNN':
        model.load_state_dict(torch.load('checpoint/{}/MLPCNN/best_model_bandpatches{}.pth'.format(args.dataset, args.band_patches)))
    else:
        raise ValueError("Wrong Parameters")
    model.eval()
    tar_v, pre_v = valid_epoch(model, label_val_loader, criterion, optimizer)
    OA2, AA_mean2, Kappa2, AA2 , matrix = output_metric(tar_v, pre_v)

    # output classification maps
    pre_u = tesdepoch(model, label_true_loader, criterion, optimizer)
    prediction_matrix = np.zeros((mirror_image.shape[0], mirror_image.shape[1]), dtype=float)
    for i in range(true_xy.shape[0]):
        prediction_matrix[true_xy[i, 0], true_xy[i, 1]] = pre_u[i]


    # mask = np.zeros([mirror_image.shape[0], mirror_image.shape[1]])
    # for i in range(mirror_image.shape[0]):
    #     for j in range(mirror_image.shape[1]):
    #         if mirror_image[i,j,1] != 0:
    #             mask[i,j] = 1
    #         else:
    #             mask[i,j] = 0
    # prediction_matrix = prediction_matrix * mask

    pre = label2image(prediction_matrix, colormap=colormap)
    plt.subplot(1, 1, 1)
    plt.imshow(pre)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    #savemat('./out/VIT/2011/20110312.mat', {'P': prediction_matrix})  #保存成.mat文件
    pre_name = "{}_bandpatchs{}.jpeg".format(model_name,args.band_patches)
    pre_outpath = os.path.join(pre_path, pre_name)
    io.imsave(pre_outpath, pre, quality=100)
    ###保存预测的label为npy方便后续土地利用变化作图
    np_name = "{}_bandpatchs{}.npy".format(model_name, args.band_patches)
    pre_np_path = os.path.join(pre_path, np_name)
    np.save(pre_np_path,prediction_matrix)

elif args.flag_test == 'train':
    best_OA = 0
    print("start training")
    tic = time.time()
    for epoch in range(args.epoches):
        # train model
        model.train()
        train_acc, train_obj, tar_t, pre_t, net, epochs = train_epoch(model, label_train_loader, criterion, optimizer, epochs=args.epoches)
        OA1, AA_mean1, Kappa1, AA1, matrix = output_metric(tar_t, pre_t)

        print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}".format(epoch + 1, train_obj, train_acc))

        if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):
            model.eval()
            tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
            OA2, AA_mean2, Kappa2, AA2, matrix = output_metric(tar_v, pre_v)

        if OA2 > best_OA:
            best_OA = OA2
            torch.save(net.state_dict(), r'./checpoint/{}/{}/best_model_bandpatches{}.pth'.format(args.dataset,model_name,args.band_patches))
        scheduler.step()
    toc = time.time()
    print("Running Time: {:.2f}".format(toc - tic))
    print("**************************************************")

print("Final result:")
print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
print(AA2)
print(matrix)
print("**************************************************")
print("Parameter:")


def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k, v))


print_args(vars(args))











