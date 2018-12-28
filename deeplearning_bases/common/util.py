# coding: utf-8
import numpy as np


def smooth_curve(x):
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]

#打乱训练数据
def shuffle_dataset(x, t):
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t

#image to column 从图像到矩阵
def im2col(input_data, filter_h, filter_w, stride=1, pad=0): #输入数据（数据量，通道，高，长）滤波器高、宽  步幅 填充
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1 #输出的高
    out_w = (W + 2*pad - filter_w)//stride + 1 #输出的宽

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant') #填充处理
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    #按高度和宽度分拆，滤波器的第y+1层
    for y in range(filter_h):
        #步幅乘输出高度+y
        y_max = y + stride * out_h
        #第x+1列
        for x in range(filter_w):
            x_max = x + stride*out_w
            #因为col[:,y,x,:]已经把第三个和第四个维度确定了，需要赋值过去的是后两个维度
            #col和img的前两个维度都是N和C，所以不用broadcast就能匹配上
            #而col的后两个冒号则是通过y:y_max:stride, x:x_max:stride来确定的
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    #transpose转换多维数组的轴的顺序,reshape转换行为N*out_h*out_w,列数由系统计算
    #transpose转化后，前三个相乘为N*out_h*out_w,列为C*filter_h*filter_w,一通道+二通道+三通道
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_w
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]


    return img[:, :, pad:H + pad, pad:W + pad]


#测试im2col
# x1 = np.random.rand(10,3,7,7)
# col2 = im2col(x1,5,5,stride=1,pad=0)
# print(col2.shape)