import numpy
import numpy as np
import os
import pywt
import torch
import scipy.signal as signal
from  scipy.fftpack import fft, fftshift,ifft
from scipy.fftpack import dct

train_data = np.load("D:/BaiduNetdiskDownload/gjbc/train/10type_sort_train_data_8192.npy")
print(train_data)
print("*"*50)
print(train_data[0])
print(train_data.shape[1])
print("*"*50)
print(train_data.shape)

train_label = np.load("D:/BaiduNetdiskDownload/gjbc/train/10type_sort_train_label_8192.npy")
print(train_label)
print("#"*50)
print(train_label[0])
print("#"*50)

val_data = np.load("D:/BaiduNetdiskDownload/gjbc/val/10type_sort_eval_data_8192.npy")
print(val_data.shape)
val_label = np.load("D:/BaiduNetdiskDownload/gjbc/val/10type_sort_eval_label_8192.npy")

test_data = np.load("D:/BaiduNetdiskDownload/gjbc/test/10type_sort_test_data_8192.npy")
print(test_data.shape)

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
    return fd

train_output_p1 = create_folder("D:/BaiduNetdiskDownload/gjbc/feature6/train")
#train_output_p2 = create_folder("/Users/ouyangkangdemacbook/Desktop/高级编程作业 数据集/gjbc/feature/train/power")
val_output_p1 = create_folder("D:/BaiduNetdiskDownload/gjbc/feature6/val")
#val_output_p2 = create_folder("/Users/ouyangkangdemacbook/Desktop/高级编程作业 数据集/gjbc/feature/val/power")
test_output_p1 = create_folder("D:/BaiduNetdiskDownload/gjbc/feature6/test")
print(train_output_p1)

lent = train_data.shape[1]
def Energy(data):
    energy = []
    sum = 0
    for i in range(lent):
        sum =sum +(int(data[i])*int(data[i]))
        if (i+1) % 256 ==0 :
            energy.append(sum)
            sum = 0
        elif i ==len(data)-1:
            energy.append(sum)

    return energy

def Power_spec(data1,data2, size=256):
    #data1 = np.vstack((data1[0],data1[4257],data1[14338],data1[18608],data1[35795],
    #                   data1[40541],data1[46878],data1[57176],data1[60870],data1[65928]))

    power_spec = np.zeros(shape=(1,4128))
    for i in range(32):
        data = data2[i*size:(i+1)*size]
        #x = data1[i*size:(i+1)*size]
        #cor1 = np.correlate(data,data,'same')
        #cor2 = np.correlate(data_class,data,'same')
        spec1 = fft(data, size)
        #spec2 = fft(x,size)
        #spec3 = fft(x+data,size)
        #spec4 = spec3-spec2-spec1
        spec1 = np.abs(spec1)
        spec1 = spec1*spec1
        #spec4 = np.abs(spec4)
        #spec1 = np.log2(spec2)
        #spec2 = np.log2(spec2)
        spec1 = spec1 / np.max(spec1)
        #spce4 = spec4 / np.max(spec4)
        #spec2 = spec2 / np.max(spec2)
        #spec = list(spec)
        power_spec[0,i*129:(i+1)*129] = spec1[0:129]
        #power_spec[0,i*258+129:(i+1)*258] = spec4[0:129]
    #spec1 = fft(data, size)
    #spec = np.abs(spec1)
    #spec = spec*spec
    #power_spec = spec/np.max(spec)
    #power_spec = power_spec[0:2049]
    #arg = np.imag(spec1)
    #arg = arg[0:2049]

    return power_spec

def Sgn(data):
    if data >= 0:
        return 1
    else:
        return 0

def Zerocross(data):
    zerocross = []
    sum = 0
    for i in range(lent):
        sum =sum +np.abs(Sgn(data[i])-Sgn(data[i-1]))
        if (i + 1) % 256 == 0:
            zerocross.append(sum)
            sum = 0
        elif i == len(data) - 1:
            zerocross.append(sum)

    return zerocross

def Amplitude_diff(data):
    amplitude_diff = []
    sum = 0
    for i in range(lent-1):
        sum = sum + np.abs(int(data[i+1])-int(data[i]))
        if (i+1) % 256 ==0 :
            amplitude_diff.append(sum)
            sum = 0
        elif i ==len(data)-1:
            amplitude_diff.append(sum)

    return amplitude_diff

def Dct_amp(data):
    dct_coe1 = dct(data)
    dct_amp = []
    sum = 0
    for i in range(lent-1):
        sum = sum + dct_coe1[i]
        if (i+1) % 256 ==0 :
            dct_amp.append(sum)
            sum = 0
        elif i ==len(data)-1:
            dct_amp.append(sum)
    return dct_amp

def Dct_ind(data):
    dct_coe1 = dct(data)
    dct_coe2 = sorted(dct_coe1)
    dct_coe2 = dct_coe2[::-1]
    k = dct_coe2[64]
    ind = np.where(dct_coe1>k)
    ind1 = list(ind)
    dct_ind = ind1[0]
    return dct_ind

def Peak_amp(data):
    data1 = sorted(data)
    data1 = data1[::-1]
    peak_amp = np.sum(data1[0:5])/5
    return peak_amp

def Peak_ind(data):
    data1 = sorted(data)
    data1 = data1[::-1]
    k = data1[5]
    ind = np.where(data>k)
    ind1 = list(ind)
    peak_ind = ind1[0]
    peak_ind = np.mean(peak_ind[1:len(peak_ind)]-peak_ind[0:len(peak_ind)-1])

    return peak_ind

def Dwt(data):
    w = pywt.Wavelet('sym3')
    cA, cD = pywt.dwt(data,wavelet=w)
    dwt_coe = np.hstack((cA,cD))
    return dwt_coe

def Wavedec(data):
    coeffs = pywt.wavedec(data,'db1',level=2)
    cA2,cD2,cD1 = coeffs
    wavedec_coe = cA2
    return wavedec_coe




line = train_data.shape[0]
print(line)
val_line = val_data.shape[0]
print(val_data.shape[0])
test_line = test_data.shape[0]
print(test_line)

#folder_name = 'ener'
#train_output_feature_1 = os.mkdir(os.path.join(train_output_path, folder_name))

train_feature = numpy.zeros(shape=(line,1088))
val_feature = numpy.zeros(shape=(val_line,1088))
test_feature = numpy.zeros(shape=(test_line,1088))
feature_imp = np.load('D:/BaiduNetdiskDownload/gjbc/feature6/feature_imp.npy')
print(len(feature_imp))
for index in range(line):
    #energy = Energy(train_data[index])
    #data1 = train_data[index]
    #for i in range(16):
    power_spec = Power_spec(train_data[14339],train_data[index])
    #zerocross = Zerocross(train_data[index])
    #amplitudediff = Amplitude_diff(train_data[index])
    #dct_amp = Dct_amp(train_data[index])
    #dct_ind = Dct_ind(train_data[index])
    #peak_amp = Peak_amp(train_data[index])
    #peak_ind = Peak_ind(train_data[index])
    #wavedec_coe = Wavedec(train_data[index])
    print(index)
    ave = sum(feature_imp) / len(feature_imp)
    inds = np.where(feature_imp > ave)
    power_spec = power_spec[0][inds]
    #print(power_spec.shape)
    train_feature[index] = power_spec
    #print(len(train_feature[index]))
    #train_feature[index] = np.hstack((energy,dct_amp,dct_ind,peak_amp,peak_ind))

np.save(os.path.join(train_output_p1, 'trainfeature' + '.npy'), train_feature)


for index in range(val_line):
    #val_energy = Energy(val_data[index])
    #data1 = val_data[index]
    #for i in range(16):
    #    data = data1[i:i+512]
    #    val_power_spec = 0
    #    sum = Power_spec(data)
    #    val_power_spec = sum+val_power_spec
    #val_power_spec = val_power_spec/16
    val_power_spec = Power_spec(train_data[14339],val_data[index])
    #val_zerocross = Zerocross(val_data[index])
    #val_amplitudediff = Amplitude_diff(val_data[index])
    #val_dct_amp = Dct_amp(val_data[index])
    #val_dct_ind = Dct_ind(val_data[index])
    #val_peak_amp = Peak_amp(val_data[index])
    #val_peak_ind = Peak_ind(val_data[index])
    #val_wavedec_coe = Wavedec(val_data[index])
    print(index)
    ave = sum(feature_imp) / len(feature_imp)
    inds = np.where(feature_imp > ave)
    val_power_spec = val_power_spec[0][inds]
    val_feature[index] = val_power_spec
    #val_feature[index] = np.hstack((val_energy,val_dct_amp,val_dct_ind,val_peak_amp,val_peak_ind))


np.save(os.path.join(val_output_p1, 'valfeature'  + '.npy'), val_feature)

for index in range(test_line):
    #test_energy = Energy(test_data[index])
    test_power_spec = Power_spec(train_data[14339],test_data[index])
    #test_zerocross = Zerocross(test_data[index])
    #test_amplitudediff = Amplitude_diff(test_data[index])
    #test_dct_amp = Dct_amp(test_data[index])
    #test_dct_ind = Dct_ind(test_data[index])
    #test_peak_amp = Peak_amp(test_data[index])
    #test_peak_ind = Peak_ind(test_data[index])
    #test_wavedec_coe = Wavedec(test_data[index])
    print(index)
    ave = sum(feature_imp) / len(feature_imp)
    inds = np.where(feature_imp > ave)
    test_power_spec = test_power_spec[0][inds]
    test_feature[index] = test_power_spec
    #test_feature[index] = np.hstack((test_energy,test_power_spec,test_zerocross,test_amplitudediff,test_dct_amp,test_dct_ind,test_peak_amp,test_peak_ind))


np.save(os.path.join(test_output_p1, 'testfeature'  + '.npy'), test_feature)
