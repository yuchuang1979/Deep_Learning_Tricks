#-------------------------------------------------------------------------------
# Name:        模块1
# Purpose:
#
# Author:      Xiuyu
#
# Created:     30/10/2013
# Copyright:   (c) Xiuyu 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#!/usr/bin/env python
import util
import convdata as cd
from data import *
import numpy as np
import cv2
from cv2 import cv
import mydata as m

def cifar100toimages():
    file = r'D:\work\sunxiuyu\cifar-100-python\test'
    outfolder = r'D:\work\sunxiuyu\cifar-10-py-colmajor\tmp'
    dict = util.unpickle(file)
    numlcass = np.array(dict['fine_labels'],np.int).max()
    fine_classes = {}
    mean = np.zeros(32*32*3,dtype=np.double)
    for i in range(0,len(dict['data'])):
        mean += dict['data'][i]
    mean = mean / len(dict['data'])

    metafile = r'D:\work\sunxiuyu\cifar-10-py-colmajor\cifar10\batches.meta'
    outmetafile = r'D:\work\sunxiuyu\cifar-10-py-colmajor\cifar-100\batches.meta'
    meta = util.unpickle(metafile)
    for key in meta:
        print key
    meta['label_names'] =[str(i) for i in range(0,100)]
    meta['data_mean'] = meta['data_mean'].reshape(3072)
    meta['data_mean'][0:1024] = mean[2048:3072] #b
    meta['data_mean'][1024:2048] = mean[1024:2048] #g
    meta['data_mean'][2048:3072] = mean[0:1024] #r
    util.pickle(outmetafile,meta)
    return

    for i in range(0,len(dict['data'])):
        fine_classes_idx = dict['fine_labels'][i]
        #m_data = dict['data'][i] - mean
        m_data = dict['data'][i]
        if fine_classes_idx in fine_classes:
            fine_classes[fine_classes_idx].append({'data':m_data,'label':dict['fine_labels'][i],'filename':dict['filenames'][i]})
        else:
            fine_classes[fine_classes_idx] = [{'data':m_data,'label':dict['fine_labels'][i],'filename':dict['filenames'][i]}]
        pass

    # random shuffle
    for i in range(0,len(fine_classes)):
        indexs = range(0,len(fine_classes[i]))
        np.random.shuffle(indexs)
        fine_classes[i] = [fine_classes[i][x] for x in indexs ]

    #save image patches
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    for i in range(0,len(fine_classes)):
        class_folder = os.path.join(outfolder,str(i))
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        for image in fine_classes[i]:
            #save image patch
            m_data = image['data']
            r = m_data[0:1024]
            g = m_data[1024:2048]
            b = m_data[2048:3072]

            r = r.reshape(32,32)
            g = g.reshape(32,32)
            b = b.reshape(32,32)

            color_img = np.zeros((32,32,3),dtype=np.uint8)
            color_img[:,:,0] = b[:,:]
            color_img[:,:,1] = g[:,:]
            color_img[:,:,2] = r[:,:]

            imagefile = os.path.join(class_folder,image['filename'])
            cv2.imwrite(imagefile,color_img)
def globalnorm(img2):
    img2 = img2 - img2.mean()
    sqrtmean = np.sqrt( (img2 ** 2).sum() + 10.0 ).mean()
    img2 = img2 / sqrtmean
    return img2
def localnorm(img):
    img1 = cv2.GaussianBlur(img,(7,7),2.0)
    img2 = 1.0 * img - img1
    #img3 = np.square(img2)
    img3 = np.square(img)
    img3 = cv2.GaussianBlur(img3,(7,7),2.0)
    img3 = np.sqrt(img3)
    img4 = np.zeros((32,32,3),np.single)
    img4 = img3
    win = 3
    m0 = img4[:,:,0].mean()
    m1 = img4[:,:,1].mean()
    m2 = img4[:,:,2].mean()
    for y in range(0,32):
        for x in range(0,32):
            img4[x,y,0] = m0 if img4[x,y,0] < m0 else img4[x,y,0]
            img4[x,y,1] = m1 if img4[x,y,1] < m1 else img4[x,y,1]
            img4[x,y,2] = m2 if img4[x,y,2] < m2 else img4[x,y,2]
    img2 = img2 / img4
    return img2
def SVHNImage2Valid():
    #extrapath = r'D:\work\sunxiuyu\SVHN\pics\extra'
    path = r'D:\work\sunxiuyu\SVHN\pics\extra'
    outpath = r'D:\work\sunxiuyu\SVHN\pics\mid'
    fpath = r'D:\work\sunxiuyu\SVHN\pics\train'

    ffiles = m.getfiles(fpath,10,outpath,'.png')
    files = m.getfiles(path,10,outpath,'.png')
    for i in range(0, len(files)):
        outfolder = os.path.join(outpath,str(i))
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        lsize = 18000 - len(ffiles[i])
        for j in range(0 , lsize):
            print files[i][j][0]
            outfile = os.path.join(outfolder,files[i][j][1])
            import shutil
            #shutil.move(files[i][j][0],outfile)
            shutil.copyfile(files[i][j][0],outfile)
            #raise
def SVHNtoImages():
    svhnfile = r'D:\work\sunxiuyu\SVHN\extra_32x32.mat'
    outfolder = r'D:\work\sunxiuyu\SVHN\pics\extra'
    import scipy.io as sio
    data = sio.loadmat(svhnfile)
    num_classes = {}
    for i in range(0,len(data['y'])):
        image = data['X'][:,:,:,i]
        image1 = np.zeros((32,32,3),dtype=np.uint8)
        image1[:,:,0] = image[:,:,2]
        image1[:,:,1] = image[:,:,1]
        image1[:,:,2] = image[:,:,0]
        #image1 = localnorm(image1)
        print len(data['y']) - i
        #cv2.imshow("image",image1)
        #cv2.imshow("image1",image1[:,:,0])
        #cv2.imshow("image2",image1[:,:,2])
        #cv2.imshow("image3",image1[:,:,1])
        #cv2.imshow("norm1",norm1)
        #cv2.waitKey(0)
        label = data['y'][i]
        label = label[0]
        if label == 10:
            label = 0
        if label in num_classes:
            num_classes[label].append({'image':image1,'label':label})
        else:
            num_classes[label] = [{'image':image1,'label':label}]

    if not os.path.exists(outfolder):
            os.makedirs(outfolder)

    for i in range(0,len(num_classes)):
        class_folder = os.path.join(outfolder,str(i))
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        k = 0
        for image in num_classes[i]:
            imagefile = 'e_' + str(i) + '_' + str(k) + '.png'
            #print imagefile
            imagefile = os.path.join(class_folder,imagefile)
            cv2.imwrite(imagefile,image['image'])
            k += 1


def getfiles(imagefolder,outfolder,ext):
    filenames = []
    #read filenames to list
    for basename in os.listdir(imagefolder):
        if os.path.splitext(basename)[1] == ext:
            filename = os.path.join(imagefolder,basename)
            outfile = os.path.join(outfolder,basename)
            filenames.append((filename,outfile))
    #shuffle the list
    import random
    random.random()
    random.shuffle(filenames)
    return filenames

def preprocessing():
    folder = r'D:\work\sunxiuyu\SVHN\large-lcn'
    #outfolder = r'D:\work\sunxiuyu\SVHN\svhn-valid-large-1'
    datasize = 32 * 32 * 3
    #meta = util.unpickle(metafile)
    #mean = np.zeros(datasize,np.double)
    num = 0
    begin = 25
    for i in range(begin,begin + 16):
        batch_file = os.path.join(folder,'data_batch_' + str(i))
        print batch_file
        buffer = util.unpickle(batch_file)
        data = buffer['data']
        dim2 = len(data)
        data = np.transpose(data)
        dim1 = len(data)
        print dim1
        newbuffer = np.zeros((dim1,dim2),np.single)
        for i in range(0,len(data)):
            img1 = data[i].reshape(3,32,32)
            img = np.zeros((32,32,3),np.single)
            result = np.zeros((3,32,32),np.single)
            img[:,:,0] = img1[0,:,:]
            img[:,:,1] = img1[1,:,:]
            img[:,:,2] = img1[2,:,:]
            #cv2.imshow( "img1",img )
            #cv2.waitKey(0)
            result[0,:,:] = img[:,:,0]
            result[1,:,:] = img[:,:,1]
            result[2,:,:] = img[:,:,2]
            #print result[0,:,:]
            newbuffer[i] = result.reshape(3 * 32 * 32)
        newbuffer = np.transpose(newbuffer)
        buffer['data'] = newbuffer
        util.pickle(batch_file,buffer)
    return

colors1 = ['r-','b-','g-','c-','m-','k-','y-']
colors =  ['r*:', 'b*:','g*:','c*:','m*:','k*:','y*:']

def plot_cost():
    import pylab as pl
    models = [
    #{'name':'baseline','filename':r'D:\work\sunxiuyu\bin\nec-head\baseline\models\ConvNet__2014-02-28_17.39.21'},
    ]
    import shownet as sn
    import gpumodel as gm
    import math
    from math import sqrt, ceil, floor
    pl.figure(1)
    i = 0
    step = 50
    total_ters = 300
    testing_freq =7
    samples =7
    show_train = False
    for m in models:
        import shownet as sn
        import gpumodel as gm
        import convnet as cn
        load_dic = gm.IGPUModel.load_checkpoint(m['filename'])
        test_errors = [o[0]['logprob'][1] for o in load_dic['model_state']['test_outputs']]
        test_errors = np.row_stack(test_errors)
        test_errors = np.tile(test_errors, (1, testing_freq))
        test_errors = list(test_errors.flatten())

        train_errors = [o[0]['logprob'][1] for o in load_dic['model_state']['train_outputs']]
        train_errors = np.row_stack(train_errors)
        train_errors = list(train_errors.flatten())

        test_errors = test_errors[1:len(test_errors): samples]
        train_errors = train_errors[1:len(train_errors): samples]

        x = range(0, len(test_errors))


        pl.plot(x, test_errors, colors1[i % len(colors1)], label=m['name'] + '_test' )
        if show_train:
            pl.plot(x, train_errors, colors[i % len(colors)], label=m['name'] + '_train' )

        i += 1
        pl.legend()
        bank = int(len(test_errors) * 1.0 / total_ters * step)
        ticklocs = range(0,len(test_errors), bank)
        ticklabels = range(0,len(ticklocs))
        ticklabels =[x * step  for x in ticklabels]
        pl.xticks(ticklocs, ticklabels)
        pl.xlabel('Epoch')
        pl.ylabel('Test Error Rates')
        pl.title('probs')
    pl.show()


def main():
    plot_cost()
    return
    #SVHNImage2Valid()
    #return
    #preprocessing()
    #return

if __name__ == '__main__':
    main()
