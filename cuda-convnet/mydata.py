#-------------------------------------------------------------------------------
# Name:        mydata
# Purpose:
#
# Author:      Xiuyu
#
# Created:     05/11/2013
# Copyright:   (c) Xiuyu 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#!/usr/bin/env python
import util
import numpy as np
import numpy.random as nr
import cv2
from cv2 import cv
import random
import os
from data import *

# DataProvider for applications with a mask output
class MaskDataProvider(DataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_dic = {}
        self.data_dic['data'] = []
        self.data_dic['mask'] = []
        self.data_dic['filename'] = []
        if test:
            self.patch_in_pic = 64
            self.pic_in_batch = 28
            imagefolder = r'D:\work\sunxiuyu\weizmann_horse_db\test\pic'
            maskfolder = r'D:\work\sunxiuyu\weizmann_horse_db\test\mask'
        else:
            self.patch_in_pic = 64
            self.pic_in_batch = 60
            imagefolder = r'D:\work\sunxiuyu\weizmann_horse_db\train\pic'
            maskfolder = r'D:\work\sunxiuyu\weizmann_horse_db\train\mask'
        ext = '.jpg'

        for basename in os.listdir(imagefolder):
            if os.path.splitext(basename)[1] == ext:
                imgname = os.path.join(imagefolder,basename)
                maskname = os.path.join(maskfolder,basename)
                img = cv2.imread(imgname)
                mask = cv2.imread(maskname,0)
                a,mask = cv2.threshold(mask, 200, 255,cv2.THRESH_BINARY)
                mask = mask / 255
                self.data_dic['data'].append(img)
                self.data_dic['mask'].append(mask)
                self.data_dic['filename'].append(maskname)

    def get_next_batch(self):
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        bidx = batchnum - self.batch_range[0]
        picidx = bidx * self.pic_in_batch

        patch_width,patch_height = 64, 64
        data = np.zeros((3 * 64 * 64, self.pic_in_batch * self.patch_in_pic),np.single)
        data_mask = np.zeros((64 * 64,self.pic_in_batch * self.patch_in_pic),np.single)
        for i in range(0, self.pic_in_batch):
                img = self.data_dic['data'][i + picidx]
                mask = self.data_dic['mask'][i + picidx]
                for k in range(0, self.patch_in_pic):
                    while 1:
                        startW,startH = 0,0
                        nr.random()
                        startW = nr.randint(0,img.shape[1] - patch_width)
                        startH = nr.randint(0,img.shape[0] - patch_height)
                        pImg = img[startH : startH + patch_height, startW : startW + patch_width,:]
                        pMask = mask[startH : startH + patch_height, startW : startW + patch_width]
                        num = pMask.sum()
                        if num == 0 or num > patch_width * patch_height * img.shape[2] / 4:
                            break
                    dImg = np.zeros((3,64,64),np.single)
                    dImg[0,:,:] = pImg[:,:,0]
                    dImg[1,:,:] = pImg[:,:,1]
                    dImg[2,:,:] = pImg[:,:,2]
                    dImg = dImg.reshape(64 * 64 * 3)
                    dMask = pMask.reshape(64 * 64 * 1)
                    data[:,i * self.patch_in_pic + k] = dImg[:]
                    data_mask[:,i * self.patch_in_pic + k] = dMask[:]
        return epoch, batchnum, [data, data_mask]

    def get_data_dims(self, idx=0):
        if idx == 0:
            return 64**2 * 3
        else:
            return 64**2

class CroppedCIFARDataProvider32(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)

        self.border_size = dp_params['crop_border']
        self.inner_size = 32
        self.data_mult = 1
        self.num_colors = 3

        for d in self.data_dic:
            d['data'] = np.require(d['data'], requirements='C')
            d['labels'] = np.require(np.tile(d['labels'].reshape((1, d['data'].shape[1])), (1, self.data_mult)), requirements='C')

        self.cropped_data = [np.zeros((self.get_data_dims(), self.data_dic[0]['data'].shape[1]*self.data_mult), dtype=np.single) for x in xrange(2)]

        self.batches_generated = 0
        self.data_mean = np.zeros((3,32,32))# self.batch_meta['data_mean'].reshape((3,32,32))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)

        cropped = self.cropped_data[self.batches_generated % 2]

        self.__trim_borders(datadic['data'], cropped)
        # cropped -= self.data_mean
        self.batches_generated += 1
        return epoch, batchnum, [cropped, datadic['labels']]

    def get_data_dims(self, idx=0):
        return self.inner_size**2 * 3 if idx == 0 else 1

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return np.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)

    def __trim_borders(self, x, target):
        y = x.reshape(3, 32, 32, x.shape[1])

        if self.test: # don't need to loop over cases
            target[:,:] = y.reshape((self.get_data_dims(), x.shape[1]))
        else:
            z = np.zeros((3,self.border_size * 2 + self.inner_size,self.border_size * 2 + self.inner_size))
            startzx = self.border_size
            endzx = self.border_size + self.inner_size
            #y = np.pad(y, ((0,0),(self.border_size,self.border_size),(self.border_size,self.border_size),(0,0)),'constant',constant_values=0)
            for c in xrange(x.shape[1]): # loop over cases
                if self.border_size != 0:
                    z[:,startzx :endzx,startzx :endzx] = y[:,:,:,c]
                    startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)
                    endY, endX = startY + self.inner_size, startX + self.inner_size
                    pic = z[:,startY:endY,startX:endX]
                else:
                    pic = y[:,:,:,c]
                if nr.randint(2) == 0: # also flip the image with 50% probability
                    pic = pic[:,:,::-1]
                target[:,c] = pic.reshape((self.get_data_dims(),))


class CroppedNECDataProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)

        self.border_size = dp_params['crop_border']
        self.inner_size = 64 - self.border_size*2
        self.multiview = dp_params['multiview_test'] and test
        self.num_views = 5*2
        self.data_mult = self.num_views if self.multiview else 1
        self.num_colors = 3

        for d in self.data_dic:
            d['data'] = n.require(d['data'], requirements='C')
            d['labels'] = n.require(n.tile(d['labels'].reshape((1, d['data'].shape[1])), (1, self.data_mult)), requirements='C')

        self.cropped_data = [n.zeros((self.get_data_dims(), self.data_dic[0]['data'].shape[1]*self.data_mult), dtype=n.single) for x in xrange(2)]

        self.batches_generated = 0
        self.data_mean = self.batch_meta['data_mean'].reshape((3,64,64))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))

    def rota(self,pic):
        scale = 10

        if nr.randint(6) == 0:
            scale = nr.randint(8,13)

        if scale == 10:
            return pic

        scale = scale / 10.0
        center = (pic.shape[1] / 2, pic.shape[2] / 2)
        #print scale,angle,center
        rotMat = cv2.getRotationMatrix2D( center, 0 , scale)
        pic1 = np.array(pic,np.single)
        pic1[0,:,:] = cv2.warpAffine( pic[0,:,:],rotMat,(pic.shape[1], pic.shape[2]))
        pic1[1,:,:] = cv2.warpAffine( pic[1,:,:],rotMat,(pic.shape[1], pic.shape[2]))
        pic1[2,:,:] = cv2.warpAffine( pic[2,:,:],rotMat,(pic.shape[1], pic.shape[2]))
        return pic1
    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)

        cropped = self.cropped_data[self.batches_generated % 2]

        self.__trim_borders(datadic['data'], cropped)
        cropped -= self.data_mean
        self.batches_generated += 1
        return epoch, batchnum, [cropped, datadic['labels']]

    def get_data_dims(self, idx=0):
        return self.inner_size**2 * 3 if idx == 0 else 1

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)

    def __trim_borders(self, x, target):
        y = x.reshape(3, 64, 64, x.shape[1])

        if self.test: # don't need to loop over cases
            if self.multiview:
                start_positions = [(self.border_size - 4,self.border_size - 4),  (self.border_size - 4, self.border_size + 4),
                                   (self.border_size, self.border_size),
                                  (self.border_size + 4, self.border_size - 4), (self.border_size + 4, self.border_size + 4)]
                end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
                for i in xrange(self.num_views/2):
                    pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
                    target[:,i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((self.get_data_dims(),x.shape[1]))
                    target[:,(self.num_views/2 + i) * x.shape[1]:(self.num_views/2 +i+1)* x.shape[1]] = pic[:,:,::-1,:].reshape((self.get_data_dims(),x.shape[1]))
            else:
                pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
                target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))
        else:
            for c in xrange(x.shape[1]): # loop over cases
                startY, startX = nr.randint(self.border_size - 4,self.border_size + 5), nr.randint(self.border_size - 3,self.border_size + 4)
                endY, endX = startY + self.inner_size, startX + self.inner_size
                pic = y[:,:,:, c]
                #rota and scale
                pic = self.rota(pic)
                pic = pic[:,startY:endY,startX:endX]
                if nr.randint(2) == 0: # also flip the image with 50% probability
                    pic = pic[:,:,::-1]
                target[:,c] = pic.reshape((self.get_data_dims(),))


class CroppedSTLDataProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)

        self.border_size = dp_params['crop_border']
        self.inner_size = 96 - self.border_size*2
        self.multiview = dp_params['multiview_test'] and test
        self.num_views = 10
        self.data_mult = self.num_views if self.multiview else 1
        self.num_colors = 3

        for d in self.data_dic:
            d['data'] = n.require(d['data'], requirements='C')
            d['labels'] = n.require(n.tile(d['labels'].reshape((1, d['data'].shape[1])), (1, self.data_mult)), requirements='C')

        self.cropped_data = [n.zeros((self.get_data_dims(), self.data_dic[0]['data'].shape[1]*self.data_mult), dtype=n.single) for x in xrange(2)]

        self.batches_generated = 0
        self.data_mean = self.batch_meta['data_mean'].reshape((3,96,96))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))
        #self.data_dev = self.batch_meta['data_dev'].reshape((self.num_colors,96,96))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)

        cropped = self.cropped_data[self.batches_generated % 2]

        self.__trim_borders(datadic['data'], cropped)

        cropped -= self.data_mean
        #cropped = cropped / self.data_dev
        #cropped = n.array(cropped,n.float32)

        self.batches_generated += 1
        return epoch, batchnum, [cropped, datadic['labels']]

    def get_data_dims(self, idx=0):
        return self.inner_size**2 * 3 if idx == 0 else 1

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)

    def __trim_borders(self, x, target):
        y = x.reshape(3, 96, 96, x.shape[1])
        if self.test: # don't need to loop over cases
           if self.multiview:
                start_positions = [(0,0),  (0, self.border_size*2),
                                   (self.border_size, self.border_size),
                                  (self.border_size*2, 0), (self.border_size*2, self.border_size*2)]
                end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
                for i in xrange(self.num_views/2):
                    pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
                    target[:,i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((self.get_data_dims(),x.shape[1]))
                    target[:,(self.num_views/2 + i) * x.shape[1]:(self.num_views/2 +i+1)* x.shape[1]] = pic[:,:,::-1,:].reshape((self.get_data_dims(),x.shape[1]))
           else:
                pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
                target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))
        else:
            for c in xrange(x.shape[1]): # loop over cases
                startY = nr.randint(0,self.border_size*2 + 1)
                startX = nr.randint(0,self.border_size*2 + 1)
                endY, endX = startY + self.inner_size, startX + self.inner_size
                pic = y[:,startY:endY,startX:endX, c]
                if nr.randint(2) == 0: # also flip the image with 50% probability
                    pic = pic[:,:,::-1]
                target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))


class CroppedSVHNDataProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.border_size = dp_params['crop_border']
        self.inner_size = 32 - self.border_size*2
        self.multiview = dp_params['multiview_test'] and test
        self.num_views = 10
        self.data_mult = self.num_views if self.multiview else 1
        self.num_colors = 3
        self.batch_indexs = [x for x in range(0,len(self.data_dic))]
        self.random_channel = False and not test
        self.mean_test = False
        self.shuffle = False
        #self.data_mean = self.batch_meta['data_mean'].reshape((3,32,32))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))

        for d in self.data_dic:
            d['data'] = n.require(d['data'], requirements='C')
            d['labels'] = n.require(n.tile(d['labels'].reshape((1, d['data'].shape[1])), (1, self.data_mult)), requirements='C')

        self.cropped_data = [n.zeros((self.get_data_dims(), self.data_dic[0]['data'].shape[1]*self.data_mult), dtype=n.single) for x in xrange(2)]
        self.batches_generated = 0

    def get_next_batch(self):

        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)
        cropped = self.cropped_data[self.batches_generated % 2]

        #cropped -= self.data_mean

        shuffle_indexs = [z for z in range(0, cropped.shape[1])]
        if self.shuffle:
            import random
            random.shuffle(shuffle_indexs)

        datadic['data'][:,:]=datadic['data'][:,shuffle_indexs]
        datadic['labels'][:,:] = datadic['labels'][:,shuffle_indexs]

        self.__trim_borders(datadic['data'], cropped)
        self.batches_generated += 1

        return epoch, batchnum, [cropped, datadic['labels']]

    def get_data_dims(self, idx=0):
        return self.inner_size**2 * self.num_colors if idx == 0 else 1

    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)

    def rota(self,pic):
        scale = 10
        angle = 0

        if nr.randint(10) == 0:
            scale = nr.randint(5,9)
        if nr.randint(10) == 0:
            angle = nr.randint(-5,6)

        if self.random_channel:
            channel_indexs = [0,1,2,0,1,2,0,1,2]
            import random
            random.shuffle(channel_indexs)
            pic[0,:,:] = pic[channel_indexs[0],:,:]
            pic[1,:,:] = pic[channel_indexs[1],:,:]
            pic[2,:,:] = pic[channel_indexs[2],:,:]

        if angle == 0 and scale == 10:
            return pic

        scale = scale / 10.0
        angle = angle * 1.6
        center = (pic.shape[1] / 2 + nr.randint(-7,7), pic.shape[2] / 2 + nr.randint(-7,7))
        #print scale,angle,center
        rotMat = cv2.getRotationMatrix2D( center, angle , scale)
        pic1 = np.array(pic,np.single)
        pic1[0,:,:] = cv2.warpAffine( pic[0,:,:],rotMat,(pic.shape[1], pic.shape[2]))
        pic1[1,:,:] = cv2.warpAffine( pic[1,:,:],rotMat,(pic.shape[1], pic.shape[2]))
        pic1[2,:,:] = cv2.warpAffine( pic[2,:,:],rotMat,(pic.shape[1], pic.shape[2]))

        return pic1

    def __trim_borders(self, x, target):
        import random
        random.random()
        y = x.reshape(3, 32, 32, x.shape[1])
        if self.test: # don't need to loop over cases
           if self.multiview:
                start_positions = [(0,0),  (0, self.border_size*2),
                                   (self.border_size, self.border_size),
                                  (self.border_size*2, 0), (self.border_size*2, self.border_size*2)]
                end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
                for i in xrange(self.num_views/2):
                    pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
                    target[:,i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((self.get_data_dims(),x.shape[1]))
                    target[:,(self.num_views/2 + i) * x.shape[1]:(self.num_views/2 +i+1)* x.shape[1]] = pic[:,:,::-1,:].reshape((self.get_data_dims(),x.shape[1]))
           else:
                pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
                target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))
                if self.mean_test:
                    for c in xrange(target.shape[1]):
                        z = target[:,c].reshape((self.num_colors,self.inner_size * self.inner_size))
                        mean = (z[0,:] + z[1,:] + z[2,:]) / 3.0
                        z[0,:] = z[1,:] = z[2,:] = mean
                        target[:,c] = z.reshape((self.get_data_dims(),))
        else:
            for c in xrange(x.shape[1]): # loop over cases
                startY = nr.randint(0,self.border_size*2 + 1)
                startX = nr.randint(0,self.border_size*2 + 1)
                #startX = nr.randint(self.border_size / 2,self.border_size*2 + 1 - self.border_size / 2)
                endY, endX = startY + self.inner_size, startX + self.inner_size
                pic = y[:,startY:endY,startX:endX, c]
                #rota and scale
                pic1 = self.rota(pic)
                target[:,c] = pic1.reshape((self.get_data_dims(),))


class SVHNDataProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 3
        self.img_size = 32
        self.batch_indexs = [x for x in range(0,len(self.data_dic))]
        self.random_channel = True and not test
        self.mean_test = True and test
        self.shuffle = True and not test
        self.aug = False and not test
        # Subtract the mean from the data and make sure that both data and
        # labels are in single-precision floating point.
        for d in self.data_dic:
            d['data'] = n.require( d['data'], dtype=n.single, requirements='C')
            d['labels'] = n.require(d['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C')

        self.rota_data = n.zeros((self.get_data_dims(), self.data_dic[0]['data'].shape[1]), dtype=n.single)

    def get_data_dims(self, idx=0):
        return self.inner_size**2 * self.num_colors if idx == 0 else 1

    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)

    def rota(self,pic):
        scale = 10
        angle = 0

        if nr.randint(11) == 0:
            scale = nr.randint(4,9)
        if nr.randint(8) == 0:
            angle = nr.randint(-5,6)

        if angle == 0 and scale == 5:
            return pic

        scale = scale / 10.0
        angle = angle * 1.6
        center = (pic.shape[1] / 2 + nr.randint(-8,8), pic.shape[2] / 2 + nr.randint(-8,8))
        #print scale,angle,center
        rotMat = cv2.getRotationMatrix2D( center, angle , scale)
        #cv2.imshow("-0",np.array(pic[0,:,:] + 128,np.uint8))
        #cv2.imshow("-1",np.array(pic[1,:,:] + 128,np.uint8))
        #cv2.imshow("-2",np.array(pic[2,:,:] + 128,np.uint8))
        pic1 = np.array(pic,np.single)
        pic1[0,:,:] = cv2.warpAffine( pic[0,:,:],rotMat,(pic.shape[1], pic.shape[2]))
        pic1[1,:,:] = cv2.warpAffine( pic[1,:,:],rotMat,(pic.shape[1], pic.shape[2]))
        pic1[2,:,:] = cv2.warpAffine( pic[2,:,:],rotMat,(pic.shape[1], pic.shape[2]))


        #print scale
        #cv2.imshow("0",np.array(pic1[0,:,:]+ 128,np.uint8))
        #cv2.imshow("1",np.array(pic1[1,:,:]+ 128,np.uint8))
        #cv2.imshow("2",np.array(pic1[2,:,:]+ 128,np.uint8))
        #cv2.waitKey(0)
        return pic1

    def get_next_batch(self):
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        LabeledMemoryDataProvider.advance_batch(self)
        bidx = batchnum - self.batch_range[0]
        #print '---------------------',bidx
        if bidx == 0 and self.shuffle:
            import random
            random.random()
            random.shuffle(self.batch_indexs)
        #print self.batch_indexs
        datadic = self.data_dic[self.batch_indexs[bidx]]
        indexs = [x for x in range( 0,datadic['data'].shape[1] )]
        if self.shuffle:
            import random
            random.random()
            random.shuffle(indexs)

        self.rota_data[:,:] = datadic['data'][:,:] = datadic['data'][:,indexs]
        datadic['labels'][:,:] = datadic['labels'][:,indexs]

        tindex = [0,1,2,0,1,2,0,1,2]
        x = datadic['data']
        y = x.reshape(self.num_colors, 32 * 32, x.shape[1])
        for idx in range(0,y.shape[-1]):
            if self.random_channel:
                random.random()
                random.shuffle(tindex)
                y[:,:,idx] = y[ [tindex[0], tindex[1],tindex[2]], :, idx]
            if self.mean_test:
                meantest = ( y[0,:,idx] + y[1,:,idx] + y[2,:,idx]) / 3.0
                y[0,:,idx] = meantest
                y[1,:,idx] = meantest
                y[2,:,idx] = meantest
            if self.aug:
                self.rota_data[:,idx] = self.rota(y[:,:,idx].reshape(3,32,32)).reshape(3*32*32)
            else:
                self.rota_data[:,idx] = y[:,:,idx].reshape(3*32*32)
                #self.rota_data[:,idx] =
                #print y[0,:,idx]
                #print 'aaaaaa-----------------------------'
                #print datadic['data'].reshape(self.num_colors, 32 * 32, x.shape[1])[0,:,idx]
                #print 'bbbbbb-----------------------------'

        return epoch, batchnum, [self.rota_data, datadic['labels']]

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix.
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx == 0 else 1

    def get_plottable_data(self, data):
        return n.require( data.T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)


class CroppedCIFARDataChannelsProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)

        self.border_size = dp_params['crop_border']
        self.inner_size = 32 - self.border_size*2
        self.multiview = dp_params['multiview_test'] and test
        self.num_views = 5*2
        self.data_mult = self.num_views if self.multiview else 1
        self.num_colors = 9
        for d in self.data_dic:
            d['data'] = n.require(d['data'], requirements='C')
            d['labels'] = n.require(n.tile(d['labels'].reshape((1, d['data'].shape[1])), (1, self.data_mult)), requirements='C')

        self.cropped_data = [n.zeros((self.get_data_dims(), self.data_dic[0]['data'].shape[1]*self.data_mult), dtype=n.single) for x in xrange(2)]

        self.batches_generated = 0
        #self.data_mean = self.batch_meta['data_mean'].reshape((self.num_colors,32,32))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)

        cropped = self.cropped_data[self.batches_generated % 2]

        self.__trim_borders(datadic['data'], cropped)
        #cropped -= self.data_mean
        self.batches_generated += 1
        return epoch, batchnum, [cropped, datadic['labels']]#, datadic['filenames']

    def get_data_dims(self, idx=0):
        return self.inner_size**2 * self.num_colors if idx == 0 else 1

    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)

    def __trim_borders(self, x, target):
        y = x.reshape(self.num_colors, 32, 32, x.shape[1])

        if self.test: # don't need to loop over cases
            if self.multiview:
                start_positions = [(0,0),  (0, self.border_size*2),
                                   (self.border_size, self.border_size),
                                  (self.border_size*2, 0), (self.border_size*2, self.border_size*2)]
                end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
                for i in xrange(self.num_views/2):
                    pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
                    target[:,i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((self.get_data_dims(),x.shape[1]))
                    target[:,(self.num_views/2 + i) * x.shape[1]:(self.num_views/2 +i+1)* x.shape[1]] = pic[:,:,::-1,:].reshape((self.get_data_dims(),x.shape[1]))
            else:
                pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
                target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))
        else:
            for c in xrange(x.shape[1]): # loop over cases
                startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)
                endY, endX = startY + self.inner_size, startX + self.inner_size
                pic = y[:,startY:endY,startX:endX, c]
                if nr.randint(2) == 0: # also flip the image with 50% probability
                    pic = pic[:,:,::-1]
                target[:,c] = pic.reshape((self.get_data_dims(),))

def getfiles(imagefolder,numclass,outfolder,ext):
    filenames = []
    #read filenames to list
    for i in range(0, int(numclass)):
        classfolder = os.path.join(imagefolder,str(i))
        filenames.append([])
        for basename in os.listdir(classfolder):
            if os.path.splitext(basename)[1] == ext:
                filename = os.path.join(classfolder,basename)
                filenames[i].append((filename,basename,int(i)))
    #shuffle the list
    for i in range(0,int(numclass)):
        random.random()
        random.shuffle(filenames[i])
    return filenames

def make_batch(imagefolder,numclass,outfolder,ext='.bmp',sep=1, prev=None):
    batchname = 'data_batch_'
    #filenames -- [][(filename,basename,class)]
    filenames = getfiles(imagefolder,numclass,outfolder,ext)
    width,height = 32,32
    # for each batches
    for i in range(0,sep):
        filenames_by_batches =[]
        # seperate the data into batches
        for j in range(0, int(numclass)):
            import math
            step = int ( math.ceil( len(filenames[j]) * 1.0 / sep))
            #print step
            fbegin = step * i
            fend = step * (i + 1)
            if fend > len(filenames[j]) :
                fend = len(filenames[j])
            #print fbegin,fend
            fs = filenames[j][fbegin:fend]
            filenames_by_batches += fs
        #random_shuffle
        random.shuffle(filenames_by_batches)
        batch_data = []
        batch_filenames= []
        batch_labels = []
        bname = batchname + str(i + 1)
        #read img
        for f in filenames_by_batches:
            img = cv2.imread(f[0],1)
            width,height = img.shape[0],img.shape[1]
            #img = cv2.resize(img,(width,height))
            #bgr -> gray
            #gray = cv2.cvtColor(img,cv.CV_BGR2GRAY)
            #gray = cv2.GaussianBlur(gray,(3,3),0)
            #dx = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize = 3)
            #dy = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize = 3)
            #dm = cv2.convertScaleAbs(cv2.cartToPolar(dx,dy)[0])
            #cv2.imshow('a',img)
            if prev is not None:
                img = prev(img)
            #cv2.imshow('b',img)
            #print img
            img = np.array(img,np.single)
            b = img[:,:,0].reshape(width * height,order='C')
            g = img[:,:,1].reshape(width * height,order='C')
            r = img[:,:,2].reshape(width * height,order='C')
            #print b
            #print g
            #print r
            #raise
            #cv2.waitKey(0)
            #dm = dm.reshape(width * height,order='C')
            #im = np.array([b,g,r,dm]).reshape(width * height * 4,order='C')
            im = np.array([b,g,r]).reshape(width * height * 3,order='C')
            batch_data.append(im)
            batch_filenames.append(f[1])
            batch_labels.append(f[2])
        savefile = os.path.join(outfolder,bname)
        batch_data = np.transpose(np.array(batch_data,order='C'))
        print(len(batch_labels))
        print(len(batch_data))
        dict = {'batch_label':bname,'labels':batch_labels,'data':batch_data,'filenames':batch_filenames}
        util.pickle(savefile,dict)

    pass
