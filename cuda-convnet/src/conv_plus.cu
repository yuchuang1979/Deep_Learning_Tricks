/*
conv_plus.cu
author:				sun xiuyu
created date:		11/19/2013
lasted modified:	11/19/2013
*/
#include <conv_plus.cuh>
#include <iostream>
#include <assert.h>
#include <nvmatrix_kernels.cuh>
#include <nvmatrix.cuh>
#include <conv_util.cuh>
template<bool bp, bool add> __global__ void kShiftMap(float* image, float* target, int imgSize, int numChannels,
	int padding,int numImages, float* shiftMapX, float* shiftMapY, bool _ismirror)
{
	// img id ---> x dim
	int imgIdx = threadIdx.x + blockIdx.x * blockDim.x;
	// channel id ---> y dim
	int cIdx = threadIdx.y + blockIdx.y * blockDim.y;
	if(imgIdx >= numImages)
		return;
	if(cIdx >= numChannels)
		return;
	int outImageSize = imgSize + padding * 2;
	image  += imgIdx + imgSize * imgSize * cIdx * numImages;
	target += imgIdx + outImageSize * outImageSize * cIdx * numImages;

	shiftMapX += imgIdx;
	shiftMapY += imgIdx;
	int _shiftX = int(shiftMapX[0]);
	int _shiftY = int(shiftMapY[0]);

	for(int y = 0; y < imgSize; y++)
		for(int x =0; x < imgSize; x++)
		{
			int _x = _ismirror ? imgSize - x - 1 : x;
			int destY = y + padding + _shiftY;
			int destX = _x + padding + _shiftX;
			if(!bp)	//forward 
				if(destY >= 0 && destY < outImageSize && destX >=0 && destX < outImageSize)
				{
					target[ (destY * outImageSize + destX) * numImages ] = image[ (y * imgSize + x) * numImages ];
				}
			if(bp)
			{
				if(destY >= 0 && destY < outImageSize && destX >=0 && destX < outImageSize)
				{
					if(!add)
						image[ (y * imgSize + x) * numImages ] = target[ (destY * outImageSize + destX) * numImages ];
					else
						image[ (y * imgSize + x) * numImages ] += target[ (destY * outImageSize + destX) * numImages ];
				}
				else
				{
					if(!add)
						image[ (y * imgSize + x) * numImages ] = 0;
				}
			}

		}

}
void shiftMapBP(NVMatrix& images, NVMatrix& target, int numChannels,int padding,NVMatrix& shiftMapX, 
	NVMatrix& shiftMapY,float scaleTargets, bool _ismirror)
{
	assert(!images.isTrans());
	assert(!target.isTrans());
	int targetPixels = target.getNumRows() / numChannels;
	int numImages = target.getNumCols();
	int targetImgSize = int(sqrt(targetPixels));
	int imgSize = int( targetImgSize - padding * 2 );
	int imgPixels = imgSize * imgSize;
	images.resize(numChannels * imgPixels, numImages);
	assert(images.isContiguous());
	assert(images.getNumRows() == numChannels * imgPixels);
	dim3 threads(32,4);
	dim3 blocks( DIVUP(numImages, threads.x), DIVUP(numChannels, threads.y));
	if(scaleTargets == 0)
		kShiftMap<true,false><<<blocks,threads>>>(images.getDevData(),target.getDevData(),imgSize,numChannels, 
			padding,numImages,shiftMapX.getDevData(),shiftMapY.getDevData(),_ismirror);
	else
		kShiftMap<true,true><<<blocks,threads>>>(images.getDevData(),target.getDevData(),imgSize,numChannels, 
			padding,numImages,shiftMapX.getDevData(),shiftMapY.getDevData(),_ismirror);
}
void shiftMap(NVMatrix& images, NVMatrix& target, int numChannels,int padding,NVMatrix& shiftMapX, 
	NVMatrix& shiftMapY,float scaleTargets, bool _ismirror)
{
	assert(!images.isTrans());
	assert(!target.isTrans());
	int imgPixels = images.getNumRows() / numChannels;
	int imgSize = int(sqrt(imgPixels));
	assert(imgPixels == imgSize * imgSize);
	int numImages = images.getNumCols();
	assert(images.getNumRows() == numChannels * imgPixels);
	target.resize(numChannels * (imgSize + padding * 2) * (imgSize + padding * 2), numImages);
	assert(target.isContiguous());
	target.apply(NVMatrixOps::Zero());
	dim3 threads(32,4);
	dim3 blocks( DIVUP(numImages, threads.x), DIVUP(numChannels, threads.y));
	if(scaleTargets == 0)
		kShiftMap<false,false><<<blocks,threads>>>(images.getDevData(),target.getDevData(),imgSize,numChannels, 
			padding,numImages,shiftMapX.getDevData(),shiftMapY.getDevData(),_ismirror);
	else
		kShiftMap<false,true><<<blocks,threads>>>(images.getDevData(),target.getDevData(),imgSize,numChannels, 
			padding,numImages,shiftMapX.getDevData(),shiftMapY.getDevData(),_ismirror);
}
void transChannelById(NVMatrix& images, NVMatrix& target, int numChannels, int channelId, int num)
{
	assert(!images.isTrans());
	assert(!target.isTrans());
	assert(channelId < numChannels && channelId >= 0);
	assert(num <= numChannels);
	
	int imgPixels = images.getNumRows() / numChannels;
	int imgSize = int(sqrt(imgPixels));
	assert(imgPixels == imgSize * imgSize);
	int numImages = images.getNumCols();
	assert(images.getNumRows() == numChannels * imgPixels);
	target.resize(num * imgPixels, numImages);
	assert(target.isContiguous());
	if(num + channelId <= numChannels)
		images.copy(target,channelId * imgPixels, (channelId + num) * imgPixels, 0, numImages, 0, 0);
	else
	{
		images.copy(target,channelId * imgPixels, numChannels * imgPixels, 0, numImages, 0, 0);
		images.copy(target,0 , ((num + channelId) % numChannels) * imgPixels, 0, numImages, imgPixels * (num - ( numChannels - channelId )),0);
	}

}