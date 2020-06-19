#include "pool3d.cuh"

namespace NLC
{
	template <bool add> __global__ void kLocalAvgUndo1(float* avgGrads,float* target, const int imgSize, 
		const int numImages,const int numFilters, const int subsF, const int strideF,const int outputsF, 
		const int subsX, const int startX, const int strideX, const int outputsX, const float scaleTargets,
		const float scaleOutputs, int startImgIdx, float scale, const float* dropMask) {

		const int imgIdx = threadIdx.x + startImgIdx * blockDim.x;
		if(imgIdx >= numImages)
			return;

		const int numPixels = blockDim.y * ( blockIdx.y * imgSize + blockIdx.x) + threadIdx.y;
		const int pixelIdx = numPixels % (imgSize * imgSize);
		const int filterIdx = numPixels / (imgSize * imgSize);
		const int imgX = pixelIdx % imgSize;
		const int imgY = pixelIdx / imgSize;

		const int startOutputY = imgY - startX < subsX ? 0 : 1 + (imgY - startX - subsX) / strideX;
		const int endOutputY = MIN(outputsX, 1 + (imgY - startX) / strideX);
		const int startOutputX = imgX - startX < subsX ? 0 : 1 + (imgX - startX - subsX) / strideX;
		const int endOutputX = MIN(outputsX, 1 + (imgX - startX) / strideX);
		const int startOutputF = filterIdx < subsF ? 0 : 1 + (filterIdx - subsF) / strideF;
		const int endOutputF = MIN(outputsF, 1 + filterIdx / strideF);
	
		target	 += imgIdx + numPixels * numImages;
		avgGrads += imgIdx;

		dropMask += imgIdx;

		float tmp = 0.0f;

		for(int f = startOutputF; f< endOutputF; f++)
		{
			const float regionStartF = fmaxf(0, f * strideF);
			const float regionEndF = fminf(numFilters, f * strideF + subsF);
			const float regionSizeF = regionEndF - regionStartF;
			for(int y = startOutputY; y < endOutputY; y++)
			{
				const float regionStartY = fmaxf(0, startX + y * strideX);
				const float regionEndY = fminf(imgSize, startX + y * strideX + subsX);
				const float regionSizeY = regionEndY - regionStartY;
				for(int x = startOutputX; x < endOutputX; x++)
				{
					const float regionStartX = fmaxf(0, startX + x * strideX);
					const float regionEndX = fminf(imgSize, startX + x * strideX + subsX);
					const float regionSizeX = regionEndX - regionStartX;
					//const float regionSizeInv = 1.0f / (regionSizeX * regionSizeY * regionSizeF);
					/*
					compute the id of mask
					*/
					const int maskid = (imgX - regionStartX) + (imgY - regionStartY) * regionSizeX + (filterIdx - regionStartF) * regionSizeY * regionSizeX;
					const int outImgPx = f * outputsX * outputsX + y * outputsX + x;
					const float s = dropMask[outImgPx * numImages + maskid * numImages * outputsX * outputsX * outputsF ];
					const float avgmg = s > 1e-5 ? avgGrads[outImgPx * numImages] / s : 0 ;
					tmp += avgmg;
				}
			}
		}
		if(!add)
			target[0] = tmp;
		else
			target[0] = scaleTargets * target[0] + scaleOutputs * tmp;
	}

	template <bool add> __global__ void kLocalAvgUndo(float* avgGrads,float* target, const int imgSize, 
		const int numImages,const int numFilters, const int subsF, const int strideF,const int outputsF, 
		const int subsX, const int startX, const int strideX, const int outputsX, const float scaleTargets,
		const float scaleOutputs, int startImgIdx, float scale, const float* dropMask) {

		const int imgIdx = threadIdx.x + startImgIdx * blockDim.x;
		if(imgIdx >= numImages)
			return;

		const int numPixels = blockDim.y * ( blockIdx.y * imgSize + blockIdx.x) + threadIdx.y;
		const int pixelIdx = numPixels % (imgSize * imgSize);
		const int filterIdx = numPixels / (imgSize * imgSize);
		const int imgX = pixelIdx % imgSize;
		const int imgY = pixelIdx / imgSize;

		const int startOutputY = imgY - startX < subsX ? 0 : 1 + (imgY - startX - subsX) / strideX;
		const int endOutputY = MIN(outputsX, 1 + (imgY - startX) / strideX);
		const int startOutputX = imgX - startX < subsX ? 0 : 1 + (imgX - startX - subsX) / strideX;
		const int endOutputX = MIN(outputsX, 1 + (imgX - startX) / strideX);
		const int startOutputF = filterIdx < subsF ? 0 : 1 + (filterIdx - subsF) / strideF;
		const int endOutputF = MIN(outputsF, 1 + filterIdx / strideF);
	
		target	 += imgIdx + numPixels * numImages;
		avgGrads += imgIdx;

		dropMask += imgIdx;

		float tmp = 0.0f;

		for(int f = startOutputF; f< endOutputF; f++)
		{
			const float regionStartF = fmaxf(0, f * strideF);
			const float regionEndF = fminf(numFilters, f * strideF + subsF);
			const float regionSizeF = regionEndF - regionStartF;
			for(int y = startOutputY; y < endOutputY; y++)
			{
				const float regionStartY = fmaxf(0, startX + y * strideX);
				const float regionEndY = fminf(imgSize, startX + y * strideX + subsX);
				const float regionSizeY = regionEndY - regionStartY;
				for(int x = startOutputX; x < endOutputX; x++)
				{
					const float regionStartX = fmaxf(0, startX + x * strideX);
					const float regionEndX = fminf(imgSize, startX + x * strideX + subsX);
					const float regionSizeX = regionEndX - regionStartX;
					const float regionSizeInv = 1.0f / (regionSizeX * regionSizeY * regionSizeF);
					/*
					compute the id of mask
					*/
					const int maskid = (imgX - regionStartX) + (imgY - regionStartY) * regionSizeX + (filterIdx - regionStartF) * regionSizeY * regionSizeX;
					const int outImgPx = f * outputsX * outputsX + y * outputsX + x;
					const float s = dropMask[outImgPx * numImages + maskid * numImages * outputsX * outputsX * outputsF ];
					const float avgmg = (s >= scale) * avgGrads[outImgPx * numImages];
					//regionSizeInv += (s >= scale);
					tmp += avgmg * regionSizeInv;
				}
			}
		}
		//tmp = regionSizeInv > 0 ? tmp / regionSizeInv : 0;
		if(!add)
			target[0] = tmp;
		else
			target[0] = scaleTargets * target[0] + scaleOutputs * tmp;
	}
	
	template <bool add> __global__ void kLocalMaxUndo(float* imgs, float* maxGrads, float* maxActs, float* target, 
		const int imgSize, const int numImages,const int numFilters, const int subsF, const int strideF,
		const int outputsF, const int subsX, const int startX, const int strideX, const int outputsX, 
		const float scaleTargets, const float scaleOutputs, int startImgIdx) {

		const int imgIdx = threadIdx.x + startImgIdx * blockDim.x;
		if(imgIdx >= numImages)
			return;

		const int numPixels = blockDim.y * ( blockIdx.y * imgSize + blockIdx.x) + threadIdx.y;
		const int pixelIdx = numPixels % (imgSize * imgSize);
		const int filterIdx = numPixels / (imgSize * imgSize);
		const int imgX = pixelIdx % imgSize;
		const int imgY = pixelIdx / imgSize;

		const int startOutputY = imgY - startX < subsX ? 0 : 1 + (imgY - startX - subsX) / strideX;
		const int endOutputY = MIN(outputsX, 1 + (imgY - startX) / strideX);
		const int startOutputX = imgX - startX < subsX ? 0 : 1 + (imgX - startX - subsX) / strideX;
		const int endOutputX = MIN(outputsX, 1 + (imgX - startX) / strideX);
		const int startOutputF = filterIdx < subsF ? 0 : 1 + (filterIdx - subsF) / strideF;
		const int endOutputF = MIN(outputsF, 1 + filterIdx / strideF);
	
		imgs	 += imgIdx + numPixels * numImages;
		target	 += imgIdx + numPixels * numImages;
		maxActs	 += imgIdx;
		maxGrads += imgIdx;

		float timg = imgs[0];

		float tmp = 0.0f;
		for(int f = startOutputF; f< endOutputF; f++)
			for(int y = startOutputY; y < endOutputY; y++)
				for(int x = startOutputX; x < endOutputX; x++)
				{
					const int outImgPx = f * outputsX * outputsX + y * outputsX + x;
					const float ma = maxActs[outImgPx * numImages];
					const float mg = maxGrads[outImgPx * numImages];
					tmp += (timg == ma) * mg;
				}
		if(!add)
			target[0] = tmp;
		else
			target[0] = scaleTargets * target[0] + scaleOutputs * tmp;
	}
	
	/*
	 * conv3DLocalMaxUndo
	 * imgs:        (numFilters, imgPixels, numImages)
	 * maxGrads:    (outNumFilters, numOutputs, numImages)
	 * rMaxActs:    (outNumFilters, numOutputs, numImages)
	 * target:      (numFilters, imgPixels, numImages)
	 */
	void conv3DLocalMaxUndo(NVMatrix& images, NVMatrix& maxGrads, NVMatrix& maxActs, NVMatrix& target,
		int numFilters, int subsF,int strideF, int outNumFilters, int subsX, int startX, int strideX,
		int outputsX, float scaleTargets, float scaleOutput) 
	{

		int outputs = outputsX * outputsX;
		int numImages = images.getNumCols();
		assert(outNumFilters == maxGrads.getNumRows() / outputs);

		int imgPixels = images.getNumRows() / numFilters;
		assert(images.getNumRows() == numFilters * imgPixels);

		int imgSize = int(sqrt(imgPixels));

		assert(imgSize * imgSize == imgPixels);
		assert(maxGrads.getNumRows() == outNumFilters * outputs);
		assert(maxGrads.getNumCols() == numImages);
		assert(!images.isTrans());
		assert(!target.isTrans());
		assert(!maxGrads.isTrans());
		assert(!maxActs.isTrans());
		assert(images.isContiguous());
		assert(maxGrads.isContiguous());
		assert(maxActs.isContiguous());
		assert(maxGrads.isSameDims(maxActs));
		assert(numFilters % 16 == 0);
		assert(strideX <= subsX);
		target.resize(images);
		assert(target.isContiguous());

		dim3 threads(32,4);
		dim3 blocks(imgSize, imgSize * DIVUP(numFilters, threads.y));

		int numBlockImage = DIVUP(numImages, threads.x);
		for( int imgIdx = 0; imgIdx < numBlockImage; imgIdx++)
		{
			if(scaleTargets == 0 && scaleOutput == 1 )
				kLocalMaxUndo<false><<<blocks,threads>>>(images.getDevData(), maxGrads.getDevData(),
					maxActs.getDevData(), target.getDevData(), imgSize, numImages,numFilters,
					subsF,strideF, outNumFilters, subsX, startX, strideX,outputsX,
					scaleTargets,scaleOutput,imgIdx);
			else
				kLocalMaxUndo<true><<<blocks,threads>>>(images.getDevData(), maxGrads.getDevData(),
					maxActs.getDevData(), target.getDevData(), imgSize, numImages,numFilters,
					subsF,strideF, outNumFilters, subsX, startX, strideX,outputsX,
					scaleTargets,scaleOutput,imgIdx);
		}
		getLastCudaError("convLocalMaxUndo: kernel execution failed");
	}

	/*
	 * conv3DLocalAvgUndo
	 * avgGrads:    (outNumFilters, numOutputs, numImages)
	 * target:      (numFilters, imgPixels, numImages)
	 */
	void conv3DLocalAvgUndo(NVMatrix& avgGrads, NVMatrix& target,
		int numFilters, int subsF,int strideF, int outNumFilters, int subsX, int startX, int strideX,
		int outputsX, int imgSize, float scaleTargets, float scaleOutput, float scale, NVMatrix& dropMask) 
	{
		int numImages = avgGrads.getNumCols();
		int outputs = outputsX * outputsX;
		int imgPixels = imgSize * imgSize;
		assert(outNumFilters == avgGrads.getNumRows() / outputs);
		assert(avgGrads.getNumRows() == outNumFilters * outputs);
		assert(!target.isTrans());
		assert(!avgGrads.isTrans());
		assert(avgGrads.isContiguous());
		assert(numFilters % 16 == 0);
		assert(strideX <= subsX);

		target.resize(numFilters * imgPixels, numImages);
		assert(target.isContiguous());

		dim3 threads(32,4);
		dim3 blocks(imgSize, imgSize * DIVUP(numFilters, threads.y));

		int numBlockImage = DIVUP(numImages, threads.x);
		for( int imgIdx = 0; imgIdx < numBlockImage; imgIdx++)
		{
			if(scaleTargets == 0 && scaleOutput == 1 )
				kLocalAvgUndo<false><<<blocks,threads>>>(avgGrads.getDevData(),target.getDevData(), 
				imgSize, numImages,numFilters,subsF,strideF, outNumFilters, subsX, startX, strideX,
				outputsX, scaleTargets,scaleOutput,imgIdx,scale,dropMask.getDevData());
			else
				kLocalAvgUndo<true><<<blocks,threads>>>(avgGrads.getDevData(),target.getDevData(), 
				imgSize, numImages,numFilters,subsF,strideF, outNumFilters, subsX, startX, strideX,
				outputsX, scaleTargets,scaleOutput,imgIdx,scale,dropMask.getDevData());
		}
		getLastCudaError("convLocalMaxUndo: kernel execution failed");
	}


	/*
	 * conv3DLocalAvgUndo1
	 * avgGrads:    (outNumFilters, numOutputs, numImages)
	 * target:      (numFilters, imgPixels, numImages)
	 */
	void conv3DLocalAvgUndo1(NVMatrix& avgGrads, NVMatrix& target,
		int numFilters, int subsF,int strideF, int outNumFilters, int subsX, int startX, int strideX,
		int outputsX, int imgSize, float scaleTargets, float scaleOutput, float scale, NVMatrix& dropMask) 
	{
		int numImages = avgGrads.getNumCols();
		int outputs = outputsX * outputsX;
		int imgPixels = imgSize * imgSize;
		assert(outNumFilters == avgGrads.getNumRows() / outputs);
		assert(avgGrads.getNumRows() == outNumFilters * outputs);
		assert(!target.isTrans());
		assert(!avgGrads.isTrans());
		assert(avgGrads.isContiguous());
		assert(numFilters % 16 == 0);
		assert(strideX <= subsX);

		target.resize(numFilters * imgPixels, numImages);
		assert(target.isContiguous());

		dim3 threads(32,4);
		dim3 blocks(imgSize, imgSize * DIVUP(numFilters, threads.y));

		int numBlockImage = DIVUP(numImages, threads.x);
		for( int imgIdx = 0; imgIdx < numBlockImage; imgIdx++)
		{
			if(scaleTargets == 0 && scaleOutput == 1 )
				kLocalAvgUndo1<false><<<blocks,threads>>>(avgGrads.getDevData(),target.getDevData(), 
				imgSize, numImages,numFilters,subsF,strideF, outNumFilters, subsX, startX, strideX,
				outputsX, scaleTargets,scaleOutput,imgIdx,scale,dropMask.getDevData());
			else
				kLocalAvgUndo1<true><<<blocks,threads>>>(avgGrads.getDevData(),target.getDevData(), 
				imgSize, numImages,numFilters,subsF,strideF, outNumFilters, subsX, startX, strideX,
				outputsX, scaleTargets,scaleOutput,imgIdx,scale,dropMask.getDevData());
		}
		getLastCudaError("convLocalMaxUndo: kernel execution failed");
	}
}