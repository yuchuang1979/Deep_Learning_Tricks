/*
3D Pooling
Added by Sun Xiuyu
*/
#ifndef CONV_POOL3D_CUH
#define	CONV_POOL3D_CUH
#include <helper_image.h>
#include <nvmatrix.cuh>

/*
3D pooling device function 
*/
namespace NLC
{
	class MaxPooler
	{
	public:
		enum{ LoopNum = 1};
		__device__ __inline__ float4 base(float* r, size_t pitch)
		{
			return make_float4(0,0,0,0);
		}
		template<int loop> __device__ __inline__ float4 compute(float4 a , float b);
		template<> __device__ __inline__ float4 compute<1>(float4 a , float b)
		{
			if(b > a.x)
				a.x = b;
			return a;
		}
		__device__ __inline__ float result(float4 a)
		{
			return a.x;
		}
	};
	class ProbMaxPooler9
	{
		float mThreshold;
		float mScale;
	public:
		enum{ LoopNum = 9};
		ProbMaxPooler9(float threshold):mThreshold(0.0f),mScale(1.0f)
		{
			mThreshold = 1 - threshold;
		}
		__device__ __inline__ float4 base(float* r, size_t pitch)
		{
			return make_float4(0.0f,0.0f,2e33,0.0f);
		}
		template<int loop> __device__ __inline__ float4 compute(float4 a , float b)
		{
			//if(b < a.z && b >= a.x)
			//	a.x = b;
			//a.y = int(a.y) + 1;
			//if(int(a.y) == loopEndX * loopEndY * loopEndF)
			//{
			//	a.y = 0;
			//	a.z = a.x; //prev max
			//	a.x = 0;
			//	mScale *= mThreshold; 
			//	a.w += a.z * mScale;
			//}
			return a;
		}
		__device__ __inline__ float result(float4 a)
		{
			return a.w;
		}
	};
	class ProbMaxPooler
	{
		float mThreshold;
	public:
		enum{ LoopNum = 4};
		ProbMaxPooler(float threshold):mThreshold(0.0f)
		{
			mThreshold = threshold;
		}
		__device__ __inline__ float4 base(float* r, size_t pitch)
		{
			return make_float4(0.0f,0.0f,0.0f,0.0f);
		}
		template<int loop> __device__ __inline__ float4 compute(float4 a , float b);

		template<> __device__ __inline__ float4 compute<4>(float4 a , float b)
		{
			if(b > a.x)
				a.x = b;
			return a;
		}

		template<> __device__ __inline__ float4 compute<3>(float4 a , float b)
		{
			if( b < a.x && b > a.y)
				a.y = b;
			return a;
		}
		template<> __device__ __inline__ float4 compute<2>(float4 a , float b)
		{
			if( b < a.y && b > a.z)
				a.z = b;
			return a;
		}
		template<> __device__ __inline__ float4 compute<1>(float4 a , float b)
		{
			if( b < a.z && b > a.w)
				a.w = b;
			return a;
		}
		__device__ __inline__ float result(float4 a)
		{
			return a.x * ( 1 - mThreshold ) + a.y * mThreshold * ( 1 - mThreshold ) + 
				(a.z + a.w) * mThreshold * mThreshold * ( 1 - mThreshold );
		}
	};
	class StochasticMaxPooler
	{
		float* mRand;
		size_t mPitch;
		float mThreshold;
	public:
		enum {LoopNum = 1};
		StochasticMaxPooler(float threshold):mThreshold(0.0f),mRand(0),mPitch(0)
		{
			mThreshold = threshold;
		}
		~StochasticMaxPooler(){mRand = 0;}
		__device__ __inline__ float4 base(float* r,size_t pitch)
		{
			mRand = r;
			mPitch = pitch;
			return make_float4(0,0,0,0);
		}
		template<int loop> __device__ __inline__ float4 compute(float4 a , float b);
		template<> __device__ __inline__ float4 compute<1>(float4 a , float b)
		{
			if(b > a.x && mRand[0] >= mThreshold)
				a.x = b;
			mRand += mPitch;
			return a;
		}
		__device__ __inline__ float result(float4 a)
		{
			return a.x;
		}
	};
	class AvgPooler
	{
	public:
		enum{ LoopNum = 1};
		__device__ __inline__ float4 base(float* r,size_t pitch)
		{
			return make_float4(0.0f,0.0f,0.0f,0.0f);
		}
		template<int loop> __device__ __inline__ float4 compute(float4 a , float b);
		template<> __device__ __inline__ float4 compute<1>(float4 a , float b)
		{
			a.x += b;
			a.y += 1;
			return a;
		}
		__device__ __inline__ float result(float4 a)
		{
			if(a.y < 1e-5) return 0.0f;
			return a.x / a.y;
		}
	};
	class StochasticAvgPooler
	{
		float* mRand;
		size_t mPitch;
		float mThreshold;
	public:
		enum {LoopNum = 1};
		StochasticAvgPooler(float threshold):mThreshold(0.0f),mRand(0),mPitch(0)
		{
			mThreshold = threshold;
		}
		~StochasticAvgPooler(){mRand = 0;}
		__device__ __inline__ float4 base(float* r,size_t pitch)
		{
			mRand = r;
			mPitch = pitch;
			return make_float4(0.0f,0.0f,0.0f,0.0f);
		}
		template<int loop> __device__ __inline__ float4 compute(float4 a , float b);
		template<> __device__ __inline__ float4 compute<1>(float4 a , float b)
		{
			if(mRand[0] >= mThreshold)
			{
				a.x += b;
			}
			a.y += 1;
			mRand += mPitch;
			return a;
		}
		__device__ __inline__ float result(float4 a)
		{
			if(a.y > 0)
				return a.x / a.y;
			else
				return 0;
		}
	};
	class StochasticAvgPooler1
	{
		float* mRand;
		size_t mPitch;
		float mThreshold;
	public:
		enum {LoopNum = 2};
		StochasticAvgPooler1(float threshold):mThreshold(0.0f),mRand(0),mPitch(0)
		{
			mThreshold = threshold;
		}
		~StochasticAvgPooler1(){mRand = 0;}
		__device__ __inline__ float4 base(float* r,size_t pitch)
		{
			mRand = r;
			mPitch = pitch;
			return make_float4(0.0f,0.0f,0.0f,0.0f);
		}
		template<int loop> __device__ __inline__ float4 compute(float4 a , float b);
		template<> __device__ __inline__ float4 compute<2>(float4 a , float b)
		{
			if(mRand[mPitch * int(a.z)] >= mThreshold)
			{
				a.x += b;
				a.y += 1;
			}
			a.z += 1;
			return a;
		}
		template<> __device__ __inline__ float4 compute<1>(float4 a , float b)
		{
			if(mRand[mPitch * int(a.w)] >= mThreshold)
				mRand[mPitch * int(a.w)] = a.y;
			else
				mRand[mPitch * int(a.w)] = 0;
			a.w += 1;
			return a;
		}
		__device__ __inline__ float result(float4 a)
		{
			if(a.y > 0.0)
				return a.x / a.y;
			return 0;
		}
	};
	class StochasticPooler
	{
		float* mRand;
		size_t mPitch;
		float prevSum;
	public:
		enum {LoopNum = 2};
		StochasticPooler():mRand(0),mPitch(0),prevSum(0){}
		~StochasticPooler(){mRand = 0;}
		__device__ __inline__ float4 base(float* r,size_t pitch)
		{
			mRand = r;
			mPitch = pitch;
			return make_float4(0.0f,0.0f,0.0f,0.0f);
		}
		template<int loop> __device__ __inline__ float4 compute(float4 a , float b);
		template<> __device__ __inline__ float4 compute<2>(float4 a , float b)
		{
			a.y += b;
			return a;
		}
		template<> __device__ __inline__ float4 compute<1>(float4 a , float b)
		{
			if(a.y < 1e-5) return a;

			float m = b / a.y;
			if(prevSum <= mRand[0] && mRand[0] < prevSum + m)
				a.x = b;
			prevSum += m;
			return a;
		}
		__device__ __inline__ float result(float4 a)
		{
			return a.x;
		}
	};
	class ProbWeightPooler
	{
	public:
		enum{ LoopNum = 1};
		__device__ __inline__ float4 base(float* r, size_t pitch)
		{
			return make_float4(0.0f,0.0f,0.0f,0.0f);
		}
		template<int loop> __device__ __inline__ float4 compute(float4 a , float b);
		template<> __device__ __inline__ float4 compute<1>(float4 a , float b)
		{
			a.x += b * b;
			a.y += b;
			return a;
		}
		__device__ __inline__ float result(float4 a)
		{
			if (a.y >1e-5)
				return a.x / a.y;
			return 0;
		}
	};
}
/*
Loop function used in the 3D pooling global function
*/
namespace NLC
{
	template<int LoopNums, class Agg> __device__ __inline__ void Loop(float4& tmp, float* imgs, const int imgSize,
		const int numImages,const int loopStartF, const int loopEndF, const int loopStartY, 
		const int loopEndY, const int loopStartX, const int loopEndX, Agg& agg)
	{
		for(int f = loopStartF; f< loopEndF; f++)
			for(int y = loopStartY; y < loopEndY; y++)
				for(int x = loopStartX; x < loopEndX; x++)
				{
					const int imgPx = f * imgSize * imgSize + y * imgSize + x;
					tmp = agg.compute<LoopNums>(tmp, imgs[imgPx * numImages]);
				}
		Loop<LoopNums - 1, Agg>(tmp, imgs,imgSize,numImages,loopStartF,loopEndF,loopStartY,loopEndY,loopStartX,loopEndX,agg);
	};
	template<> __device__ __inline__ void Loop<0,StochasticPooler>(float4& tmp, float* imgs, const int imgSize,
		const int numImages,const int loopStartF, const int loopEndF, const int loopStartY, 
		const int loopEndY, const int loopStartX, const int loopEndX, StochasticPooler& agg)
	{
		return;
	};
	template<> __device__ __inline__ void Loop<0,ProbWeightPooler>(float4& tmp, float* imgs, const int imgSize,
		const int numImages,const int loopStartF, const int loopEndF, const int loopStartY, 
		const int loopEndY, const int loopStartX, const int loopEndX, ProbWeightPooler& agg)
	{
		return;
	};
	template<> __device__ __inline__ void Loop<0,StochasticMaxPooler>(float4& tmp, float* imgs, const int imgSize,
		const int numImages,const int loopStartF, const int loopEndF, const int loopStartY, 
		const int loopEndY, const int loopStartX, const int loopEndX, StochasticMaxPooler& agg)
	{
		return;
	};
	/*template<> __device__ __inline__ void Looper::Loop<0,StochasticMaxPooler3D>(float4& tmp, float* imgs, const int imgSize,
		const int numImages,const int loopStartF, const int loopEndF, const int loopStartY, 
		const int loopEndY, const int loopStartX, const int loopEndX, StochasticMaxPooler3D& agg)
	{
		return;
	};*/
	
	template<> __device__ __inline__ void Loop<0,MaxPooler>(float4& tmp, float* imgs, const int imgSize,
		const int numImages,const int loopStartF, const int loopEndF, const int loopStartY, 
		const int loopEndY, const int loopStartX, const int loopEndX, MaxPooler& agg)
	{
		return;
	};
	template<> __device__ __inline__ void Loop<0,AvgPooler>(float4& tmp, float* imgs, const int imgSize,
		const int numImages,const int loopStartF, const int loopEndF, const int loopStartY, 
		const int loopEndY, const int loopStartX, const int loopEndX, AvgPooler& agg)
	{
		return;
	};
	template<> __device__ __inline__ void Loop<0,StochasticAvgPooler>(float4& tmp, float* imgs, const int imgSize,
		const int numImages,const int loopStartF, const int loopEndF, const int loopStartY, 
		const int loopEndY, const int loopStartX, const int loopEndX, StochasticAvgPooler& agg)
	{
		return;
	};
	template<> __device__ __inline__ void Loop<0,StochasticAvgPooler1>(float4& tmp, float* imgs, const int imgSize,
		const int numImages,const int loopStartF, const int loopEndF, const int loopStartY, 
		const int loopEndY, const int loopStartX, const int loopEndX, StochasticAvgPooler1& agg)
	{
		return;
	};
	template<> __device__ __inline__ void Loop<0,ProbMaxPooler>(float4& tmp, float* imgs, const int imgSize,
		const int numImages,const int loopStartF, const int loopEndF, const int loopStartY, 
		const int loopEndY, const int loopStartX, const int loopEndX, ProbMaxPooler& agg)
	{
		return;
	};
	template<> __device__ __inline__ void Loop<0,ProbMaxPooler9>(float4& tmp, float* imgs, const int imgSize,
		const int numImages,const int loopStartF, const int loopEndF, const int loopStartY, 
		const int loopEndY, const int loopStartX, const int loopEndX, ProbMaxPooler9& agg)
	{
		return;
	};
}

/*
Special Loop function and 3D pooling device function for StochasticMaxPooler3D
*/
namespace NLC
{
	class StochasticMaxPooler3D
	{
	public:
		float* mRand;
		size_t mPitch;
		float mThreshold;
		enum {LoopNum = 1};
		StochasticMaxPooler3D(float threshold):mThreshold(0.0f),mRand(0),mPitch(0)
		{
			mThreshold = threshold;
		}
		~StochasticMaxPooler3D(){mRand = 0;}
		__device__ __inline__ float4 base(float* r,size_t pitch)
		{
			mRand = r;
			mPitch = pitch;
			return make_float4(0,0,0,0);
		}
		__device__ __inline__ float result(float4 a)
		{
			return a.x;
		}
	};
	class ProbMaxPooler3D
	{
	public:
		float mThreshold;
		enum {LoopNum = 1};
		ProbMaxPooler3D(float threshold):mThreshold(0.0f)
		{
			mThreshold = 1 - threshold;
		}
		__device__ __inline__ float4 base(float* r,size_t pitch)
		{
			return make_float4(0,0,0,0);
		}
		__device__ __inline__ float result(float4 a)
		{
			return a.x * mThreshold + a.y * mThreshold * mThreshold + (a.z + a.w) * mThreshold * mThreshold * mThreshold; 
		}
	};
	template<> __device__ __inline__ void Loop<1,ProbMaxPooler3D>(float4& tmp, float* imgs, const int imgSize,
		const int numImages,const int loopStartF, const int loopEndF, const int loopStartY, 
		const int loopEndY, const int loopStartX, const int loopEndX, ProbMaxPooler3D& agg)
		{
			for(int y = loopStartY; y < loopEndY; y++)
				for(int x = loopStartX; x < loopEndX; x++)
				{
					float cross_map_max = 0;
					for(int f =loopStartF; f < loopEndF; f++)
					{
						const int imgPx = y * imgSize + x + imgSize * imgSize * f;
						float p = imgs[imgPx * numImages];
						cross_map_max += p;
					}
					if(cross_map_max > 1e-5)
						cross_map_max = cross_map_max / (loopEndF - loopStartF);
					if(cross_map_max > tmp.x)
					{
						tmp.x = cross_map_max;
					}
				}
			for(int y = loopStartY; y < loopEndY; y++)
				for(int x = loopStartX; x < loopEndX; x++)
				{
					float cross_map_max = 0;
					for(int f =loopStartF; f < loopEndF; f++)
					{
						const int imgPx = y * imgSize + x + imgSize * imgSize * f;
						float p = imgs[imgPx * numImages];
						cross_map_max += p;
					}
					if(cross_map_max > 1e-5)
						cross_map_max = cross_map_max / (loopEndF - loopStartF);
					if(cross_map_max > tmp.y && cross_map_max < tmp.x)
					{
						tmp.y = cross_map_max;
					}
				}
			for(int y = loopStartY; y < loopEndY; y++)
				for(int x = loopStartX; x < loopEndX; x++)
				{
					float cross_map_max = 0;
					for(int f =loopStartF; f < loopEndF; f++)
					{
						const int imgPx = y * imgSize + x + imgSize * imgSize * f;
						float p = imgs[imgPx * numImages];
						cross_map_max += p;
					}
					if(cross_map_max > 1e-5)
						cross_map_max = cross_map_max / (loopEndF - loopStartF);
					if(cross_map_max > tmp.z && cross_map_max < tmp.y)
					{
						tmp.z = cross_map_max;
					}
				}
			for(int y = loopStartY; y < loopEndY; y++)
				for(int x = loopStartX; x < loopEndX; x++)
				{
					float cross_map_max = 0;
					for(int f =loopStartF; f < loopEndF; f++)
					{
						const int imgPx = y * imgSize + x + imgSize * imgSize * f;
						float p = imgs[imgPx * numImages];
						cross_map_max += p;
					}
					if(cross_map_max > 1e-5)
						cross_map_max = cross_map_max / (loopEndF - loopStartF);
					if(cross_map_max > tmp.w && cross_map_max < tmp.z)
					{
						tmp.w = cross_map_max;
					}
				}
		};
	template<> __device__ __inline__ void Loop<1,StochasticMaxPooler3D>(float4& tmp, float* imgs, const int imgSize,
		const int numImages,const int loopStartF, const int loopEndF, const int loopStartY, 
		const int loopEndY, const int loopStartX, const int loopEndX, StochasticMaxPooler3D& agg)
		{
			for(int y = loopStartY; y < loopEndY; y++)
				for(int x = loopStartX; x < loopEndX; x++)
				{
					float cross_map_max = 0;
					float cross_map_threshold = 0;
					for(int f =loopStartF; f < loopEndF; f++)
					{
						const int imgPx = y * imgSize + x + imgSize * imgSize * f;
						float p = imgs[imgPx * numImages];
						float rate = agg.mRand[0];
						if(rate > cross_map_threshold)
						{
							cross_map_max = p;
							cross_map_threshold = rate;
						}
						agg.mRand += agg.mPitch;
					}
					if(cross_map_threshold > agg.mThreshold && cross_map_max > tmp.x)
					{
						tmp.x = cross_map_max;
					}
				}
		};
}
/*
conv3DLocalPool and kLocal3DPool 
*/
namespace NLC
{
	template<class Agg>
	__global__ void kLocal3DPool(float* imgs, float* target, const int imgSize, const int numFilters,
							   const int numImages, const int subsX, const int startX, const int strideX,
							   const int outputsX, const int subsF, const int startF, const int strideF,
							   const int outNumFilters, Agg agg, int startImgIdx, float* uniforms) {
	
		const int imgIdx = threadIdx.x + startImgIdx * blockDim.x;
		if(imgIdx >= numImages)
			return;
		const int numPixels = blockDim.y * ( blockIdx.y * outputsX + blockIdx.x) + threadIdx.y;
		const int outPixelIdx = numPixels % (outputsX * outputsX);

		const int outFilterIdx = numPixels / (outputsX * outputsX);
		const int outX = outPixelIdx % outputsX;
		const int outY = outPixelIdx / outputsX;
		const int startImgPxX = outX * strideX + startX;
		const int startImgPxY = outY * strideX + startX;
		const int startFilterIdx = outFilterIdx * strideF + startF;

		if(outFilterIdx >= outNumFilters)
			return;

		//const int idx = threadIdx.x + blockIdx.x * blockDim.x + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
		imgs += imgIdx;
		target += imgIdx + (outFilterIdx * (outputsX * outputsX) + outPixelIdx) * numImages;
		uniforms += imgIdx + (outFilterIdx * (outputsX * outputsX) + outPixelIdx) * numImages;

		const int loopStartY = MAX(0, startImgPxY);
		const int loopStartX = MAX(0, startImgPxX);
		const int loopStartF = MAX(0, startFilterIdx);
		const int loopEndY = MIN(imgSize, startImgPxY + subsX);
		const int loopEndX = MIN(imgSize, startImgPxX + subsX);
		const int loopEndF = MIN(numFilters, startFilterIdx + subsF);

		// init with random number
		const int batch = outputsX * outputsX * outNumFilters * numImages;
		float4 tmp = agg.base(uniforms,batch);
		//NLC::Looper looper;
		Loop< Agg::LoopNum, Agg>(tmp, imgs,imgSize,numImages,loopStartF, loopEndF, loopStartY, loopEndY, loopStartX, loopEndX, agg);
		target[0] = agg.result(tmp);
	};

	template<class Pooler> void conv3DLocalPool( NVMatrix& images, NVMatrix& target,
		int numFilters, int subsF,int strideF, int outNumFilters, int subsX, int startX, 
		int strideX, int outputsX, Pooler pooler, NVMatrix& dropMask)
	{

		int numImages = images.getNumCols();					//batch
		int imgPixels = images.getNumRows() / numFilters;		//像素个数, numFilters对应的是channels
		assert(images.getNumRows() == numFilters * imgPixels); 
		int imgSize = int(sqrt(imgPixels));
		assert(imgSize * imgSize == imgPixels);
    
		assert(!images.isTrans());
		assert(!target.isTrans());
		assert(images.isContiguous());
    
		int outputs = outputsX * outputsX;

		target.resize(outNumFilters*outputs, numImages);
		target.apply(NVMatrixOps::Zero());

		dim3 threads(32,4);
		dim3 blocks(outputsX, outputsX * DIVUP(outNumFilters, threads.y));
		//NVMatrix dropMask(subsF * subsX * subsX, blocks.x * threads.x * blocks.y * threads.y);
		int numBlockImage = DIVUP(numImages, threads.x);
		cudaFuncSetCacheConfig(kLocal3DPool<Pooler>, cudaFuncCachePreferL1);
		for( int imgIdx = 0; imgIdx < numBlockImage; imgIdx++)
		{
			kLocal3DPool<Pooler><<<blocks, threads>>>(images.getDevData(), target.getDevData(),
			imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, subsF, 0, strideF,
			outNumFilters, pooler, imgIdx, dropMask.getDevData());
		}
	};
}
/*
 convLocalMaxUndo and convLocalAvgUndo
 */
namespace NLC{
void conv3DLocalMaxUndo(NVMatrix& images, NVMatrix& maxGrads, NVMatrix& maxActs, NVMatrix& target,
		int numFilters, int subsF,int strideF, int outNumFilters, int subsX, int startX, int strideX,
		int outputsX, float scaleTargets, float scaleOutput);

void conv3DLocalAvgUndo(NVMatrix& avgGrads, NVMatrix& target,
		int numFilters, int subsF,int strideF, int outNumFilters, int subsX, int startX, int strideX,
		int outputsX, int imageSize, float scaleTargets, float scaleOutput, float scale, NVMatrix& dropMask);

void conv3DLocalAvgUndo1(NVMatrix& avgGrads, NVMatrix& target,
		int numFilters, int subsF,int strideF, int outNumFilters, int subsX, int startX, int strideX,
		int outputsX, int imageSize, float scaleTargets, float scaleOutput, float scale, NVMatrix& dropMask);
}

#endif