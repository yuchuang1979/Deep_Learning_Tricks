#ifndef CONV_PLUS_CUH
#define CONV_PLUS_CUH
#include <util.cuh>
#include <conv_util.cuh>
void transChannelById(NVMatrix& images, NVMatrix& target, int numChannels, int channelId, int num);
void shiftMap(NVMatrix& images, NVMatrix& target, int numChannels,
	int padding,NVMatrix& shiftMapX, NVMatrix& shiftMapY, float scaleTargets, bool ismirror);
void shiftMapBP(NVMatrix& images, NVMatrix& target, int numChannels,
	int padding,NVMatrix& shiftMapX, NVMatrix& shiftMapY, float scaleTargets, bool ismirror);

namespace NLC
{
	class NLCNVMatrixOps
	{
	public:
		class GetShiftFromUniform {
		private:
			int _shift;
			float _bank;
		public:
			GetShiftFromUniform(const float shift) : _shift(shift),_bank( 1.0f / (_shift * 2 + 1)) {
			};
			__device__ inline float operator()(const float a) const {
				return (int(a / _bank) - _shift);
			};
		};
	};
}
#endif