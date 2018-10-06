#pragma

#include <Windows.h>

class Timer
{
public:
	Timer()
	{
		QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(&m_freq));		
	}

	inline void tic(void)
	{
		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&ts));
	}

	inline double toc(void)
	{
		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&te));		
		return (te - ts) / (double)m_freq;
	}	

private:
		
	__int64 m_freq;
	__int64 ts, te;
};