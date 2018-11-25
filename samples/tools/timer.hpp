#pragma once

#include <chrono>

class Timer
{
public:
	using time_point = std::chrono::high_resolution_clock::time_point;

	inline void tic(void)
	{
		ts = std::chrono::high_resolution_clock::now();
	}

	inline double toc(void) const
	{
		auto te = std::chrono::high_resolution_clock::now();
		return std::chrono::duration_cast<std::chrono::duration<double>>(te - ts).count();
	}	

private:
	time_point ts;
};