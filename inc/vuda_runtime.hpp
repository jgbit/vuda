#pragma once

#include "vuda.hpp"

/*
    Wrapper for vuda functions
*/

////////////////////////////////////////////////
// Types
////////////////////////////////////////////////

#define cudaSuccess vuda::vudaSuccess

#define cudaHostAllocDefault vuda::hostAllocDefault
#define cudaHostAllocPortable vuda::hostAllocPortable
#define cudaHostAllocMapped vuda::hostAllocMapped
#define cudaHostAllocWriteCombined vuda::hostAllocWriteCombined

#define cudaMemcpyHostToHost vuda::memcpyHostToHost
#define cudaMemcpyHostToDevice vuda::memcpyHostToDevice
#define cudaMemcpyDeviceToHost vuda::memcpyDeviceToHost
#define cudaMemcpyDeviceToDevice vuda::memcpyDeviceToDevice

typedef vuda::stream_t cudaStream_t;
typedef vuda::event_t cudaEvent_t;
typedef vuda::error_t cudaError_t;
typedef vuda::memcpyKind cudaMemcpyKind;
typedef vuda::deviceProp cudaDeviceProp;

////////////////////////////////////////////////
// Device Management
////////////////////////////////////////////////

//
// Returns which device is currently being used.
inline cudaError_t cudaGetDevice(int* device)
{
    return vuda::getDevice(device);
}

//
// Returns the number of compute-capable devices.
inline cudaError_t cudaGetDeviceCount(int* count)
{
    return vuda::getDeviceCount(count);
}

//
// Returns information about the compute-device.
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device)
{
    return vuda::getDeviceProperties(prop, device);
}


// Destroy all allocations and reset all state on the current device in the current process.
inline cudaError_t cudaDeviceReset(void)
{
    // TODO: no vuda reset
    return cudaSuccess;
}

/*//
// Sets flags to be used for device executions.
cudaError_t cudaSetDeviceFlags(unsigned int  flags)
{

}

//
// Gets the flags for the current device.
cudaError_t cudaGetDeviceFlags(unsigned int* flags)
{
    
}*/

// Set device to be used for GPU executions.
inline cudaError_t cudaSetDevice(int device)
{
    return vuda::setDevice(device);
}


////////////////////////////////////////////////
// Error Handling
////////////////////////////////////////////////

// Returns the description string for an error code.
inline const char* cudaGetErrorString(cudaError_t error)
{
    // TODO: no vuda error string
    return "vuda::getErrorString";
}

////////////////////////////////////////////////
// Stream Management
////////////////////////////////////////////////

/*
// Add a callback to a compute stream.
inline cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int  flags)
{

}

// Attach memory to a stream asynchronously.
inline cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length = 0, unsigned int  flags = cudaMemAttachSingle)
{

}

// Begins graph capture on a stream.
inline cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode)
{

}

// Create an asynchronous stream.
inline cudaError_t cudaStreamCreate(cudaStream_t* pStream)
{

}

// Create an asynchronous stream.
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int  flags)
{

}

// Create an asynchronous stream with the specified priority.
inline cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int  flags, int  priority)
{

}

// Destroys and cleans up an asynchronous stream.
inline cudaError_t cudaStreamDestroy(cudaStream_t stream)
{

}

// Ends capture on a stream, returning the captured graph.
inline cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph)
{

}

// Query capture status of a stream.
inline cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus ** pCaptureStatus, unsigned long long* pId)
{

}

// Query the flags of a stream.
inline cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags)
{

}

// Query the priority of a stream.
inline cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int* priority)
{

}

// Returns a stream's capture status. 
inline cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus ** pCaptureStatus)
{

}

// Queries an asynchronous stream for completion status.
inline cudaError_t cudaStreamQuery(cudaStream_t stream)
{

}

// Make a compute stream wait on an event.
inline cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int  flags)
{

}

// Swaps the stream capture interaction mode for a thread.
inline cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode ** mode)
{

}
*/

// Waits for stream tasks to complete.
inline cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    return vuda::streamSynchronize(stream);
}

////////////////////////////////////////////////
// Event Management
////////////////////////////////////////////////

// Creates an event object.
inline cudaError_t cudaEventCreate(cudaEvent_t* event)
{
    return vuda::eventCreate(event);
}

// Destroys an event object.
inline cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    return vuda::eventDestroy(event);
}

// Computes the elapsed time between events.
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end)
{
    return vuda::eventElapsedTime(ms, start, end);
}

/*// Queries an event's status.
inline cudaError_t cudaEventQuery(cudaEvent_t event)
{

}*/

// Records an event.
inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0)
{
    return vuda::eventRecord(event, stream);
}

// Waits for an event to complete.
inline cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
    return vuda::eventSynchronize(event);
}

////////////////////////////////////////////////
// Memory Management
////////////////////////////////////////////////

//
// Frees memory on the device.
inline cudaError_t cudaFree(void* devPtr)
{
    return vuda::free(devPtr);
}

//
// Frees page-locked memory.
inline cudaError_t cudaFreeHost(void* ptr)
{
    return vuda::freeHost(ptr);
}

//
// Allocate memory on the device.
inline cudaError_t cudaMalloc(void** devPtr, size_t size)
{
    return vuda::malloc(devPtr, size);
}

//
// Allocates page-locked memory on the host.
inline cudaError_t cudaMallocHost(void** ptr, size_t size)
{
    return vuda::mallocHost(ptr, size);
}

//
// Allocates page-locked memory on the host.
inline cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags)
{
    return vuda::hostAlloc(pHost, size, flags);
}

//
// Copies data between host and device.
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
{
    return vuda::memcpy(dst, src, count, kind);
}

//
// Copies data between host and device.
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0)
{
    return vuda::memcpy(dst, src, count, kind, stream);
}

//
// Initializes or sets device memory to a value.
/*inline cudaError_t cudaMemset(void* devPtr, int  value, size_t count)
{
    return vuda::memset
}*/