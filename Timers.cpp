#include "Timers.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#include <sys/resource.h>

#endif

Timers::~Timers() {}


double Timers::second(void) {
#if defined(_WIN32)
    LARGE_INTEGER t;
    static double oofreq;
    static int checkedForHighResTimer;
    static BOOL hasHighResTimer;

    if (!checkedForHighResTimer) {
        hasHighResTimer = QueryPerformanceFrequency(&t);
        oofreq = 1.0 / (double)t.QuadPart;
        checkedForHighResTimer = 1;
    }
    if (hasHighResTimer) {
        QueryPerformanceCounter(&t);
        return (double)t.QuadPart * oofreq;
    }
    else {
        return (double)GetTickCount() / 1000.0;
    }
}
#elif defined (_linux)
    clock_t cputime = clock();
    double t = (double)(cputime / CLOCKS_PER_SEC);
    return t;
#elif defined(__linux__) || defined(__QNX__)
#include <stddef.h>
#include <sys/resource.h>
#include <sys/time.h>
double second(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#endif