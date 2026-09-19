// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

// ThreadSanitizer suppressions, compiled into the test binary so the suite needs
// no TSAN_OPTIONS to be clean.
//
// Arrow and Parquet come from conda and are not instrumented, so inside them the
// sanitizer sees only the calls it intercepts -- operator delete, pthread_mutex_lock,
// memcpy -- and none of the atomic reference counting that orders them. The
// parquet reader's pre-buffering runs on Arrow's IO thread pool, and the last
// reference to a read-range cache is released on one of those threads while the
// main thread still holds the reader's mutex; with the refcount invisible the
// two look unordered and the report is a data race in libarrow with no cpplink
// frame in it. `called_from_lib` ignores the interceptors those libraries call
// and nothing else: every access cpplink's own instrumented code makes is still
// checked.

// GCC announces the sanitizer with a macro, clang through __has_feature.
#if defined(__SANITIZE_THREAD__)
#define CPPLINK_THREAD_SANITIZER 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define CPPLINK_THREAD_SANITIZER 1
#endif
#endif

#ifdef CPPLINK_THREAD_SANITIZER

extern "C" const char* __tsan_default_suppressions() {
    return "called_from_lib:libarrow\n"
           "called_from_lib:libparquet\n";
}

#endif
