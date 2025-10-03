#ifndef _TIME_UTILS_H_
#define _TIME_UTILS_H_

#include <stdio.h>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>

using time_point = std::chrono::system_clock::time_point;
using time_point_steady = std::chrono::steady_clock::time_point;

namespace time_utils {
inline std::string get_string_time(const time_point t, const std::string &format = "%Y-%m-%d %H:%M:%S", const bool mini_sec = true) {
    const std::time_t time = std::chrono::system_clock::to_time_t(t);
    const std::tm time_info = *std::localtime(&time);
    std::ostringstream oss;
    oss << std::put_time(&time_info, format.c_str());
    if (mini_sec) {
        oss << std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch()).count() % 1000;
    }
    return oss.str();
}

inline int64_t diff_time_chrono(time_point start_time, time_point end_time) {
    return std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
}

inline int64_t diff_time_seconds(time_point start_time, time_point end_time) {
    return std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
}

inline double diff_time_double(time_point start_time, time_point end_time) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
}

inline float diff_time_float(time_point start_time, time_point end_time) {
    return std::chrono::duration_cast<std::chrono::duration<float>>(end_time - start_time).count();
}

inline int64_t diff_time_milliseconds(time_point start_time, time_point end_time) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
}

inline time_point now() {
    return std::chrono::system_clock::now();
}

inline double get_current_time_double() {
    return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now().time_since_epoch()).count();
}

template <typename T>
inline T now() {
    return std::chrono::duration_cast<std::chrono::duration<T>>(std::chrono::system_clock::now().time_since_epoch()).count();
}

inline std::string get_current_time_string(std::string format = "%Y-%m-%d %H:%M:%S") {
    const time_point now = time_utils::now();
    return get_string_time(now, format);
}

inline time_point build_time(int year, int month, int day, int hour, int minute, int second) {
    std::tm time_info{};
    time_info.tm_year = year - 1900;
    time_info.tm_mon = month - 1;
    time_info.tm_mday = day;
    time_info.tm_hour = hour;
    time_info.tm_min = minute;
    time_info.tm_sec = second;
    std::time_t time = std::mktime(&time_info);
    return std::chrono::system_clock::from_time_t(time);
}

inline int64_t convert_to_seconds(const time_point& time) {
    return std::chrono::duration_cast<std::chrono::seconds>(time.time_since_epoch()).count();
}

inline time_point_steady get_current_time_steady() {
    return std::chrono::steady_clock::now();
}

inline int64_t convert_to_seconds(const time_point_steady& time) {
    return std::chrono::duration_cast<std::chrono::seconds>(time.time_since_epoch()).count();
}

}  // namespace time_utils

#endif