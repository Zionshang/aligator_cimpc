#include <iostream>
#include <chrono>
#include <thread>
#include <string>
#include <limits>

class Timer
{
public:
    explicit Timer(const std::string &name)
        : name(name), count(0),
          average_duration(0.0),
          max_duration(std::numeric_limits<long long>::min()),
          min_duration(std::numeric_limits<long long>::max()) {}

    void start()
    {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void stop()
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        ++count;
        // 迭代法更新平均值
        average_duration += (duration - average_duration) / count;

        if (duration > max_duration)
        {
            max_duration = duration;
        }
        if (duration < min_duration)
        {
            min_duration = duration;
        }

        std::cout << "[" << name << "] 本次耗时: " << duration << " ms\n";
    }

    ~Timer()
    {
        if (count > 0)
        {
            std::cout << "[" << name << "] 平均耗时: " << average_duration << " ms\t"
                      << "最大耗时: " << max_duration << " ms\t"
                      << "最小耗时: " << min_duration << " ms\n";
        }
        else
        {
            std::cout << "[" << name << "] 未进行计时.\n";
        }
    }

private:
    std::string name;
    std::chrono::high_resolution_clock::time_point start_time;
    int count;
    double average_duration; // 用 double 存储浮点平均值
    long long max_duration;
    long long min_duration;
};