#include <iostream>
#include <chrono>
#include <thread>
#include <string>

class Timer
{
public:
    explicit Timer(const std::string &name) : name(name), total_duration(0), count(0) {}

    void start()
    {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void stop()
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        total_duration += duration;
        ++count;

        std::cout << "[" << name << "] 本次耗时: " << duration << " ms\n";
    }

    ~Timer()
    {
        if (count > 0)
        {
            std::cout << "[" << name << "] 平均耗时: " << (total_duration / count) << " ms\n";
        }
        else
        {
            std::cout << "[" << name << "] 未进行计时.\n";
        }
    }

private:
    std::string name;
    std::chrono::high_resolution_clock::time_point start_time;
    long long total_duration;
    int count;
};