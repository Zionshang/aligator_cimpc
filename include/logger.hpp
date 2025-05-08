#pragma once
#include <fstream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <iostream>

void saveVectorsToCsv(const std::string &filename, const std::vector<Eigen::VectorXd> &vectors)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        throw "Unable to open file for writing";
    }

    for (const auto &vec : vectors)
    {
        for (int i = 0; i < vec.size(); ++i)
        {
            file << vec(i);
            if (i < vec.size() - 1)
                file << ",";
        }
        file << "\n";
    }
    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}

void saveVectorsToCsv(const std::string &filename, const std::vector<double> &vectors)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        throw "Unable to open file for writing";
    }

    for (const auto &vec : vectors)
    {
        file << vec;
        file << "\n";
    }
    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}
