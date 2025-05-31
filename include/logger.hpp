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

std::vector<Eigen::VectorXd> readVectorsFromCsv(const std::string &filename)
{
    std::vector<Eigen::VectorXd> vectors;
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file for reading");
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> values;

        while (std::getline(ss, cell, ','))
        {
            try
            {
                values.push_back(std::stod(cell));
            }
            catch (const std::invalid_argument &e)
            {
                std::cerr << "Invalid number in CSV: " << cell << std::endl;
                throw;
            }
        }

        if (!values.empty())
        {
            Eigen::VectorXd vec(values.size());
            for (size_t i = 0; i < values.size(); ++i)
            {
                vec(i) = values[i];
            }
            vectors.push_back(vec);
        }
    }

    file.close();
    std::cout << "Data read from " << filename << std::endl;
    return vectors;
}