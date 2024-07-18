#include <iostream>
#include <fstream>
#include <random>
#include <cinttypes>

int main(int argc, char *argv[])
{
  uint32_t dim1 = std::stoi(argv[1]);
  uint32_t dim2 = std::stoi(argv[2]);
  std::ofstream file(argv[3], std::ios_base::binary);

  std::mt19937 mt{std::random_device{}()};
  std::uniform_real_distribution<double> dist(-100.0, 100.0);

  file.write(reinterpret_cast<char *>(&dim1), sizeof(dim1));
  file.write(reinterpret_cast<char *>(&dim2), sizeof(dim2));

  uint64_t size = dim1 * dim2;
  for (uint64_t i = 0; i < size; i++)
  {
    double val = dist(mt);
    file.write(reinterpret_cast<char *>(&val), sizeof(val));
  }
}