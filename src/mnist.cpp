#include "mnist.h"

constexpr uint32_t imageMagic = 2051;
constexpr uint32_t labelMagic = 2049;

TrainingDataSet get_MNIST(const char *images, const char *labels)
{
  std::ifstream imagesFile(images, std::ios_base::binary);
  std::ifstream labelsFile(labels, std::ios_base::binary);

  // check magic numbers
  uint32_t magic;
  imagesFile.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  magic = __builtin_bswap32(magic);
  if (magic != imageMagic)
  {
    std::cout << magic << std::endl;
    std::cout << imageMagic << std::endl;
    throw std::invalid_argument("Invalid image file");
  }

  labelsFile.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  magic = __builtin_bswap32(magic);
  if (magic != labelMagic)
  {
    throw std::invalid_argument("Invalid label file");
  }

  // check number of images and labels
  uint32_t numImages;
  uint32_t numLabels;
  imagesFile.read(reinterpret_cast<char *>(&numImages), sizeof(numImages));
  labelsFile.read(reinterpret_cast<char *>(&numLabels), sizeof(numLabels));
  numImages = __builtin_bswap32(numImages);
  numLabels = __builtin_bswap32(numLabels);

  if (numImages != numLabels)
  {
    throw std::invalid_argument("Number of images and labels do not match");
  }

  // get image size

  uint32_t imgRows, imgCols;
  imagesFile.read(reinterpret_cast<char *>(&imgRows), sizeof(imgRows));
  imagesFile.read(reinterpret_cast<char *>(&imgCols), sizeof(imgCols));
  imgRows = __builtin_bswap32(imgRows);
  imgCols = __builtin_bswap32(imgCols);

  uint32_t imgSize = imgRows * imgCols;

  // start importing
  TrainingDataSet dataSet;
  dataSet.numData = numImages;
  dataSet.data = new TrainingData[numImages];

  Vector img = Vector::create(imgSize);
  Vector label = Vector::create(10);
  uint8_t *imgCache = new uint8_t[imgSize];
  uint8_t labelCache;
  for (uint32_t i = 0; i < numImages; i++)
  {
    imagesFile.read(reinterpret_cast<char *>(imgCache), sizeof(uint8_t) * imgSize);
    labelsFile.read(reinterpret_cast<char *>(&labelCache), sizeof(uint8_t));
    img.reset();
    label.reset();
    for (uint32_t j = 0; j < imgSize; j++)
    {
      img.vec.get()[j] = imgCache[j] / 255.0;
    }
    label.vec.get()[labelCache] = 1.0;

    dataSet.data[i].input = Vector::copy(img);
    dataSet.data[i].output = Vector::copy(label);
  }

  delete[] imgCache;
  return dataSet;
}