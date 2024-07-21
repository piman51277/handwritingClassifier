#include "mnist.h"

constexpr uint32_t imageMagic = 2051;
constexpr uint32_t labelMagic = 2049;

TrainingDataSet get_MNIST(const char *images, const char *labels, uint32_t batchSize)
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

  // compute number of batches
  uint32_t numBatches = numImages / batchSize;

  // start importing
  TrainingDataSet dataSet;
  dataSet.numData = numBatches;
  dataSet.data = new TrainingData[numBatches];
  double *imageT;
  double *labelT;
  cudaMallocHost(&imageT, imgSize * batchSize * sizeof(double));
  cudaMallocHost(&labelT, 10 * batchSize * sizeof(double));
  uint8_t *imgCache = new uint8_t[imgSize];
  uint8_t *labelCache = new uint8_t[1];

  uint32_t setPtr = 0;
  uint32_t ptr = 0;
  while (ptr < numImages)
  {
    // if there aren't enough images left to fill a batch, discard
    if (numImages - ptr < batchSize)
    {
      std::cout << "Discarding " << numImages - ptr << " images" << std::endl;
      break;
    }

    // read images and labels
    for (uint32_t i = 0; i < batchSize; i++)
    {
      imagesFile.read(reinterpret_cast<char *>(imgCache), imgSize);
      labelsFile.read(reinterpret_cast<char *>(labelCache), 1);

      for (uint32_t j = 0; j < imgSize; j++)
      {
        imageT[j * batchSize + i] = imgCache[j] / 255.0;
      }
      for(uint32_t j = 0; j < 10; j++)
      {
        labelT[j * batchSize + i] = 0.0;
      }
      labelT[i + batchSize * labelCache[0]] = 1.0;
    }

    Matrix input = Matrix::create(imgSize, batchSize);
    Matrix expected = Matrix::create(10, batchSize);

    cudaMemcpy(input.mat.get(), imageT, imgSize * batchSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(expected.mat.get(), labelT, 10 * batchSize * sizeof(double), cudaMemcpyHostToDevice);

    dataSet.data[setPtr].input = input;
    dataSet.data[setPtr].expected = expected;

    setPtr++;
    ptr += batchSize;
  }

  delete[] imgCache;
  delete[] labelCache;
  cudaFreeHost(imageT);
  cudaFreeHost(labelT);

  return dataSet;
}