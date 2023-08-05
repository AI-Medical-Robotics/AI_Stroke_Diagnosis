#include <gtest/gtest.h>
#include <itkImage.h>
#include <itkImageRegionIterator.h>
// #include <itkForwardFFTImageFilter.h>

#include <itkImportImageFilter.h>

// Pixel type and dimension of image defined
using PixelType = unsigned char;
constexpr unsigned int Dimension = 3;

// Image type defined
using ImageType = itk::Image<PixelType, Dimension>;

TEST(ITKSmartPointersTest, ImageAllocationDeallocation) {
    // Define the image size and region
    ImageType::SizeType imageSize;
    imageSize.Fill(10);

    ImageType::RegionType imageRegion;
    imageRegion.SetSize(imageSize);

    // Create the image using a smart pointer (behaves like itk::SmartPointer)
    // supports both unique and shared ownership
    typename ImageType::Pointer image = ImageType::New();
    image->SetRegions(imageRegion);

    // Allocate memory for the image
    image->Allocate();

    // Fill image with a test value;
    itk::ImageRegionIterator<ImageType> it(image, imageRegion);
    it.GoToBegin();
    while(!it.IsAtEnd()) {
        it.Set(127);
        ++it;
    }

    // Verify allocated memory is not null
    EXPECT_TRUE(image->GetBufferPointer() != nullptr);

    // Manually Deallocate the memory for the image (not needed just for testing)
    image->Initialize();

    // Verify memory is deallocated
    EXPECT_TRUE(image->GetBufferPointer() == nullptr);
}


TEST(ITKGenHeaderTest, ImportImageFilter) {
    using ImportFilterType = itk::ImportImageFilter<PixelType, Dimension>;

    auto importFilter = ImportFilterType::New();
    EXPECT_TRUE(importFilter.IsNotNull());
}