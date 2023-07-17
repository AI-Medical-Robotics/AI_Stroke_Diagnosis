#include <gtest/gtest.h>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>

class VTKImageTest : public ::testing::Test {
protected:
    vtkSmartPointer<vtkImageData> image;

    void SetUp() override {
        image = vtkSmartPointer<vtkImageData>::New();
        image->SetDimensions(100, 100, 1);
        image->AllocateScalars(VTK_UNSIGNED_CHAR, 1);
    }

    void TearDown() override {
        if (image != nullptr) {
            image->ReleaseData();
        }

        image = nullptr;
    }
};

// Test case to visualize a 2D randomly generated VTK image
TEST_F(VTKImageTest, ImageAllocDealloc) {
    if (image->GetScalarPointer()) {
        std::cout << "vtkImageData is allocated\n";
    }

    std::cout << "image number of points = " << image->GetNumberOfPoints() << std::endl;

    EXPECT_TRUE(image->GetScalarPointer() != nullptr);
    EXPECT_EQ(image->GetNumberOfPoints(), 10000);
}
