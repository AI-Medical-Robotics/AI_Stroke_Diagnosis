#include <gtest/gtest.h>
// #include "MRIDataLoader.h"

#include <vtkSmartPointer.h>
#include <vtkImageData.h>

// TEST(MRIDataLoaderTest1, VerifyPathsExist) {
//     MRIDataLoader data1;
//     data1.SetupPaths();

//     // Verify that the paths exist
//     EXPECT_TRUE(std::filesystem::is_directory(data1.icpsr_data_dir));
//     EXPECT_TRUE(std::filesystem::is_regular_file(data1.raw_dwi_path));
//     EXPECT_TRUE(std::filesystem::is_regular_file(data1.raw_t2w_anat_path));
// }

// TEST(MRIDataLoaderTest2, VerifyPathsExist) {
//     MRIDataLoader data;
//     data.SetupPaths();

//     // Load the raw DWI image
//     vtkSmartPointer<vtkImageData> image = data.LoadRawDWIImage();

//     // Verify that the image is not null
//     EXPECT_TRUE(image != nullptr);

//     // Verify that the image dimensions are greater than 0
//     int* dimensions = image->GetDimensions();
//     EXPECT_GT(dimensions[0], 0);
//     EXPECT_GT(dimensions[1], 0);
//     EXPECT_GT(dimensions[2], 0);
// }

class VTKSmartPointerTest : public ::testing::Test {
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
TEST_F(VTKSmartPointerTest, SharedReferenceCounting) {
    vtkSmartPointer<vtkImageData> image2 = image;
    EXPECT_EQ(image->GetReferenceCount(), 2);
}

