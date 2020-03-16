#ifndef ITKLOADVOLUME_H // INCLUDE GUARD
#define ITKLOADVOLUME_H // INCLUDE GUARD

#include <iostream>

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"


/* This function returns the spacing and size. For 3D images */
void itkLoadVolumeHeader3D(const char inputFilename[], float* spacing, int* size){
	// Read image
	typedef  itk::Image<float, 3> float3DImage;
	typedef  itk::ImageFileReader< float3DImage  > ReaderType;
	ReaderType::Pointer reader = ReaderType::New();

	reader->SetFileName( inputFilename );

	try{
		reader->Update();
	}
	catch( itk::ExceptionObject & err ){
		std::cerr << "ExceptionObject caught !" << std::endl;
		std::cerr << err << std::endl;
	}

	float3DImage::Pointer image = reader->GetOutput();

	// Read spacing
	const float3DImage::SpacingType& sp = image->GetSpacing();
	spacing[0] = sp[0];
	spacing[1] = sp[1];
	spacing[2] = sp[2];

	// Get dimension
	float3DImage::RegionType region = image->GetLargestPossibleRegion();
	float3DImage::SizeType sz = region.GetSize();
	size[0] = sz[0];
	size[1] = sz[1];
	size[2] = sz[2];
}

// This function is for 2D image
void itkLoadVolumeHeader2D(const char inputFilename[], float* spacing, int* size){
	// Read image
	typedef  itk::Image<float, 2> float2DImage;
	typedef  itk::ImageFileReader< float2DImage  > ReaderType;
	ReaderType::Pointer reader = ReaderType::New();

	reader->SetFileName( inputFilename );

	try{
		reader->Update();
	}
	catch( itk::ExceptionObject & err ){
		std::cerr << "ExceptionObject caught !" << std::endl;
		std::cerr << err << std::endl;
	}

	float2DImage::Pointer image = reader->GetOutput();

	// Read spacing
	const float2DImage::SpacingType& sp = image->GetSpacing();
	spacing[0] = sp[0];
	spacing[1] = sp[1];

	// Get dimension
	float2DImage::RegionType region = image->GetLargestPossibleRegion();
	float2DImage::SizeType sz = region.GetSize();
	size[0] = sz[0];
	size[1] = sz[1];
}


/* This function loads a 3D volume into a linear array. */
void itkLoadVolume3D(const char* inputFilename, float* imageArr){
	// Read image
	typedef  itk::Image<float, 3> float3DImage;
	typedef  itk::ImageFileReader< float3DImage  > ReaderType;
	ReaderType::Pointer reader = ReaderType::New();

	reader->SetFileName( inputFilename );

	try{
		reader->Update();
	}
	catch( itk::ExceptionObject & err ){
		std::cerr << "ExceptionObject caught !" << std::endl;
		std::cerr << err << std::endl;
	}

	float3DImage::Pointer image = reader->GetOutput();

	// Get dimension
	float3DImage::RegionType region = image->GetLargestPossibleRegion();
	float3DImage::SizeType sz = region.GetSize();

	int Csz = sz[0];
	int Rsz = sz[1];
	int Dsz = sz[2];

	// Loop through image with index
	typedef itk::ImageRegionIteratorWithIndex< float3DImage > IteratorType;
	IteratorType copyIter( image, image->GetLargestPossibleRegion() );


	int C, R, D;
	for ( copyIter.GoToBegin(); !copyIter.IsAtEnd(); ++copyIter){
		float3DImage::IndexType idx = copyIter.GetIndex();
		int c = idx[0];
		int r = idx[1];
		int d = idx[2];
		// Assign values
		imageArr[c + r*Csz + d*Csz*Rsz] = (float)copyIter.Get();

		C=c;
		R=r;
		D=d;
	}

}

/* This function loads a 3D volume into a linear array. */
void itkLoadVolume2D(const char* inputFilename, float* imageArr){
	// Read image
	typedef  itk::Image<float, 2> float2DImage;
	typedef  itk::ImageFileReader< float2DImage  > ReaderType;
	ReaderType::Pointer reader = ReaderType::New();

	reader->SetFileName( inputFilename );

	try{
		reader->Update();
	}
	catch( itk::ExceptionObject & err ){
		std::cerr << "ExceptionObject caught !" << std::endl;
		std::cerr << err << std::endl;
	}

	float2DImage::Pointer image = reader->GetOutput();

	// Get dimension
	float2DImage::RegionType region = image->GetLargestPossibleRegion();
	float2DImage::SizeType sz = region.GetSize();

	int Csz = sz[0];
	int Rsz = sz[1];
	int Dsz = sz[2];

	// Loop through image with index
	typedef itk::ImageRegionIteratorWithIndex< float2DImage > IteratorType;
	IteratorType copyIter( image, image->GetLargestPossibleRegion() );


	int C, R;
	for ( copyIter.GoToBegin(); !copyIter.IsAtEnd(); ++copyIter){
		float2DImage::IndexType idx = copyIter.GetIndex();
		int c = idx[0];
		int r = idx[1];
		// Assign values
		imageArr[c + r*Csz ] = (float)copyIter.Get();

		C=c;
		R=r;
	}

}

// This functions write a 3D volume from a linear array to a nii image
void itkWrtieVolume3D(const char outputFilename[], float* imageArr, float* sp, int* sz){
	// Initialize the image
	// Set start index and size
	typedef  itk::Image<float, 3> float3DImage;
	float3DImage::RegionType region;
	float3DImage::IndexType start;
	start[0] = 0;
	start[1] = 0;
	start[2] = 0;

	float3DImage::SizeType size;
	size[0] = sz[0];
	size[1] = sz[1];
	size[2] = sz[2];
	region.SetSize(size);
	region.SetIndex(start);

	float3DImage::Pointer image = float3DImage::New();
	image->SetRegions(region);
	image->Allocate(); // Allocate memory

	// Set spacing
	float3DImage::SpacingType spacing;
	spacing[0] = sp[0];
	spacing[1] = sp[1];
	spacing[2] = sp[2];
	image->SetSpacing(spacing);

	// Loop through image with index
	int Csz = sz[0];
	int Rsz = sz[1];
	int Dsz = sz[2];
	typedef itk::ImageRegionIteratorWithIndex< float3DImage > IteratorType;
	IteratorType copyIter( image, image->GetLargestPossibleRegion() );

	int C, R, D;
	for ( copyIter.GoToBegin(); !copyIter.IsAtEnd(); ++copyIter){
		float3DImage::IndexType idx = copyIter.GetIndex();
		int c = idx[0];
		int r = idx[1];
		int d = idx[2];
		// Assign values
		copyIter.Set(imageArr[c + r*Csz + d*Csz*Rsz]);
		C=c;
		R=r;
		D=d;
	}

	typedef  itk::ImageFileWriter<float3DImage> WriterType;
	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(outputFilename);
	writer->SetInput(image);
	writer->Update();
}

// This functions write a 2D volume from a linear array to a nii image
void itkWrtieVolume2D(const char outputFilename[], float* imageArr, float* sp, int* sz){
	// Initialize the image
	// Set start index and size
	typedef  itk::Image<float, 2> float2DImage;
	float2DImage::RegionType region;
	float2DImage::IndexType start;
	start[0] = 0;
	start[1] = 0;

	float2DImage::SizeType size;
	size[0] = sz[0];
	size[1] = sz[1];
	region.SetSize(size);
	region.SetIndex(start);

	float2DImage::Pointer image = float2DImage::New();
	image->SetRegions(region);
	image->Allocate(); // Allocate memory

	// Set spacing
	float2DImage::SpacingType spacing;
	spacing[0] = sp[0];
	spacing[1] = sp[1];
	image->SetSpacing(spacing);

	// Loop through image with index
	int Csz = sz[0];
	int Rsz = sz[1];
	typedef itk::ImageRegionIteratorWithIndex< float2DImage > IteratorType;
	IteratorType copyIter( image, image->GetLargestPossibleRegion() );

	int C, R;
	for ( copyIter.GoToBegin(); !copyIter.IsAtEnd(); ++copyIter){
		float2DImage::IndexType idx = copyIter.GetIndex();
		int c = idx[0];
		int r = idx[1];
		// Assign values
		copyIter.Set(imageArr[c + r*Csz]);
		C=c;
		R=r;
	}

	typedef  itk::ImageFileWriter<float2DImage> WriterType;
	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(outputFilename);
	writer->SetInput(image);
	writer->Update();
}



#endif // INCLUDE GUARD
