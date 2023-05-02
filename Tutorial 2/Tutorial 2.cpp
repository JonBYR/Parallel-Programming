#include <iostream>
#include <vector>
#include <algorithm>
#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.pgm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char** argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm";
	bool colour = false; //this will be needed later if the image is a colour image
	for (int i = 1; i < argc; i++) { //code for command line execution
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned short> im_input(image_filename.c_str());
		CImg<unsigned char> converted_input; //needed to store the converted image
		CImg<unsigned char> image_input = (im_input /= (im_input.max() > 255 ? 257 : 1)); //https://github.com/GreycLab/CImg/issues/218 allows convertion between 16 and 8 bit images using code from this github
		//CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input, "input");
		//a 3x3 convolution mask implementing an averaging filter
		std::vector<float> convolution_mask = { 1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9 };
		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);
		cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 4 - device operations
		int bins;
		if (image_input.spectrum() == 3)
		{
			converted_input = image_input.get_RGBtoYCbCr(); //colour images need to be converted to YCbCr as RGB has no intensity channel
			image_input = image_input.get_channel(0); //the histogram works with intensities so we only need the y channel
			colour = true;
		}

		std::vector<unsigned int> possibleBins{ 8, 16, 32, 64, 128, 256 }; //potential bin sizes
		std::cout << "Please enter the number of bins you want, 8, 16, 32, 64, 128, 256: Anything else will be the upper bound for the bin entered. Values larger than 256 will default to 256. Any non number will default to an 8 binned output." << std::endl; //ask for bin sizes
		std::cin >> bins; //users inputs number of bins
		if (std::find(possibleBins.begin(), possibleBins.end(), bins) == possibleBins.end())
		{
			if (bins < possibleBins[0] || bins < 0) bins = possibleBins[0]; //if bins is less than 8 then the closest value is the first index
			else if (bins > possibleBins[possibleBins.size() - 1]) bins = possibleBins[possibleBins.size() - 1]; //if bins is greater than the last index element the closest value is the last index
			else //for all other values find the value in the array that is the next highest bin in the array 
			{
				auto const it = std::lower_bound(possibleBins.begin(), possibleBins.end(), bins);
				bins = *it; //https://stackoverflow.com/questions/8647635/elegant-way-to-find-closest-value-in-a-vector-from-above
			}
		}
		std::vector<unsigned int> histogram(bins); //make histogram with number of elements = number of bins
		std::vector<unsigned int> cumHistogram(bins); //do this for a cumulative and normalised histogram as they will be the same size
		std::vector<unsigned int> nomHistogram(bins);
		cl::Event histEvent;
		cl::Event cumEvent;
		cl::Event normEvent;
		cl::Event imageEvent; //create events to record information for kernel and memory transfer
		cl::Event memEvent; //event used for any memory processes
		float totalEvent = 0; //use this to record total kernel execution time
		float queueEvent = 0;
		float submitEvent = 0; //floats needed to store total queue and submission times
		size_t histoSize = bins * sizeof(unsigned int); //establish amount of memory needed for histograms
		size_t binSize = bins; //establish amount of memory needed for NDenqueue function
		//device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size()); //read only means the content of the buffer cannot be overwritten
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image
		cl::Buffer dev_histogram(context, CL_MEM_READ_WRITE, histoSize);
		cl::Buffer dev_cumulative(context, CL_MEM_READ_WRITE, histoSize); //write means that data in the buffer can be overwritten
		cl::Buffer dev_normalise(context, CL_MEM_READ_WRITE, histoSize); //histoSize is the memory that is assigned to the buffer
		cl::Buffer dev_bins(context, CL_MEM_READ_ONLY, sizeof(int));
		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0], NULL, &memEvent); //these functions are used for memory transfer
		queue.enqueueWriteBuffer(dev_bins, CL_TRUE, 0, sizeof(int), &bins, NULL, &memEvent); //this is done for read only buffers
		//4.2 Setup and execute the kernel (i.e. device code)
		vector<unsigned char> output_buffer(image_input.size());
		string histType;
		std::cout << "What histogram would you like to use. Type 1 for a global histogram. Type anything else for the local histogram" << std::endl;
		std::cin >> histType;
		if (histType == "1") //user input dictates the type of histogram
		{
			cl::Kernel kernel_histSimp = cl::Kernel(program, "hist_simple");
			kernel_histSimp.setArg(0, dev_image_input);
			kernel_histSimp.setArg(1, dev_histogram);
			kernel_histSimp.setArg(2, dev_bins);
			queue.enqueueNDRangeKernel(kernel_histSimp, cl::NullRange, image_input.size(), cl::NullRange, NULL, &histEvent); //global histogram isn't using work groups so nullrange is used
			queue.enqueueReadBuffer(dev_histogram, CL_TRUE, 0, histoSize, &histogram[0], NULL);
		}
		else
		{
			cl::Kernel kernel_hist = cl::Kernel(program, "hist_atomic"); //state the kernel we are using
			kernel_hist.setArg(0, dev_image_input); //arguments given are buffers
			kernel_hist.setArg(1, dev_histogram); //assign the arguments to the kernel
			kernel_hist.setArg(2, cl::Local(histoSize)); //size of local memory which is used for more efficent operations via work items
			kernel_hist.setArg(3, dev_bins);
			queue.enqueueNDRangeKernel(kernel_hist, cl::NullRange, image_input.size(), cl::NDRange(binSize), NULL, &histEvent); //kernel split into work groups that have a set bin size (which are our partial histograms)
			//image_input.size() acts as the key for the get_global_id()
			//cl::NDRange(binSize) acts as the key for get_local_id()
			queue.enqueueReadBuffer(dev_histogram, CL_TRUE, 0, histoSize, &histogram[0], NULL, &memEvent); //write the result to the histogram buffer dev_histogram (which in turns writes to the histogram)
			//memory transfer times would also extend to reading from the buffer as information is transferred from the buffer to the variable
		}
		std::cout << "Histogram Kernel Execution time in nanoseconds: " <<
			histEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			histEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl; //getProfilingInfo records data in nanoseconds for the kernel execution time
		std::cout << "Full histogram kernel information: " << GetFullProfilingInfo(histEvent, ProfilingResolution::PROF_US) << std::endl; //getFullProfilingInfo includes the queue and submission times of the kernel as well
		totalEvent += histEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - histEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>(); //the total program execution time will be the amount of time each kernel takes to execute which will be accumulated
		queueEvent += histEvent.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() - histEvent.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>(); //equation for queue times for each kernel which will be accumulated for the total queue time
		submitEvent += histEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() - histEvent.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>(); //equation for submission times for each kernel which will be accumulated for total submission time
		std::string answer;
		//std::cout << histogram << std::endl;
		std::cout << std::endl;
		std::cout << "Would you like blelloch or simple a local hillis steele or regular hillis: Type blelloch for blelloch, simple for simple or local for the local hillis. Any other answer will default to hillis steele" << std::endl;
		std::cin >> answer; //like histogram we get user input
		if (answer == "blelloch") //this code uses the exclusive blelloch scan
		{
			cl::Kernel kernel_copy = cl::Kernel(program, "histoCopy"); //assign to different kernel
			kernel_copy.setArg(0, dev_histogram);
			kernel_copy.setArg(1, dev_cumulative); //need to copy information to cumulatve histogram buffer first to preserve original histogram rather than overwritting it
			queue.enqueueNDRangeKernel(kernel_copy, cl::NullRange, cl::NDRange(binSize), cl::NullRange, NULL, &cumEvent); //copying the histogram should be an O(1) complexity so we can include it into our cumulative event as it won't have an effect on the blelloch scan's time
			cl::Kernel kernel_ble = cl::Kernel(program, "scan_bl");
			kernel_ble.setArg(0, dev_cumulative);
			queue.enqueueNDRangeKernel(kernel_ble, cl::NullRange, cl::NDRange(binSize), cl::NullRange, NULL, &cumEvent);
			queue.enqueueReadBuffer(dev_cumulative, CL_TRUE, 0, histoSize, &cumHistogram[0], NULL, &memEvent);
			//std::cout << cumHistogram << std::endl; //scan functions are used to make the cumulative histogram
		}
		else if (answer == "simple") //this code uses a simple scan which is just a for loop
		{
			cl::Kernel kernel_simple = cl::Kernel(program, "scan_simple"); //scan functions are used for the cumulative histogram
			kernel_simple.setArg(0, dev_histogram); //we first take the result from the hist_atomic kernel
			kernel_simple.setArg(1, dev_cumulative);
			queue.enqueueNDRangeKernel(kernel_simple, cl::NullRange, cl::NDRange(binSize), cl::NullRange, NULL, &cumEvent); //function calls the kernel
			queue.enqueueReadBuffer(dev_cumulative, CL_TRUE, 0, histoSize, &cumHistogram[0], NULL, &memEvent); //cumulative histogram is the output
			//std::cout << cumHistogram << std::endl;
		}
		else if (answer == "local")
		{
			cl::Kernel kernel_local = cl::Kernel(program, "scan_local_hs");
			kernel_local.setArg(0, dev_histogram);
			kernel_local.setArg(1, dev_cumulative);
			kernel_local.setArg(2, cl::Local(histoSize));
			kernel_local.setArg(3, cl::Local(histoSize)); //using local memory so in practise should run faster like the local histogram
			queue.enqueueNDRangeKernel(kernel_local, cl::NullRange, cl::NDRange(binSize), cl::NDRange(binSize), NULL, &cumEvent); //function calls the kernel
			queue.enqueueReadBuffer(dev_cumulative, CL_TRUE, 0, histoSize, &cumHistogram[0], NULL, &memEvent); //cumulative histogram is the output
			//std::cout << cumHistogram << std::endl;
		}
		else
		{
			cl::Kernel kernel_scan = cl::Kernel(program, "scan_hs"); //this code uses the hillis steele
			kernel_scan.setArg(0, dev_histogram);
			kernel_scan.setArg(1, dev_cumulative);
			queue.enqueueNDRangeKernel(kernel_scan, cl::NullRange, cl::NDRange(binSize), cl::NullRange, NULL, &cumEvent);
			queue.enqueueReadBuffer(dev_cumulative, CL_TRUE, 0, histoSize, &cumHistogram[0], NULL, &memEvent);
			//std::cout << cumHistogram << std::endl;
		} //all scan functions will work the same way but have different complexities for work and span
		std::cout << "Cumulative Histogram Execution time in nanoseconds: " <<
			cumEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			cumEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		totalEvent += cumEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - cumEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		std::cout << "Full cumulative histogram kernel information: " << GetFullProfilingInfo(cumEvent, ProfilingResolution::PROF_US) << std::endl;
		queueEvent += cumEvent.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() - cumEvent.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		submitEvent += cumEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() - cumEvent.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
		std::cout << std::endl;
		cl::Kernel kernel_normal = cl::Kernel(program, "normalise"); //used to make normalised histogram
		kernel_normal.setArg(0, dev_cumulative); //take result from cumulative (scan) kernel and then output to the normalised histogram
		kernel_normal.setArg(1, dev_normalise);
		queue.enqueueNDRangeKernel(kernel_normal, cl::NullRange, cl::NDRange(binSize), cl::NullRange, NULL, &normEvent);
		queue.enqueueReadBuffer(dev_normalise, CL_TRUE, 0, histoSize, &nomHistogram[0], NULL, &memEvent);
		//std::cout << nomHistogram << std::endl;
		std::cout << "Normalised Histogram Execution time in nanoseconds: " <<
			normEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			normEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		totalEvent += normEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - normEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		queueEvent += normEvent.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() - normEvent.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		submitEvent += normEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() - normEvent.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
		std::cout << "Full normalised histogram kernel information: " << GetFullProfilingInfo(normEvent, ProfilingResolution::PROF_US) << std::endl;
		cl::Kernel kernel_back = cl::Kernel(program, "back_project"); //used to map back to original image
		std::cout << std::endl;
		kernel_back.setArg(0, dev_image_input); //takes the original image
		kernel_back.setArg(1, dev_image_output); //output to new image
		kernel_back.setArg(2, dev_normalise); //values in this histogram will be used as a look up table
		kernel_back.setArg(3, dev_bins);
		queue.enqueueNDRangeKernel(kernel_back, cl::NullRange, image_input.size(), cl::NullRange, NULL, &imageEvent);
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0], NULL, &memEvent); //data read in will work be used on an output image
		std::cout << "Look Up Table Execution time in nanoseconds: " <<
			imageEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			imageEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		totalEvent += imageEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - imageEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		queueEvent += imageEvent.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() - imageEvent.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		submitEvent += imageEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() - imageEvent.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
		std::cout << "Full look up table kernel information: " << GetFullProfilingInfo(imageEvent, ProfilingResolution::PROF_US) << std::endl;
		std::cout << std::endl;
		std::cout << "Accumulated Buffer Read/Write Time in nanoseconds: " <<
			memEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			memEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		totalEvent += memEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - memEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		queueEvent += memEvent.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() - memEvent.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		submitEvent += memEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() - memEvent.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
		std::cout << "Accumulated Buffer Read/Write information: " << GetFullProfilingInfo(memEvent, ProfilingResolution::PROF_US) << std::endl;
		std::cout << std::endl;
		std::cout << "Full memory transfer in nanoseconds " << totalEvent << std::endl; //total event will only be taking the combined execution times of each kernel as well as the write and read buffers. This is therefore the total memory transfer of the program 
		std::cout << "Full queue times for program in nanoseconds " << queueEvent << std::endl; //output total queue time of program
		std::cout << "Full submission times for program in nanoseconds " << submitEvent << std::endl; //output total submission time
		std::cout << "Total program performance in nanoseconds " << totalEvent + queueEvent + submitEvent << std::endl; //total program execution time is equal to total queue time + total submission time + total execution time
		//GetFullProfilingInfo rounds the values which means there will be slight variance between the nanoseconds value and the microseconds value as the nanoseconds is more exact as it isn't rounding 
		std::cout << std::endl;
		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		if (colour)
		{
			for (int i = 0; i < output_image.width(); i++)
			{
				for (int j = 0; j < output_image.height(); j++)
				{
					converted_input(i, j, 0) = output_image(i, j); //the output_image has the values for the updated intensities which we will update the original image
				}
			}
			output_image = converted_input.get_YCbCrtoRGB(); //convert the output image back to RGB
		}

		std::cout << "Histogram" << std::endl;
		std::cout << histogram << std::endl;
		std::cout << std::endl;
		std::cout << "Cumulative Histogram" << std::endl;
		std::cout << cumHistogram << std::endl;
		std::cout << std::endl;
		std::cout << "Normalised Histogram" << std::endl;
		std::cout << nomHistogram << std::endl;
		std::cout << std::endl;
		CImgDisplay disp_output(output_image, "output");
		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}