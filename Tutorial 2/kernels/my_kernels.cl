//a simple OpenCL kernel which copies all pixels from A to B
kernel void histoCopy(global uint* A, global uint* B) 
{
	int id = get_global_id(0);
	B[id] = A[id];
}
kernel void hist_simple(global const uchar* A, global int* H, global const int* nr_bins) { 
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = (A[id] / 255.0f) * (nr_bins[0] - 1); //take value as a bin index
	//bins would go from 0 to max size of bin so we need to subtract one from bin size
	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}//simple histogram taken from tutorial 3 workshops
kernel void hist_atomic(global const uchar* A, global uint* H, local uint* LH, global const int* nr_bins) 
{
	int id = get_global_id(0); //id will be looking through each pixel in the image
	int lid = get_local_id(0); //lid will be the index for each work group of the histogram
	int bin_index = (A[id] / 255.0f) * (nr_bins[0] - 1); //maximum size a uchar can be is 255, and we need to essentially normalise our bin_index to be within the size of the bins
	//reason this is done is because the size of the bins could be <256 and A[id] could exceed 255
	LH[lid] = 0; //initialise each index of the partial histogram as 0
	barrier(CLK_LOCAL_MEM_FENCE); //sync the step
	//increment function
	atomic_inc(&LH[bin_index]); //assumes bins are set to 0 first, then increment everytime the bin_index is found
	barrier(CLK_LOCAL_MEM_FENCE); //sync step again
	if(lid < nr_bins[0]) atomic_add(&H[lid], LH[lid]); //merge histograms together
}//more efficent histogram taken from the lectures
kernel void scan_simple(global uint* A, global uint* B) 
{
	int id = get_global_id(0); //id will be each element in the histogram
	int N = get_global_size(0); //essentially a serial implimentation
	int temp = 0;
	for(int i = 0; i < id + 1; i++) //id will be the limit of what we go up to, for example id = 0 would mean only adding up to the 1st element and so on
	{
		temp += A[i]; //each value will increase by each value in the histogram
	}
	B[id] = temp; //the cumulative histogram at id index will be the addition of all previous ids
}
kernel void scan_hs(global uint* A, global uint* B) { //both the hillis steele scans and blelloch scan are taken from tutorial 3 workshops
	int id = get_global_id(0);
	int N = get_global_size(0);
	global uint* C; //additional buffer used to avoid data overwrite and keep scan inclusive

	for (int stride = 1; stride <= N; stride *= 2) { //stride will increase by the power of 2 each time until it is equal to the size of the histogram
		B[id] = A[id];
		if (id >= stride)
			B[id] += A[id - stride]; //B[id] will be all the added elements from the previous stride

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

		C = A; A = B; B = C; //swap A & B between steps
	}
}
kernel void scan_local_hs(__global const uint* A, global uint* B, local uint* scratch_1, local uint* scratch_2) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	local int *scratch_3;//used for buffer swap

	//cache all N values from global memory to local memory
	scratch_1[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i <= N; i *= 2) {
		if (lid >= i)
			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
		else
			scratch_2[lid] = scratch_1[lid];

		barrier(CLK_LOCAL_MEM_FENCE);

		//buffer swap
		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;
	}

	//copy the cache to output array
	B[id] = scratch_1[lid];
}
kernel void scan_bl(global uint* A) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	int t;

	//up-sweep
	for (int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride*2)) == 0)
			A[id] += A[id - stride]; //A[id] adds elements in current stride

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}

	//down-sweep
	if (id == 0)
		A[N-1] = 0;//exclusive scan

	barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

	for (int stride = N/2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride*2)) == 0) {
			t = A[id]; //take current value
			A[id] += A[id - stride]; //reduce and get next value for t (addition of prior value)
			A[id - stride] = t;		 //move
		}

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}
}
kernel void normalise(global uint* A, global uint* H) 
{
	int id = get_global_id(0);
	int last = A[get_global_size(0) - 1]; //casted to a float later to prevent integar division
	H[id] = (A[id] / (float)last) * 255; //255 is the maximum value a pixel could be (as it goes from 0-255)
	//last is the maximum value of the histogram, 255 is the maximum value of a pixel
}
kernel void back_project(global const uchar* A, global uchar* B, global uint* H, global const int* nr_bins) 
{
	int id = get_global_id(0);
	int last = nr_bins[0] - 1; //get last element of the histogram
	int pixel = (A[id] / 255.0f) * last; 
	B[id] = H[pixel]; //like we do in the histogram we need to normalise for our bin size
}