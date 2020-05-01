#include <assert.h>
#include <stdio.h>

#define NAMELEN 255

#define CSC(call) do { 		\
	cudaError_t e = call;	\
	if (e != cudaSuccess) {	\
		fprintf(stderr, "ERROR: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
		exit(0);			\
	}						\
} while(0)




texture<uchar4, 2, cudaReadModeElementType> tex;


__global__ void kernel(uchar4 *dev_data, int w, int h, int r, float *dev_a, bool isColumns){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	int offsetx = gridDim.x * blockDim.x;
	int offsety = gridDim.y * blockDim.y;


	uchar4 pixel;
	float red, green, blue;

	for(int x = idx; x < w; x += offsetx){
		for(int y = idy; y < h; y += offsety){
			red = 0.0;
			green = 0.0;
			blue = 0.0;

			for(int i = -r; i <=r; i++){

				if (isColumns) {
				pixel = tex2D(tex, x + i, y);
				}
				else {
				pixel = tex2D(tex, x, y + i);
				}

				red += dev_a[r + i] * pixel.x;
				green += dev_a[r + i] * pixel.y;
				blue += dev_a[r + i] * pixel.z;
			}

			dev_data[y * w + x] = make_uchar4(red, green, blue, 0.0);
		}
	}
}

class CUGaussianBlur {
public:

	float *dev_a;
	cudaArray *arr;
	uchar4 *data;
	uchar4 *dev_data;
	cudaChannelFormatDesc ch;

	int w;
	int h;

	CUGaussianBlur(int r, int _w, int _h, uchar4 *_data)  {

		w = _w;
		h = _h;
		data = _data;

		
		int n = 2 * r + 1;
		float sum = 0.0;

		float a[n];

		for(int i = -r; i <= r; i++) {
			a[i + r] = exp(-1.0 * (i * i) / (2 * r * r));
			sum += a[i + r];
		}

		for(int i = 0; i < n; i++){
			a[i] /= sum;
		}

		CSC(cudaMalloc(&dev_a, sizeof(float) * n));
		CSC(cudaMemcpy(dev_a, a, sizeof(float) * n, cudaMemcpyHostToDevice));

		
		ch = cudaCreateChannelDesc<uchar4>();
		CSC(cudaMallocArray(&arr, &ch, w, h));
		CSC(cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * h * w, cudaMemcpyHostToDevice));

		tex.addressMode[0] = cudaAddressModeClamp;
		tex.addressMode[1] = cudaAddressModeClamp;
		tex.channelDesc = ch;
		tex.filterMode = cudaFilterModePoint;
		tex.normalized = false;

		CSC(cudaBindTextureToArray(tex, arr, ch));
		CSC(cudaMalloc(&dev_data, sizeof(uchar4) * w * h));

	} 

	void run_kernel(int r) {
		kernel <<<dim3(8,16), dim3(8,32) >>> (dev_data, w, h, r, dev_a, false);

		CSC(cudaUnbindTexture(tex));
		CSC(cudaMemcpyToArray(arr, 0, 0, dev_data, sizeof(uchar4) * w * h, cudaMemcpyDeviceToDevice));
		CSC(cudaBindTextureToArray(tex, arr, ch));

		kernel <<< dim3(8,16), dim3(16,32) >>> (dev_data, w, h, r, dev_a, true);


		CSC(cudaMemcpy(data, dev_data, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

	}

	~CUGaussianBlur(){

		free(data);

		CSC(cudaUnbindTexture(tex));
		CSC(cudaFreeArray(arr));
		CSC(cudaFree(dev_data));
		CSC(cudaFree(dev_a));	
	}
};


int main() {
	int r, w, h;
	
	char input_name[256];
	char output_name[256];


	scanf("%s", input_name);
	scanf("%s", output_name);
	scanf("%d", &r);

	FILE *in = fopen(input_name, "rb");

	fread(&w, sizeof(int), 1, in);
	fread(&h, sizeof(int), 1, in);

	uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
	

	fread(data, sizeof(uchar4), w * h, in);
	fclose(in);

	CUGaussianBlur image = CUGaussianBlur(r, w, h, data);
	if(r != 0){
		
		image.run_kernel(r);
	}

	FILE *out = fopen(output_name, "wb");
	fwrite(&w, sizeof(int), 1, out);
	fwrite(&h, sizeof(int), 1, out);
	fwrite(image.data, sizeof(uchar4), w * h, out);

	fclose(out);

	return 0;
}