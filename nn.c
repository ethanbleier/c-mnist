#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.001f
#define EPOCHS 20
#define BATCH_SIZE 64
#define IMAGE_SIZE 28
#define TRAIN_SPLIT 0.8

#define TRAIN_IMG_PATH "data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "data/train-labels.idx1-ubyte"

typedef struct {
	float *weights, *biases;
	int input_size, output_size;
} Layer;

typedef struct {
	Layer hidden, output;
} Network;

typedef struct {
	unsigned char *images, *labels;
	int nImages;
} InputData;

void init_layer(Layer *layer, int in_size, int out_size) {
	int n = in_size * out_size;
	float scale = sqrtf(2.0f / in_size);
	 layer->input_size = in_size;
	 layer->output_size = out_size;
	 layer->weights = malloc(n * sizeof(float));
	 layer->biases = calloc(out_size, sizeof(float));

	 for (int i = 0; i < n; i++) 
	 	layer->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
}

void forward(Layer *layer, float *input, float *output) {
	for (int i = 0; i < layer->output_size; i++) {
		output[i] = layer->biases[i];
		for (int j = 0; j < layer->input_size; j++) 
			output[i] += input[j] * layer->weights[j * layer->output_size + i];
	}
}

void backward(Layer *layer, float *input, float *output_grad, float *input_grad, float lr) {
	for (int i = 0; i < layer->output_size; i++) {
		for (int j = 0; j < layer->input_size; j++) {
			int idx = j * layer->output_size + i;
			float grad = output_grad[i] * input[j];
			layer->weights[idx] -= lr * grad;
			if (input_grad)
				input_grad[j] += output_grad[i] * layer->weights[idx];
		}
		layer->biases[i] -= lr * output_grad[i];
	}
}

void read_mnist_images(const char *filename, unsigned char **images, int *nImages) {
	FILE *file = fopen(filename, "rb");
	if (!file) exit(1);

	int temp, rows, cols;
	fread(&temp, sizeof(int), 1, file);
	fread(nImages, sizeof(int), 1, file);
	*nImages = __builtin_bswap32(*nImages);
	
	fread(&rows, sizeof(int), 1, file);
	fread(&cols, sizeof(int), 1, file);

	rows = __builtin_bswap32(rows);
	cols = __builtin_bswap32(cols);

	*images = malloc((*nImages) * IMAGE_SIZE * IMAGE_SIZE);
	fread(*images, sizeof(unsigned char), (*nImages) * IMAGE_SIZE * IMAGE_SIZE, file);
	fclose(file);
}

void read_mnist_labels(const char *filename, unsigned char **labels, int *nLabels) {
	FILE *file = fopen(filename, "rb");
	if (!file) exit(1);

	int temp;
	fread(&temp, sizeof(int), 1, file);
	fread(nLabels, sizeof(int), 1, file);
	*nLabels = __builtin_bswap32(*nLabels);

	*labels = malloc(*nLabels);
	fread(*labels, sizeof(unsigned char), *nLabels, file);
	fclose(file);
}

int main() {
	Network net;
	InputData data = {0};
	float learning_rate = LEARNING_RATE, img[INPUT_SIZE];

	srand(time(NULL));



	read_mnist_images("datasets/train-images.idx3-ubyte", NULL, NULL);
	read_mnist_labels("datasets/train-labels.idx1-ubyte", NULL, NULL);
}
