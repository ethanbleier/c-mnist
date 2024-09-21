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
	read_mnist_images("datasets/train-images.idx3-ubyte", NULL, NULL);
	read_mnist_labels("datasets/train-labels.idx1-ubyte", NULL, NULL);
}
