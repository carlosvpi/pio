#include "nn.h"

struct _array {
	float* data;
	int length;
};

// newArray returns a data structure containing an array of floats and their length
ARRAY newArray (int length) {
	ARRAY array = (ARRAY) malloc(sizeof(struct _array));
	array->data = (float*) calloc(length, sizeof(float));
	array->length = length;
	return array;
}

// cleanArray sets the array's floats to 0
void cleanArray (ARRAY array) {
	memset(array->data, 0, array->length * sizeof(float));
}

// copyData copies the data of an array to a pointer
void copyData (float* dst, ARRAY array) {
	memcpy(dst, array->data, array->length * sizeof(float));
}

// freeArray frees the memory of an array
void freeArray (ARRAY array) {
	free(array->data);
	free(array);
}

// assignData assigns some data to the array's float pointer
void assignData (ARRAY array, float* data) {
	array->data = data;
}

// A layer represents the weights and bases between an input and an output arrays
struct _layer {
	float** weights;
	float* bases;
	ARRAY input;
	ARRAY output;
};

LAYER newLayer (ARRAY input, int length) {
	LAYER layer = (LAYER) malloc(sizeof(struct _layer));
	layer->input = input;
	layer->output = newArray(length);
	layer->weights = (float**) malloc(input->length * sizeof(float*));
	layer->bases = (float*) malloc(length * sizeof(float));
	for (int i = 0; i < input->length; i++) {
		layer->weights[i] = (float*) malloc(length * sizeof(float));
		for (int j = 0; j < length; j++) {
			layer->weights[i][j] = 2 * ((float) rand()) / ((float) RAND_MAX) - 1;
		}
	}
	for (int j = 0; j < length; j++) {
		layer->bases[j] = 2 * ((float) rand()) / ((float) RAND_MAX) - 1;
	}
	return layer;
}

void freeLayer (LAYER layer) {
	for (int i = 0; i < layer->input->length; i++) {
		free(layer->weights[i]);
	}
	free(layer->weights);
	free(layer->bases);
	freeArray(layer->input);
	freeArray(layer->output);
	free(layer);
}

ARRAY execLayer (LAYER layer) {
	cleanArray(layer->output);
	for (int j = 0; j < layer->output->length; j++) {
		for (int i = 0; i < layer->input->length; i++) {
			layer->output->data[j] += layer->input->data[i] * layer->weights[i][j];
		}
		layer->output->data[j] += layer->bases[j];
		layer->output->data[j] = 1 / (1 + exp(-layer->output->data[j]));
	}
	return layer->output;
}

struct _nn {
	ARRAY input;
	LAYER* layers;
	int length;
};

NN newNN (ARRAY input, int length, int* lengths) {
	NN nn = (NN) malloc(sizeof(struct _nn));
	nn->layers = (LAYER*) malloc(length * sizeof(LAYER));
	nn->length = length;
	nn->input = input;
	for (int i = 0; i < length; i++) {
		nn->layers[i] = newLayer(input, lengths[i]);
		input = nn->layers[i]->output;
	}
	return nn;
}

void freeNN(NN nn) {
	freeArray(nn->input);
	for (int i = 0; i < nn->length; i++) {
		freeLayer(nn->layers[i]);
	}
	free(nn->layers);
	free(nn);
}

NN execNN (NN nn, float* inputData) {
	if (inputData != NULL) {
		assignData(nn->input, inputData);
	}
	for (int i = 0; i < nn->length; i++) {
		execLayer(nn->layers[i]);
	}
	return nn;
}

float* getOutput (NN nn) {
	return nn->layers[nn->length - 1]->output->data;
}

// ARRAY getErrors (float* obtained, float* expected, int length) {
// 	ARRAY errors = newArray(length);
// 	for (int i = 0; i < length; i++) {
// 		errors[i] = expected[i] - obtained[i]
// 		errors[i] = errors[i] * errors[i] / 2
// 	}
// 	return errors;
// }

int getWidth (NN nn) {
	int width = 0;
	for (int i = 0; i < nn->length; i++) {
		if (nn->layers[i]->input->length > width) {
			width = nn->layers[i]->input->length;
		}
	}
	return width;
}

NN backpropagate (NN nn, float* expected, float alpha) {
	float error, out;
	int lastLayer = nn->length - 1;
	int width = getWidth(nn);
	size_t byteWidth = width * sizeof(float);
	float* deltas1 = (float*) malloc(byteWidth);
	float* deltas2 = (float*) malloc(byteWidth);
	float* currentDeltas;
	float* futureDeltas;

	for (int i = 0; i < nn->layers[lastLayer]->output->length; i++) {
		deltas1[i] = nn->layers[lastLayer]->output->data[i] - expected[i];
	}
	for (int layer = nn->length - 1; layer >= 0; layer--) {
		currentDeltas = (nn->length - layer) % 2 == 1 ? deltas1 : deltas2;
		futureDeltas = (nn->length - layer) % 2 == 1 ? deltas2 : deltas1;
		memset(futureDeltas, 0, byteWidth);
		for (int j = 0; j < nn->layers[layer]->output->length; j++) {
			out = nn->layers[layer]->output->data[j];
			error = currentDeltas[j] * out * (1 - out); // D(Ei/outi) * D(outi/neti)
			for (int i = 0; i < nn->layers[layer]->input->length; i++) {
				nn->layers[layer]->weights[i][j] -= alpha * error * nn->layers[layer]->input->data[i]; // D(Ei/neti) * D(neti/wij)
				futureDeltas[i] += error * nn->layers[layer]->weights[i][j]; // D(Ei/neti) * D(neti/ini)
			}
			nn->layers[layer]->bases[j] -= alpha * error * nn->layers[layer]->bases[j]; // D(Ei/neti) * D(neti/bi)
		}
	}
	return nn;
}

float calculateError(float* output, float* expected, int length) {
	float error = 0;
	for (int i = 0; i < length; i++) {
		error += (expected[i] - output[i]) * (expected[i] - output[i]) / 2;
	}
	return error;
}

struct _train_options {
	int maxTime;
	float maxError;
	float alpha;
};

TRAIN_OPTIONS newTrainOptions (int maxTime, float maxError, float alpha) {
	TRAIN_OPTIONS trainOptions = (TRAIN_OPTIONS) malloc(sizeof(struct _train_options));
	trainOptions->maxTime = maxTime;
	trainOptions->maxError = maxError;
	trainOptions->alpha = alpha;
	return trainOptions;
}

void freeTrainOptions (TRAIN_OPTIONS trainOptions) {
	free(trainOptions);
}

struct _console {
	int elapsed;
	float error;
	int loops;
};

CONSOLE newConsole () {
	CONSOLE console = (CONSOLE) malloc(sizeof(struct _console));
	console->elapsed = 0;
	console->error = 0;
	console->loops = 0;
	return console;
}

void freeConsole (CONSOLE console) {
	free(console);
}

void logElapsed (CONSOLE console, int elapsed) {
	if (console == NULL) return;
	console->elapsed = elapsed;
}

void logError (CONSOLE console, float error) {
	if (console == NULL) return;
	console->error = error;
}

void logIncrementLoops (CONSOLE console) {
	if (console == NULL) return;
	console->loops++;
}

void startConsole () {
	printf("Elapsed\tLoops\tError\n");
}

void endConsole () {
	printf("\n");
}

void displayConsole (CONSOLE console) {
	printf("\r%ds\t%d\t%f", console->elapsed, console->loops, console->error);
}

NN train(NN nn, float** inputs, float** outputs, int testBedSize, TRAIN_OPTIONS trainOptions, CONSOLE console) {
	int maxTime = trainOptions == NULL ? 10 : trainOptions->maxTime;
	float maxError = trainOptions == NULL ? 0.1 : trainOptions->maxError;
	float alpha = trainOptions == NULL ? 0.1 : trainOptions->alpha;
	int startTime = time(NULL);
	int elapsed = time(NULL) - startTime;
	float error;
	do {
		error = 0;
		for (int i = 0; i < testBedSize && elapsed < maxTime; i++) {
			execNN(nn, inputs[i]);
			error += calculateError(nn->layers[nn->length - 1]->output->data, outputs[i], nn->layers[nn->length - 1]->output->length);
			backpropagate(nn, outputs[i], alpha);
			elapsed = time(NULL) - startTime;
			logElapsed(console, elapsed);
		}
		error = error / testBedSize;
		logError(console, error);
		logIncrementLoops(console);
		displayConsole(console);
	} while (elapsed < maxTime && error > maxError);
	return nn;
}

NN loadNNRaw(FILE* file) {
	int inputLength;
	int length;
	int* lengths;
	fread(&inputLength, 1, sizeof(int), file);
	fread(&length, 1, sizeof(int), file);
	lengths = (int*) malloc(length * sizeof(int));
	for (int i = 0; i < length; i++) {
		fread(&(lengths[i]), 1, sizeof(int), file);
	}
	NN nn = newNN(newArray(inputLength), length, lengths);
	for (int i = 0; i < length; i++) {
		fread(nn->layers[i]->weights, sizeof(float), lengths[i] * lengths[i + 1], file);
		fread(nn->layers[i]->bases, sizeof(float), lengths[i + 1], file);
	}
	return nn;
}

NN loadNN(char* filename) {
	FILE* file = fopen(filename, "rb");
	if (file == NULL) {
		return NULL;
	}
	NN nn = loadNNRaw(file);
	fclose(file);
	return nn;
}

void saveNNRaw(NN nn, FILE* file) {
	fwrite(&nn->input->length, 1, sizeof(int), file);
	fwrite(&nn->length, 1, sizeof(int), file);
	for (int i = 0; i < nn->length; i++) {
		fwrite(&nn->layers[i]->input->length, 1, sizeof(int), file);
	}
	for (int i = 0; i < nn->length; i++) {
		fwrite(nn->layers[i]->weights, sizeof(float), nn->layers[i]->input->length * nn->layers[i]->output->length, file);
		fwrite(nn->layers[i]->bases, sizeof(float), nn->layers[i]->output->length, file);
	}
}

void saveNN(NN nn, char* filename) {
	FILE* file = fopen(filename, "wb");
	if (file == NULL) {
		return;
	}
	saveNNRaw(nn, file);
	fclose(file);
}