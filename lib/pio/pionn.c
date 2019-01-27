#define SENTENCE_LENGTH 140
#define MEMORY_LENGTH 100

struct _pio {
	NN nn;
	int sentenceLength;
	float*** data; // [depth][layer (inputs, not outputs!)][index of datum]
};

typedef struct _pio* PIO;

PIO newPio (NN nn) {
	int length = 3;
	PIO pio = (PIO) malloc(sizeof(struct _pio));
	pio->nn = nn != NULL ? nn : newNN(newArray(ASCII_DATA_LENGTH + MEMORY_LENGTH), length, (int[3]) {ASCII_DATA_LENGTH + MEMORY_LENGTH, ASCII_DATA_LENGTH + MEMORY_LENGTH, ASCII_DATA_LENGTH + MEMORY_LENGTH});
	pio->sentenceLength = SENTENCE_LENGTH;
	pio->data = (float***) malloc(pio->sentenceLength * sizeof(float**));
	for (int i = 0; i < pio->sentenceLength; i++) {
		pio->data[i] = (float**) malloc((pio->nn->length + 1) * sizeof(float*)); // add +1 to get the final output
		for (int layer = 0; layer < pio->nn->length; layer++) {
			pio->data[i][layer] = (float*) malloc(pio->nn->layers[layer]->input->length * sizeof(float));
		}
		pio->data[i][pio->nn->length] = (float*) malloc(pio->nn->layers[pio->nn->length - 1]->output->length * sizeof(float));
	}
	return pio;
}

void freePio (PIO pio) {
	for (int i = 0; i < pio->sentenceLength; i++) {
		for (int layer = 0; layer < pio->nn->length; layer++) {
			free(pio->data[i][layer]);
		}
		free(pio->data[i]);
	}
	free(pio->data);
	freeNN(pio->nn);
}

int getSentenceLength (char* sentence) {
	int i = 0;
	while(sentence[i++]) {}
	return i;
}

PIO backpropagatePio (PIO pio, char* sentence, float alpha) {
	NN nn = pio->nn;
	float error, out;
	int lastLayer = nn->length - 1;
	int width = getWidth(nn);
	size_t byteWidth = width * sizeof(float);
	float* deltas1 = (float*) malloc(byteWidth);
	float* deltas2 = (float*) malloc(byteWidth);
	float* currentDeltas;
	float* futureDeltas;
	int minOutputIndex = 0;
	int maxOutputIndex = ASCII_DATA_LENGTH;
	int sentenceLength = getSentenceLength(sentence);
	float* charData;
	// int minInputIndex = 0;
	// int maxInputIndex = ASCII_DATA_LENGTH;

	for (int charIndex = sentenceLength - 1; charIndex >= 0; charIndex--) {
		charData = ascii2data(sentence[charIndex], NULL);
		for (int i = 0; i < ASCII_DATA_LENGTH; i++) {
			deltas1[i] = pio->data[charIndex][lastLayer + 1][i] - charData[i];
		}
		for (int layer = nn->length - 1; layer >= 0; layer--) {
			currentDeltas = (nn->length - layer) % 2 == 1 ? deltas1 : deltas2;
			futureDeltas = (nn->length - layer) % 2 == 1 ? deltas2 : deltas1;
			memset(futureDeltas, 0, byteWidth);
			if (layer == nn->length - 1) {
				minOutputIndex = charIndex == sentenceLength - 1 ? 0 : ASCII_DATA_LENGTH;
				maxOutputIndex = charIndex == sentenceLength - 1 ? ASCII_DATA_LENGTH : nn->layers[layer]->output->length;
			} else {
				minOutputIndex = 0;
				maxOutputIndex = nn->layers[layer]->output->length;
			}
			// if (layer == 0) {
			// 	minInputIndex = 
			// } else {
			// 	minInputIndex = 0;
			// 	maxInputIndex = nn->layers[layer]->input->length;
			// }
			for (int j = minOutputIndex; j < maxOutputIndex; j++) {
				out = pio->data[charIndex][layer + 1][j]; // layer + 1 because data stores the inputs, not the outputs
				error = currentDeltas[j] * out * (1 - out); // D(Ei/outi) * D(outi/neti)
				for (int i = 0; i < nn->layers[layer]->input->length; i++) {
					nn->layers[layer]->weights[i][j] -= alpha * error * pio->data[charIndex][layer][i]; // D(Ei/neti) * D(neti/wij)
					futureDeltas[i] += error * nn->layers[layer]->weights[i][j]; // D(Ei/neti) * D(neti/ini)
				}
				nn->layers[layer]->bases[j] -= alpha * error * nn->layers[layer]->bases[j]; // D(Ei/neti) * D(neti/bi)
			}
		}
	}
	return pio;
}

PIO generateAndCompare (PIO pio, char* sentence) { // fills the pio->data
	int i = 0;
	NN nn = pio->nn;
	// Set to 0 memory-part of input
	memset(nn->layers[0]->input->data + ASCII_DATA_LENGTH, 0, MEMORY_LENGTH);

	while (sentence[i]) {
		// Put char in char-part of input
		ascii2data(sentence[i], nn->layers[0]->input->data);

		execNN(nn, NULL);

		// Copy nn layers data to nn->data (input of layer L to data[L])
		for (int layer = 0; layer < nn->length; layer++) {
			memcpy(pio->data[i][layer], nn->layers[layer]->input->data, nn->layers[layer]->input->length * sizeof(float));
		}
		memcpy(pio->data[i][nn->length], nn->layers[nn->length - 1]->output->data, nn->layers[nn->length - 1]->output->length * sizeof(float));

		// Copy memory-part of output to memory-part of input
		memcpy(nn->layers[0]->input->data + ASCII_DATA_LENGTH, nn->layers[nn->length - 1]->output->data + ASCII_DATA_LENGTH, MEMORY_LENGTH * sizeof(float));

		i++;
	}
	return pio;
}

char* generate (PIO pio) { // fills the pio->data
	NN nn = pio->nn;
	char* output = (char*) malloc((SENTENCE_LENGTH + 1) * sizeof(char*));
	// Set to 0 input
	memset(nn->layers[0]->input->data, 0, nn->layers[0]->input->length);
	for (int i = 0; i < SENTENCE_LENGTH; i++) {
		execNN(nn, NULL);

		// Copy output to input
		memcpy(nn->layers[0]->input->data, nn->layers[nn->length - 1]->output->data, (MEMORY_LENGTH + ASCII_DATA_LENGTH) * sizeof(float));

		output[i] = data2ascii(nn->layers[nn->length - 1]->output->data);
	}
	output[SENTENCE_LENGTH] = 0;
	return output;
}

float calculatePioError (PIO pio, char* sentence) {
	int i = 0;
	float error = 0;

	while (sentence[i + 1]) {
		error += calculateError(pio->data[i][pio->nn->length], ascii2data(sentence[i + 1], NULL), ASCII_DATA_LENGTH);
		i++;
	}

	return error / i;
}

PIO trainPio (PIO pio, float alpha, char** sentences, int testBedSize, TRAIN_OPTIONS trainOptions, CONSOLE console) {
	int maxTime = trainOptions == NULL ? 10 : trainOptions->maxTime;
	float maxError = trainOptions == NULL ? 1.4 : trainOptions->maxError;
	int startTime = time(NULL) / 1000;
	int elapsed = time(NULL) / 1000 - startTime;
	float error;
	NN nn = pio->nn;
	do {
		error = 0;
		for (int i = 0; i < testBedSize && elapsed < maxTime; i++) {
			generateAndCompare(pio, sentences[i]);
			error += calculatePioError(pio, sentences[i]);
			backpropagatePio(pio, sentences[i], alpha);
			elapsed = time(NULL) / 1000 - startTime;
			logElapsed(console, elapsed);
		}
		error = error / testBedSize;
		logError(console, error);
		logIncrementLoops(console);
		displayConsole(console);
		fflush(stdout);
	} while (elapsed < maxTime && error > maxError);
	return pio;
}
