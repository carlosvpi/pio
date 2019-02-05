#define SENTENCE_LENGTH 140
#define MEMORY_LENGTH 50

NN newPio () {
	int length = 3;
	return newNN(
		newArray(ASCII_DATA_LENGTH + MEMORY_LENGTH),
		length,
		(int[3]) {ASCII_DATA_LENGTH + MEMORY_LENGTH, ASCII_DATA_LENGTH + MEMORY_LENGTH, ASCII_DATA_LENGTH + MEMORY_LENGTH}
	);
}

int getSentenceLength (char* sentence) {
	int i = 0;
	while(sentence[i++]) {}
	return i - 1;
}

char* generate (NN nn, char firstChar) { // fills the pio->data
	char* output = (char*) malloc((SENTENCE_LENGTH + 1) * sizeof(char*));
	// Set to 0 input
	for (int i = 0; i < nn->layers[0]->input->length; i++) {
		nn->layers[0]->input->data[i] = 0;
	}
	ascii2data(firstChar, nn->layers[0]->input->data);
	output[0] = firstChar;
	for (int i = 0; i < SENTENCE_LENGTH; i++) {
		execNN(nn, NULL);

		// Copy output to input
		memcpy(nn->layers[0]->input->data, nn->layers[nn->length - 1]->output->data, (MEMORY_LENGTH + ASCII_DATA_LENGTH) * sizeof(float));

		output[i + 1] = data2ascii(nn->layers[nn->length - 1]->output->data);
	}
	output[SENTENCE_LENGTH] = 0;
	return output;
}


/* -nn:		 		neural network (consider only weights and base weights)
 * -data:	 		neural network data (replace nn's data by this in the computation)
 * -alpha:			alpha
 * -finalDeltas:	deltas of the output layer
 * -backPropagateMemory: 1=apply backpropa in the last layer only to the memory-part of the output; 0=apply backpropa in the last layer only to the char-part of the output
 * +deltas:			deltas of the input layer computed by the backpropagation algorithm. To be used for previous characters
 */

NN backpropagatePio (NN nn, float** data, float alpha, float* finalDeltas, char backPropagateMemory, float* deltas) {
	float error, out;
	int width = getWidth(nn);
	size_t byteWidth = width * sizeof(float);
	float* deltas1 = (float*) malloc(byteWidth);
	float* deltas2 = (float*) malloc(byteWidth);
	float* currentDeltas;
	float* futureDeltas;
	int minOutputIndex = 0;
	int maxOutputIndex = ASCII_DATA_LENGTH;

	memcpy(deltas1, finalDeltas, (ASCII_DATA_LENGTH + MEMORY_LENGTH) * sizeof(float));

	// Compute backpropagation
	for (int layer = nn->length - 1; layer >= 0; layer--) {
		// swap pointers to deltas (we alternate between these two, avoiding having to reserve more memory inside the loop)
		currentDeltas = (nn->length - layer) % 2 == 1 ? deltas1 : deltas2;
		futureDeltas = (nn->length - layer) % 2 == 1 ? deltas2 : deltas1;
		for (int i = 0; i < width; i++) {
			futureDeltas[i] = 0;
		}

		// determine the range of the output for which we will modify the weights,
		// for the last layer, we select the Char-part or the memory-part depending on backPropagateMemory
		// for other layers we select the whole (current-layer) output
		minOutputIndex = (layer == nn->length - 1 && backPropagateMemory == 1) ? ASCII_DATA_LENGTH : 0;
		maxOutputIndex = (layer == nn->length - 1 && backPropagateMemory == 0) ? ASCII_DATA_LENGTH : nn->layers[layer]->output->length;
		// printf("\n%d to %d\n",minOutputIndex, maxOutputIndex);
		for (int j = minOutputIndex; j < maxOutputIndex; j++) {
			out = data[layer + 1][j]; // layer + 1 because data stores the inputs, not the outputs
			error = currentDeltas[j] * out; // * (1 - out); // D(Ei/outi) * D(outi/neti)
			// if (j == 7 || j == 6 || j == 37) {
			// 	printf("i: %d, out: %f, currDelt: %f, error: %f\n", j, out, currentDeltas[j], error);
			// }
			// if (backPropagateMemory) {
			// 	printf("%f ", currentDeltas[j]);
			// }
			// printf("j: %d, error: %f\n---------------------------\n", j, error);
			for (int i = 0; i < nn->layers[layer]->input->length; i++) {
				futureDeltas[i] += error * nn->layers[layer]->weights[i][j]; // D(Ei/neti) * D(neti/ini)
				// if (layer == nn->length - 1) {
				// 	printf("i: %d, %f -> ", i, nn->layers[layer]->weights[i][j]);
				// }
				nn->layers[layer]->weights[i][j] -= alpha * error * data[layer][i]; // D(Ei/neti) * D(neti/wij)
				// if (layer == nn->length - 1) {
				// 	printf("%f, (%f)\n", nn->layers[layer]->weights[i][j], alpha * error * data[layer][i]);
				// }
			}
			// if (layer == nn->length - 1) {
			// 	printf("\n");
			// }
			nn->layers[layer]->bases[j] -= alpha * error; // D(Ei/neti) * D(neti/bi)
		}
		// exit(0);
		// if (backPropagateMemory) {
		// 	printf("\n");
		// }
	}

	if (deltas != NULL) {
		memcpy(deltas, futureDeltas, nn->layers[0]->input->length * sizeof(float));
		// printf("%d\n----\n", nn->layers[0]->input->length);
		// for (int i = 0; i < ASCII_DATA_LENGTH + MEMORY_LENGTH; i++) {
		// 	printf("%d, %f = %f\n", i, deltas[i], futureDeltas[i]);
		// }
		// exit(0);

		// for (int i = 0; i < ASCII_DATA_LENGTH + MEMORY_LENGTH; i++) {
		// 	printf("%f ", futureDeltas[i]);
		// }
		// printf("\n");
	}
	free(currentDeltas);
	free(futureDeltas);

	return nn;
}

NN trainPio (NN nn, char** sentences, int sentencesLength, TRAIN_OPTIONS trainOptions, CONSOLE console) {
	int maxTime = trainOptions == NULL ? 10 : trainOptions->maxTime;
	float maxError = trainOptions == NULL ? 1.4 : trainOptions->maxError;
	float alpha = trainOptions == NULL ? 0.1 : trainOptions->alpha;
	int startTime = time(NULL);
	int elapsed = time(NULL) - startTime;
	float error;
	float* charData = (float*) malloc(ASCII_DATA_LENGTH * sizeof(float));
	// Save sentences lengths
	int* sentenceLengths = (int*) malloc(sentencesLength * sizeof(int));
	for (int s = 0; s < sentencesLength; s++) {
		sentenceLengths[s] = getSentenceLength(sentences[s]);
	}
	// data stores for each sentence s * chars c and c-1 in s, the values of nn's layers (inputs) and nn's output in order to perform shuffled backpropagation
	int totalChars = 0;
	char* chars = (char*) malloc(totalChars * sizeof(char));
	for (int s = 0; s < sentencesLength; s++) {
		for (int c = totalChars; c < totalChars + sentenceLengths[s]; c++) {
			chars[c] = sentences[s][c];
		}
		totalChars += sentenceLengths[s];
	}
	float*** data = (float***) malloc(totalChars * sizeof(float**));
	for (int c = 0; c < totalChars; c++) {
		data[c] = (float**) malloc((nn->length + 1) * sizeof(float*));
		for (int layer = 0; layer < nn->length; layer++) {
			data[c][layer] = (float*) malloc(nn->layers[layer]->input->length * sizeof(float));
		}
		data[c][nn->length] = (float*) malloc(nn->layers[nn->length - 1]->output->length * sizeof(float));
	}
	 // store (index - 1, index), to be shuffled the same way as data, to allow to recover the index of the previous character for data[c]
	int* shuffledIndexes = (int*) malloc(totalChars * sizeof(int));
	for (int i = 0; i < totalChars; i++) {
		shuffledIndexes[i] = i;
	}
	// deltas stores for each sentence s * char c in s, the deltas that reach the first layer.
	float** deltas = (float**) malloc(totalChars * sizeof(float*));
	for (int c = 0; c < totalChars; c++) {
		deltas[c] = (float*) malloc(nn->layers[0]->input->length * sizeof(float));
	}

	float* deltasPrime = (float*) malloc(ASCII_DATA_LENGTH * sizeof(float));

	do {
		error = 0;
		// generate data
		for (int c = 0; c < totalChars; c++) {
			// Set to 0 memory-part of input
			if (c == 0 || chars[c] == 0) {
				for (int i = 0; i < MEMORY_LENGTH; i++) {
					nn->layers[0]->input->data[ASCII_DATA_LENGTH + i] = 0;
				}
			}

			// Put char in char-part of input
			ascii2data(chars[c], nn->layers[0]->input->data);

			execNN(nn, NULL);

			// Copy nn layers data to nn->data[s][c] (input of layer L to data[s][c][L])
			for (int layer = 0; layer < nn->length; layer++) {
				memcpy(data[c][layer], nn->layers[layer]->input->data, nn->layers[layer]->input->length * sizeof(float));
			}

			memcpy(data[c][nn->length], nn->layers[nn->length - 1]->output->data, nn->layers[nn->length - 1]->output->length * sizeof(float));

			// Copy memory-part of output to memory-part of input
			memcpy(nn->layers[0]->input->data + ASCII_DATA_LENGTH, nn->layers[nn->length - 1]->output->data + ASCII_DATA_LENGTH, MEMORY_LENGTH * sizeof(float));
		}

		// Shuffle testbed
		float** auxCharPP;
		int auxInt;
		int randIndex;
		// for (int c = 0; c < totalChars - 1; c++) {
		// 	randIndex = (int) floor(((float) rand()) / ((float) RAND_MAX) * (totalChars - c) + c);
		// 	// auxCharPP = data[randIndex];
		// 	// data[randIndex] = data[c];
		// 	// data[c] = auxCharPP;
		// 	auxInt = shuffledIndexes[randIndex];
		// 	shuffledIndexes[randIndex] = shuffledIndexes[c];
		// 	shuffledIndexes[c] = auxInt;
		// }
		int shuffledIndex;
		int shuffledNextIndex;

		// backpropagation through the entire testbed: only char-part
		for (int c = 0; c < totalChars - 1 && elapsed < maxTime; c++) {
			shuffledIndex = shuffledIndexes[c];
			shuffledNextIndex = shuffledIndexes[c + 1];
			ascii2data(chars[shuffledNextIndex], charData);

			// Compute the error array into deltas1
			error += calculateError(data[shuffledIndex][nn->length], charData, ASCII_DATA_LENGTH);
			// int i = 7; // 37
			// if (elapsed > 10) {
				// for (int i = 0; i < ASCII_DATA_LENGTH; i++) {
					// printf("%d, %f\n", i, data[shuffledIndex][nn->length][i]);
				// }
				// printf("\n");
				// exit(0);
			// }
			// printf("37, %f\n", data[shuffledIndex][nn->length][37]);
			// error += data2ascii(data[shuffledIndex][nn->length]) != chars[shuffledNextIndex]
			// 	? 0.5
			// 	: 0;
			for (int i = 0; i < ASCII_DATA_LENGTH; i++) {
				deltasPrime[i] = data[shuffledIndex][nn->length][i] - charData[i];
			}
			// int i = 7; // 37
			// for (int i = 0; i < ASCII_DATA_LENGTH; i++) {
			// 	printf("%d, %f\t\t%f\n", i, data[shuffledIndex][nn->length][i], deltasPrime[i]);
			// }
			// exit(0);
			// printf("\n");
			/* deltasPrime[37] -> -0
			 * deltasPrime[51,52] -> +1
			 * deltasPrime[x] -> +0
			 * ?
			 */
			backpropagatePio(nn, data[shuffledIndex], alpha, deltasPrime, 0, deltas[c]);
			// int i = 51; // 37
			// for (int i = 0; i < ASCII_DATA_LENGTH; i++) {
				// printf("%d, %f\n", i, deltas[c][i]);
			// }
			// exit(0);

			// printf("%d\n-----\n", c);
			// for (int i = 0; i < nn->layers[0]->input->length; i++) {
			// 	printf("%f ", deltas[c][i]);
			// }
			// printf("\n");
			// Store deltas[c] where it should
			elapsed = time(NULL) - startTime;
			logElapsed(console, elapsed);
		}

		// // backpropagation through the entire testbed, only memory-part
		// for (int c = 1, s = 0, sc = 1; c < totalChars && elapsed < maxTime; c++) {
		// 	// error += calculatePioError(pio, sentences[s]);

		// 	if (sc == sentenceLengths[s]) {
		// 		sc = 0;
		// 		s++;
		// 	} else {
		// 		backpropagatePio(nn, data[shuffledIndexes[c - 1]], alpha, deltas[c], 1, NULL);
		// 		// Store deltas[c] where it should
		// 		elapsed = time(NULL) - startTime;
		// 		logElapsed(console, elapsed);
		// 	}
		// 	sc++;
		// }

		// error = error / totalChars;
		logError(console, error);
		logIncrementLoops(console);
		displayConsole(console);
		fflush(stdout);
	} while (elapsed < maxTime && error > maxError);

	// Free memory
	free(charData);
	free(shuffledIndexes);
	free(deltasPrime);
	free(sentenceLengths);
	for (int c = 0; c < totalChars; c++) {
		for (int layer = 0; layer < nn->length + 1; layer++) {
			free(data[c][layer]);
		}
		free(data[c]);
	}
	free(data);
	for (int c = 0; c < totalChars; c++) {
		free(deltas[c]);
	}
	free(deltas);

	return nn;
}
