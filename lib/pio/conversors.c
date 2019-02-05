#define ASCII_DATA_LENGTH 64

float* newAsciiData () {
	return (float*) calloc(ASCII_DATA_LENGTH, sizeof(float));
}

void freeAsciiData (float* asciiArray) {
	free(asciiArray);
}

float* ascii2data (char code, float* data) {
	if (code < 32) {
		code = 64;
	} else if (code > 96 && code <= 122) {
		code -= 32;
	} else if (code > 122) {
		code = 64;
	}
	code -= 32;
	float* src = data != NULL ? data : (float*) malloc(ASCII_DATA_LENGTH * sizeof(float));
	for (int i = 0; i < ASCII_DATA_LENGTH; i++) {
		src[i] = 0;
	}
	src[code] = 1;
	return src;
}

char data2ascii (float* data) {
	char maxIndex = 0;
	for (int i = 0; i < ASCII_DATA_LENGTH; i++) {
		if (data[i] > data[(int) maxIndex]) {
			maxIndex = i;
		}
	}
	maxIndex += 32;
	if (maxIndex >= 65) {
		maxIndex += 32;
	}
	return maxIndex;
}