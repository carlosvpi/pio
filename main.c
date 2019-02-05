#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
// #include "lib/util/util.h"
// #include "lib/types/list.c"
// #include "lib/types/tree.c"
#include "lib/nn/nn.h"
#include "lib/nn/nn.c"
#include "lib/pio/conversors.c"
#include "lib/pio/pionn.c"


// !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_

void copySentence(char* string, char** sentences, int i) {
	int length = getSentenceLength(string);
	sentences[i] = (char*) malloc((length + 1) * sizeof(char));
	memcpy(sentences[i], string, length * sizeof(char));
	sentences[i][length] = 0;
}

int main(void) {
	printf("\n");
	printf("#### PIO ####\n");
	printf("-------------\n");

	float* output;

	// NN pio = loadNN("nns/pio");
	NN pio = NULL;

	if (pio == NULL) {
		pio = newPio();
		saveNN(pio, "nns/pio");
	}

	char** sentences = (char**) malloc(4 * sizeof(char*));
	// copySentence("Le", sentences, 0);
	copySentence("Learning to tweet", sentences, 0);
	// copySentence("Learning to tweet", sentences, 0);
	// copySentence("You are awesome", sentences, 1);
	// copySentence("Computing forever", sentences, 2);
	// copySentence("Literally is not figuratively", sentences, 3);

	TRAIN_OPTIONS trainOptions = newTrainOptions(900, 0.01, 0.015);
	CONSOLE console = newConsole();
	startConsole();
	trainPio(pio, sentences, 1, trainOptions, console);
	endConsole();

	printf("%s\n", generate(pio, 'L'));

	printf("\nEND\n");
}
