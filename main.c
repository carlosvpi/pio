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

	//PIO pio = newPio(loadNN("nns/pio"));
	PIO pio = NULL;

	if (pio == NULL) {
		pio = newPio(NULL);
		saveNN(pio->nn, "nns/pio");
	}

	char** sentences = (char**) malloc(4 * sizeof(char*));
	copySentence("Learning to tweet", sentences, 0);
	copySentence("You are awesome", sentences, 1);
	copySentence("Computing forever", sentences, 2);
	copySentence("Literally is not figuratively", sentences, 3);

	CONSOLE console = newConsole();
	startConsole();
	trainPio(pio, 0.15, sentences, 1, NULL, console);
	endConsole();

	printf("%s\n", generate(pio));

	printf("\nEND\n");

	// for (int i = 0; i < 1; i++) {
	// 	match = newMatch();
	// 	alpha = 20 / sqrt(i + 10);
	// 	blackWon = trainPlayingMatch(match, nn, alpha);
	// 	// printf("\n\n\n\n\n\n---------------------------------------\n");
	// 	printf("match: %d, alpha: %.3f, ", i, alpha);
	// 	if (blackWon == TRUE) {
	// 		printf(KBLU "Black" KWHT " won, ");
	// 	} else {
	// 		printf(KYEL "White" KWHT " won, ");
	// 	}
	// 	saveMatch(match, "sgf/m1.sgf");
	// 	saveNN(nn, "nns/mk");
	// }
	// // printf("\n");
	// // saveMatch(match, "sgf/m1.sgf");

	// // BOOL run(&next, current, previous, workboard1, workboard2, nn, stone) {



	// // freeNN(nn);

	// // gocl();

	// // printf("List: %d\n", list);
	// // printf("Items: %d\n", list->items);
	// // printf("Items[0]: %d\n", list->items[0]);
	// // printf("Items[0][0]: %d\n", ((char*) list->items[0])[0]);
	// // printf("Items[0][1]: %d\n", ((char*) list->items[0])[1]);
}
