#ifndef _NN_H_
#define _NN_H_

#define ALPHA 0.1

typedef struct _array* ARRAY;
typedef struct _layer* LAYER;
typedef struct _nn* NN;
typedef struct _train_options* TRAIN_OPTIONS;
typedef struct _console* CONSOLE;

ARRAY newArray (int length);
void cleanArray (ARRAY array);
void freeArray (ARRAY array);
void assignData (ARRAY array, float* data);
LAYER newLayer (ARRAY input, int length);
void freeLayer (LAYER layer);
ARRAY execLayer (LAYER layer);
NN newNN (ARRAY input, int length, int* lengths);
void freeNN(NN nn);
NN execNN (NN nn, float* inputData);
NN backpropagate (NN nn, float* expected, float alpha);
TRAIN_OPTIONS newTrainOptions (int maxTime, float maxError, float alpha);
CONSOLE newConsole ();
void freeConsole (CONSOLE console);
void logElapsed (CONSOLE console, int elapsed);
void logError (CONSOLE console, float error);
void startConsole ();
void endConsole ();
void displayConsole (CONSOLE console);
void freeTrainOptions (TRAIN_OPTIONS trainOptions);
NN train(NN nn, float** inputs, float** outputs, int testBedSize, TRAIN_OPTIONS trainOptions, CONSOLE console);
NN loadNN(char* filename);
void saveNN(NN nn, char* filename);
float* getOutput (NN nn);
NN loadNNRaw(FILE* file);
void saveNNRaw(NN nn, FILE* file);

#endif