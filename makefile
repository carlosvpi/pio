build:
	gcc main.c -framework OpenCL && ./a.out

test:
	gcc test.c -framework OpenCL && ./a.out

build++:
	g++ main.c -framework OpenCL && ./a.out
