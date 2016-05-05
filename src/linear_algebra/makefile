sgemm_test: sgemm.o
	g++ sgemm.o ../components_test.cpp -o sgemm_test

sgemm.o: sgemm_generate
	./sgemm_generate -g sgemm -o . target=host
	cp sgemm.h ../../include
sgemm_generate: sgemm_generator.cpp
	g++ sgemm_generator.cpp ../../tools/GenGen.cpp -g -std=c++11 -fno-rtti -I ../../include/halide -L ../../bin -lHalide -o sgemm_generate

clean:
	rm sgemm_test sgemm.o sgemm.h sgemm_generate
	rm -r *dSYM
