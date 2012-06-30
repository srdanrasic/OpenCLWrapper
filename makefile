UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
LIBS = -lpng -framework OpenCL
endif
ifeq ($(UNAME), Linux)
LIBS = -lOpenCL -lpng
endif

exec_cmd = ./bin/seminar ./resources/test_image.png 3

seminar:
	@echo "Compiling example..."
	@g++ src/*.cpp src/oclw/*.cpp -O3 -fopenmp -msse3 ${LIBS} -o bin/seminar
	
	@echo "Successfully completed!"
	@echo "To try, execute: make run"
	
run:
	@printf "Executing: ${exec_cmd}\n\n"
	@${exec_cmd}
	@echo
