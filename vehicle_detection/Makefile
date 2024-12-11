PROJECT   =    aeb_od.elf 
OBJ       :=   main.o dputils.o dpu_yolov5n_od.elf
CXX       :=   g++
CC        :=   gcc

# linking libraries of OpenCV
LDFLAGS   =   -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_imgcodecs -lopencv_videoio -lopencv_freetype -lxml2

# linking libraries of DNNDK 
LDFLAGS   +=  -lhineon -ln2cube -lpthread -lm -ldl

CUR_DIR   =   $(shell pwd)
SRC       =   $(CUR_DIR)/src

MODEL     =   $(CUR_DIR)/model
VPATH     =   $(SRC)



CFLAGS    :=  -lglog -Wall -Wpointer-arith -std=c++11 -ffast-math `pkg-config --cflags --libs opencv` -fpermissive -O3 -I$(CUR_DIR)/src/

all: $(BUILD) $(PROJECT)
 
$(PROJECT): $(OBJ)
	$(CXX) $(CFLAGS) $^ -o $(CUR_DIR)/$@ $(LDFLAGS)
 
%.o : %.cc
	$(CXX) -c $(CFLAGS) $< -o $@

%.o : %.cpp
	$(CXX) -c $(CFLAGS) $< -o $@

%.elf : 
	cp $(MODEL)/$@ $@ 

clean:
	$(RM) $(OBJ)
	$(RM) $(PROJECT)
