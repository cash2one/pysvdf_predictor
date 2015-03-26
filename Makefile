export CC  = ~/.jumbo/opt/gcc48/bin/gcc
export CXX = ~/.jumbo/opt/gcc48/bin/g++
export CFLAGS = -Wall -O3 -msse2 -fPIC

# specify tensor path
INSTALL_PATH= ../bin
BIN = svd_feature svd_feature_infer
OBJ = apex_svd_data.o apex_svd.o apex_reg_tree.o
.PHONY: clean all

all: $(BIN)
export LDFLAGS= -pthread -lm 

pysvdf_predictor.so: pysvdf_predictor.cc $(OBJ) apex_svd_data.h 
	$(CXX) -shared -fPIC $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc, $^)

svd_feature: svd_feature.cpp $(OBJ) apex_svd_data.h 
svd_feature_infer: svd_feature_infer.cpp $(OBJ) apex_svd_data.h 
apex_svd.o: apex_svd.cpp apex_svd.h apex_svd_model.h apex_svd_data.h solvers/*/*.h 
apex_svd_data.o: apex_svd_data.cpp apex_svd_data.h
apex_reg_tree.o: solvers/gbrt/apex_reg_tree.cpp solvers/gbrt/apex_reg_tree.h

$(BIN) : 
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

install:
	cp -f -r $(BIN)  $(INSTALL_PATH)

clean:
	$(RM) $(OBJ) $(BIN) *~
