/***************************************************************************
 * 
 * Copyright (c) 2015 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/
 
 
 
/**
 * @file pysvdf_predictor.cc
 * @author gusimiu(com@baidu.com)
 * @date 2015/03/26 09:28:01
 * @brief 
 *  
 **/
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include </home/users/gusimiu/.jumbo/include/python2.7/Python.h> //包含python的头文件

#include <ctime>
#include <cstring>
#include <climits>
#include "apex_svd.h"
#include "apex-utils/apex_task.h"
#include "apex-utils/apex_utils.h"
#include "apex-utils/apex_config.h"
#include "apex-tensor/apex_random.h"

using namespace apex_svd;

struct SVDFeature_Model_t {
    SVDTypeParam _mtype;
    ISVDTrainer  *_predictor;
};

SVDFeature_Model_t g_model;

static PyObject *wrapper_load(PyObject *self, PyObject *args) 
{
    const char* filename = PyString_AsString(PyTuple_GetItem(args, 0));
    fprintf(stderr, "prepare to load model: [%s]", filename);

    FILE* model_file = fopen(filename, "rb");
    fread( &g_model._mtype, sizeof(SVDTypeParam), 1, model_file );
    g_model._predictor = create_svd_trainer( g_model._mtype );
    g_model._predictor->load_model( model_file );
    fclose(model_file);
    g_model._predictor->init_trainer();
    return Py_BuildValue("i", 0);//把c的返回值n转换成python的对象
}

static PyObject *wrapper_predict(PyObject *self, PyObject *args) 
{
    PyObject* global_features = PyTuple_GetItem(args, 0);
    PyObject* user_features = PyTuple_GetItem(args, 1);
    PyObject* item_features = PyTuple_GetItem(args, 2);

    SVDFeatureCSR::Elem e;
    e.num_global = (int)PyList_Size(global_features);
    e.num_ufactor = (int)PyList_Size(user_features);
    e.num_ifactor = (int)PyList_Size(item_features);
    e.alloc_space();
    for (int i=0; i<e.num_global; ++i) {
        PyObject* tup = PyList_GetItem(global_features, i);
        e.index_global[i] = PyInt_AsLong( PyTuple_GetItem(tup, 0) );
        e.value_global[i] = PyFloat_AsDouble( PyTuple_GetItem(tup, 1) );
        //fprintf(stderr, "G %d:%f\n", e.index_global[i], e.value_global[i]); 
    }
    for (int i=0; i<e.num_ufactor; ++i) {
        PyObject* tup = PyList_GetItem(user_features, i);
        e.index_ufactor[i] = PyInt_AsLong( PyTuple_GetItem(tup, 0) );
        e.value_ufactor[i] = PyFloat_AsDouble( PyTuple_GetItem(tup, 1) );
        //fprintf(stderr, "U %d:%f\n", e.index_ufactor[i], e.value_ufactor[i]); 
    }
    for (int i=0; i<e.num_ifactor; ++i) {
        PyObject* tup = PyList_GetItem(item_features, i);
        e.index_ifactor[i] = PyInt_AsLong( PyTuple_GetItem(tup, 0) );
        e.value_ifactor[i] = PyFloat_AsDouble( PyTuple_GetItem(tup, 1) );
        //fprintf(stderr, "I %d:%f\n", e.index_ifactor[i], e.value_ifactor[i]); 
    }

    float out = g_model._predictor->predict(e);
    e.free_space();
    return Py_BuildValue("f", out);//把c的返回值n转换成python的对象
}


// 3 方法列表
static PyMethodDef CKVDictFunc[] = {
    // 读取文件到词典，可选是否是内存结构
    { "load", wrapper_load, METH_VARARGS, "load models.\n\t\tpysvdf_predictor.load(model_name)"},
    // 查找信息
    { "predict", wrapper_predict, METH_VARARGS, "predict:\n\t\tpysvdf_predictor.predict([], [(uidx, uval), (uidx, uval), ...], [(iidx, ival), (iidx, ival), ...])"},
    { NULL, NULL, 0, NULL }
};
// 4 模块初始化方法
PyMODINIT_FUNC initpysvdf_predictor(void) {
    //初始模块，把CKVDictFunc初始到c_kvdict中
    PyObject *m = Py_InitModule("pysvdf_predictor", CKVDictFunc);
    if (m == NULL) {
        fprintf(stderr, "Init module failed!\n");
        return;
    }
}

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
