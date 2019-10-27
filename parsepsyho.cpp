#include "parsepsyho.h"
#include <Python.h>
#include <QDebug>

void ParsePsyho::run(void)
{
    Py_Initialize();

    Py_Finalize();
}

void ParsePsyho::setPhoto(QString path)
{
    photo = path;
}

int ParsePsyho::getType()
{
    return (int)type;
}

double ParsePsyho::getPredict()
{
    return predict;
}

