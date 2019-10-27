QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11
CONFIG += no_keywords

DEFINES += QT_DEPRECATED_WARNINGS

SOURCES += \
    data.cpp \
    main.cpp \
    mainwindow.cpp \
    parsepsyho.cpp

HEADERS += \
    data.h \
    mainwindow.h \
    parsepsyho.h

FORMS += \
    data.ui \
    mainwindow.ui

unix|win32: LIBS += -L$$PWD/Python/libs/ -lpython37

INCLUDEPATH += $$PWD/Python/include
DEPENDPATH += $$PWD/Python/include

win32:!win32-g++: PRE_TARGETDEPS += $$PWD/Python/libs/python37.lib
else:unix|win32-g++: PRE_TARGETDEPS += $$PWD/Python/libs/libpython37.a
