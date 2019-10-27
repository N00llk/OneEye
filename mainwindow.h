#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "data.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private Q_SLOTS:
    void on_loadPhoto_clicked();
    void on_createPhoto_clicked();

private:
    Ui::MainWindow *ui;
    class data *wind;
};
#endif // MAINWINDOW_H
