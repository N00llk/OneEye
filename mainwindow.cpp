#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    wind = new class data(this);
    wind->hide();
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_loadPhoto_clicked()
{
    QString str = QFileDialog::getOpenFileName(nullptr, "Open Dialog", "", "*.jpg");
    if(str.size())
    {
        hide();
        wind->setPhoto(str);
        wind->show();
    }
}

void MainWindow::on_createPhoto_clicked()
{

}
