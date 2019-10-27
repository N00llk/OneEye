#include "data.h"
#include "ui_data.h"
#include <QPixmap>

data::data(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::data)
{
    ui->setupUi(this);
    thread = new ParsePsyho;
}

data::~data()
{
    delete ui;
}

void data::closeEvent(QCloseEvent *)
{
    parentWidget()->show();
}

void data::setPhoto(QString path)
{
    timer = new QTimer();
    connect(timer, SIGNAL(timeout()), this, SLOT(slotTimerAlarm()));
    timer->start(1000);
    QGraphicsScene * scen = new QGraphicsScene();
    QPixmap * pix = new QPixmap();
    pix->load(path);
    scen->addPixmap(*pix);
    ui->graphicsView_4->setScene(scen);

    thread->setPhoto(path);
    thread->run();

}

void data::updateStatus(int type)
{

    QFile file("./" + QString::number(type) + ".txt");
    QString str;
    while(!file.atEnd())
    {
        str=str+file.readLine();
    }
    ui->plainTextEdit->setPlainText(str);
    switch(type)
    {
        case 1:
            ui->label->setText("Психотип: Паранойяльный психотип (Целеустремленный)");
            break;
        case 2:
            ui->label->setText("Психотип: Истероидный психотип (Демонстративный)");
            break;
        case 3:
            ui->label->setText("Психотип: Эпилептоидный- застревающий психотип");
            break;
        case 4:
            ui->label->setText("Психотип: Эпилептоидный-возбудимый психотип");
            break;
        case 5:
            ui->label->setText("Психотип: Шизоидный психотип (Странный)");
            break;
        case 6:
            ui->label->setText("Психотип: Гипертимный психотип (Жизнерадостный)");
            break;
        case 7:
            ui->label->setText("Психотип: Эмотивный психотип (Чувствительный)");
            break;
        case 8:
            ui->label->setText("Психотип: Тревожный психотип (Боязливый)");
            break;
    }
}

void data::slotTimerAlarm()
{
    if(thread->isFinished())
    {
        updateStatus(thread->getType());
        timer->stop();
    }
}
