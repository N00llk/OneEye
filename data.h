#ifndef DATA_H
#define DATA_H

#include <QDialog>
#include <QTimer>
#include "parsepsyho.h"

namespace Ui {
class data;
}

class data : public QDialog
{
    Q_OBJECT

public:
    explicit data(QWidget *parent = nullptr);
    void setPhoto(QString path);
    void updateStatus(int type);
    ~data();

private:
    Ui::data *ui;
    ParsePsyho *thread;
    int type;
    QTimer *timer;

private Q_SLOTS:
    void slotTimerAlarm();

protected:
    void closeEvent(QCloseEvent *);
};

#endif // DATA_H
