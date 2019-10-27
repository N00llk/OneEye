#ifndef PARSEPSYHO_H
#define PARSEPSYHO_H
#include <QThread>

class ParsePsyho : public QThread
{
public:
    void run(void);
    void setPhoto(QString path);
    int getType(void);
    double getPredict(void);
private:
    double type = 0;
    double predict = 0;
    QString photo;
};

#endif // PARSEPSYHO_H
