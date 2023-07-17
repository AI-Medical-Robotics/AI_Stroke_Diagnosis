#include <QApplication>
#include <QWidget>
#include <QPushButton>
#include <QLabel>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    QWidget window;
    window.resize(250, 150);
    window.setWindowTitle("QT Example");

    QLabel label(&window);

    QPushButton button("Click me!", &window);
    button.setGeometry(QRect(QPoint(80, 80), QSize(100, 20)));

    QObject::connect(&button, &QPushButton::clicked, [&label]() {
        label.setText("Hello, World!");
        label.setGeometry(QRect(QPoint(80, 50), QSize(100, 20)));
    });

    window.show();

    return app.exec();
}