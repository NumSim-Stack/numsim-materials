#ifndef NUMSIM_MATERIALS_QCUSTOMPLOT_BACKEND_H
#define NUMSIM_MATERIALS_QCUSTOMPLOT_BACKEND_H

#include "numsim-materials/postprocessing/plot_backend.h"
#include <qcustomplot.h>
#include <QApplication>
#include <stdexcept>
#include <QThread>
#include <string>
#include <vector>

namespace numsim::materials {

/// QCustomPlot implementation of plot_backend.
/// Manages a QApplication + QCustomPlot widget.
/// Calls QApplication::processEvents() on refresh() for inline event processing.
class qcustomplot_backend : public plot_backend {
public:
  /// Construct a QCustomPlot backend.
  /// A QApplication must already exist (created by the application, not the library).
  /// If no QApplication exists, throws std::runtime_error.
  qcustomplot_backend(const std::string& title = "numsim-materials") {
    if (!QApplication::instance())
      throw std::runtime_error(
          "qcustomplot_backend: QApplication must be created before constructing the backend");
    m_plot = new QCustomPlot();
    m_plot->setWindowTitle(QString::fromStdString(title));
    m_plot->setMinimumSize(800, 500);
    m_plot->legend->setVisible(true);
    m_plot->xAxis->setLabel("x");
    m_plot->yAxis->setLabel("y");
    m_plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);
  }

  ~qcustomplot_backend() override {
    delete m_plot;
  }

  int add_series(const std::string& name) override {
    int idx = static_cast<int>(m_graphs.size());
    auto* graph = m_plot->addGraph();
    graph->setName(QString::fromStdString(name));
    graph->setPen(QPen(default_color(idx), 2));
    m_graphs.push_back(graph);
    return idx;
  }

  void append(int series, double x, double y) override {
    if (series >= 0 && series < static_cast<int>(m_graphs.size()))
      m_graphs[series]->addData(x, y);
  }

  void show() override {
    m_plot->setWindowFlags(Qt::Window);
    m_plot->show();
    m_plot->raise();
    m_plot->activateWindow();
    if (QApplication::instance())
      QApplication::processEvents();
  }

  void refresh() override {
    m_plot->rescaleAxes();
    m_plot->replot();
    if (QApplication::instance()) {
      QApplication::processEvents();
      if (m_frame_delay_ms > 0)
        QThread::msleep(m_frame_delay_ms);
    }
  }

  /// Set delay between frames in milliseconds (default 0).
  /// Use ~20-50ms for visible animation.
  void set_frame_delay(int ms) noexcept { m_frame_delay_ms = ms; }

  /// Save the current plot to a PNG file.
  bool save_png(const std::string& filename, int width = 800, int height = 500) {
    return m_plot->savePng(QString::fromStdString(filename), width, height);
  }

private:
  static QColor default_color(int index) {
    static const QColor colors[] = {
      Qt::blue, Qt::red, Qt::darkGreen, Qt::magenta,
      Qt::darkCyan, Qt::darkYellow, Qt::darkRed, Qt::darkBlue
    };
    return colors[index % 8];
  }

  QCustomPlot* m_plot{nullptr};
  std::vector<QCPGraph*> m_graphs;
  int m_frame_delay_ms{0};
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_QCUSTOMPLOT_BACKEND_H
