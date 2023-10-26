#ifndef KINECT_VIDEO_DISPLAY_H
#define KINECT_VIDEO_DISPLAY_H

#include <opencv2/opencv.hpp>

#include <gtk/gtk.h>

#include <string>


class VideoDisplay {
public:
    //Display rows & columns
    VideoDisplay(int cols = 1920, int rows = 1080);
    ~VideoDisplay();
    void receiveFrame(cv::Mat &skp);
    void buildWidgets(GtkWidget *container);
    static gboolean drawCallback (GtkWidget *widget, cairo_t *cr, gpointer data);
    gboolean doDraw(cairo_t *cr);
protected:
    bool _initialized;
    int _rows, _cols;
    size_t _bytes;
    GtkWidget *_drawingArea;
    unsigned char *_buf;
};

#endif