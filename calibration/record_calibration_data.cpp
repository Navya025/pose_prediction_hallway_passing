#include "kinect_video_display.h"

#include <k4a/k4a.hpp>
#include <opencv2/opencv.hpp>

#include <gtk/gtk.h>

//https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf

GtkWidget *window;
bool keepRunning = true;
k4a::device device;
VideoDisplay videoDisplay;

gboolean exit_program(GtkWidget *widget, GdkEvent *event, gpointer data) {
    //if menu closed exit program entirely. 
    keepRunning = false;
    //exit(0);
    return TRUE;
}

static void buildUI (GtkApplication *app, gpointer user_data){
    window = gtk_application_window_new(app);
    gtk_window_set_title (GTK_WINDOW (window), "Kinect" );
    gtk_window_set_default_size(GTK_WINDOW(window), 1920, 1080 );
    gtk_widget_add_events(window, GDK_KEY_PRESS_MASK);

    VideoDisplay* videoDisplay = (VideoDisplay*) user_data;

    videoDisplay->buildWidgets(window);

    g_signal_connect(window, "destroy", G_CALLBACK(exit_program), NULL);
    gtk_widget_show_all (window);
}


void *ourThread(void *data) {
    while(keepRunning) {
        k4a::capture capture;
        if (device.get_capture(&capture, std::chrono::milliseconds(K4A_WAIT_INFINITE))) {
            k4a::image colorImage =
                capture.get_color_image();
            cv::Mat colorCVMat = cv::Mat(
                    colorImage.get_height_pixels(),
                    colorImage.get_width_pixels(),
                    CV_8UC4);
            uint8_t *buffer = colorImage.get_buffer();
            memcpy(colorCVMat.data, buffer,
                colorImage.get_width_pixels() *
                colorImage.get_height_pixels() * 4);
            cv::Mat rgbMat;
            cv:cvtColor(colorCVMat,rgbMat, cv::COLOR_BGRA2RGB);
            videoDisplay.receiveFrame(rgbMat);
            //cv::imwrite("/home/justin/out.png", colorCVMat);
            // cv::imshow("A", colorCVMat);
            // cv::waitKey(1);
        }
    }
    return NULL;
}


int main(int argc, char **argv) {
    g_thread_init(NULL);

    GtkApplication *app = gtk_application_new ("org.gtk.example", G_APPLICATION_FLAGS_NONE);
    g_signal_connect (app, "activate", G_CALLBACK (buildUI), &videoDisplay);

    k4a_device_configuration_t device_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    device_config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    device_config.camera_fps = K4A_FRAMES_PER_SECOND_30;
    device_config.color_resolution = K4A_COLOR_RESOLUTION_1080P;
    device_config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    device_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;
    device_config.synchronized_images_only = true;

    device = k4a::device::open(0);
    device.start_cameras(&device_config);

    // k4a::calibration sensor_calibration =
    //     device.get_calibration(
    //         device_config.depth_mode, device_config.color_resolution);

    pthread_t videoThread;
    pthread_create(&videoThread, NULL, ourThread, NULL);


    int status = g_application_run (G_APPLICATION (app), 0, argv);
    g_object_unref (app);

    return 0;
}