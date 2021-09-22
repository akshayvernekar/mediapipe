// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe auto_frame_graph.
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <thread>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/examples/desktop/auto_frame/autoframe_messages.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/yuv_image.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/image_frame_util.h"
// #include "mediapipe/examples/desktop/auto_frame/autoframe_messages.pb.h"

constexpr char kInputStream[] = "input_video";
constexpr char kInputSelectStream[] = "select";
constexpr char kInputDetectionStream[] = "prev_detection";
constexpr char kOutputStream[] = "output_video";
constexpr char kOutputStream_Roi[] = "curr_detection";

constexpr char kWindowName[] = "MediaPipe";

#ifndef VID_WIDTH
#define VID_WIDTH 640
#endif

#ifndef VID_HEIGHT
#define VID_HEIGHT 480
#endif

#define DEFAULT_VIDEO_IN 0
#define DEFAULT_VIDEO_OUT "/dev/video6"

#define AUTO_FRAME_GRAPH \
    "mediapipe/examples/desktop/auto_frame/graphs/combined_graph.pbtxt"

// std::string gPrevDetection;
mediapipe::CombinedDetection gPrevDetection;
bool gGrabFrames = false;
mediapipe::CombinedDetection::OP_TYPE gOperation =
    mediapipe::CombinedDetection::NOOP;

ABSL_FLAG(std::string, input_video, "",
          "Camera input device. "
          "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video, "",
          "v4l2output device "
          "If not provided, show result in a window.");

void UserInputReader() {
    char user_input;
    std::cout << "-----------Zylux IR Receiver-----------" << std::endl;
    while (gGrabFrames) {
        std::cout << "0: No Op" << std::endl;
        std::cout << "1: Autoframe" << std::endl;
        std::cout << "2: Gesture" << std::endl;
        std::cout << "3: Zoom In" << std::endl;
        std::cout << "4: Zoom Out" << std::endl;
        std::cout << "q: quit" << std::endl;
        std::cout << "Enter your choice:" << std::endl;

        std::cin >> user_input;
        std::cout << "user entered: " << user_input << std::endl;

        if (user_input == 'q') {
            std::cout << ">>>>> Quitting program" << std::endl;
            gGrabFrames = false;
        } else if (user_input == '0') {
            std::cout << ">>>>> Turning on NOOP" << std::endl;
            gOperation = mediapipe::CombinedDetection::NOOP;
        } else if (user_input == '1') {
            std::cout << ">>>>> Turning on AUTOFRAME" << std::endl;
            gOperation = mediapipe::CombinedDetection::AUTO_FRAME;
        } else if (user_input == '2') {
            std::cout << ">>>>> Turning on GESTURE_RECOG" << std::endl;
            gOperation = mediapipe::CombinedDetection::GESTURE_RECOG;
        } else if (user_input == '3') {
            std::cout << ">>>>> Zooming In" << std::endl;
            gOperation = mediapipe::CombinedDetection::ZOOM_IN;
        } else if (user_input == '4') {
            std::cout << ">>>>> Zooming Out" << std::endl;
            gOperation = mediapipe::CombinedDetection::ZOOM_OUT;
        }
    }
}
absl::Status roi_callback(mediapipe::Packet packet) {
    gPrevDetection = packet.Get<mediapipe::CombinedDetection>();
    return absl::OkStatus();
}

absl::Status RunMPPGraph() {
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
        AUTO_FRAME_GRAPH, &calculator_graph_config_contents));
    LOG(INFO) << "Get calculator auto_frame_graph config contents: "
              << calculator_graph_config_contents;
    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
            calculator_graph_config_contents);

    LOG(INFO) << "Initialize the auto_frame_graph.";
    mediapipe::CalculatorGraph auto_frame_graph;
    MP_RETURN_IF_ERROR(auto_frame_graph.Initialize(config));

    cv::VideoCapture capture;
    const bool load_video = !absl::GetFlag(FLAGS_input_video).empty();

    LOG(INFO) << "Initialize the camera or load the video.";
    if (load_video) {
        capture.open(absl::GetFlag(FLAGS_input_video));
    } else {
        capture.open(DEFAULT_VIDEO_IN);
    }

    RET_CHECK(capture.isOpened());

#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 30);
#endif

    // ----------------------------------v4l2 related code
    const bool write_to_v4l2 = !absl::GetFlag(FLAGS_output_video).empty();

    int output;
    // size_t framesize = VID_WIDTH * VID_HEIGHT * 3/2;
    size_t framesize = VID_WIDTH * VID_HEIGHT * 2;
    auto start = std::chrono::high_resolution_clock::now();

    if (write_to_v4l2) {
        // open output device
        output = open(absl::GetFlag(FLAGS_output_video).c_str(), O_RDWR);
        if (output < 0) {
            std::cerr << "ERROR: could not open output device!\n"
                      << strerror(errno);
            return absl::InvalidArgumentError("Invalid output device");
        }
        // configure params for output device
        struct v4l2_format vid_format;
        memset(&vid_format, 0, sizeof(vid_format));
        vid_format.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;

        if (ioctl(output, VIDIOC_G_FMT, &vid_format) < 0) {
            std::cerr << "ERROR: unable to get video format!\n"
                      << strerror(errno);
            return absl::InvalidArgumentError("unable to get videoformat");
        }

        vid_format.fmt.pix.width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
        vid_format.fmt.pix.height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);

        // NOTE: change this according to below filters...
        // Chose one from the supported formats on Chrome:
        // - V4L2_PIX_FMT_YUV420,
        // - V4L2_PIX_FMT_Y16,
        // - V4L2_PIX_FMT_Z16,
        // - V4L2_PIX_FMT_INVZ,
        // - V4L2_PIX_FMT_YUYV,
        // - V4L2_PIX_FMT_RGB24,
        // - V4L2_PIX_FMT_MJPEG,
        // - V4L2_PIX_FMT_JPEG
        // vid_format.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
        vid_format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
        // vid_format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUV420;

        vid_format.fmt.pix.sizeimage = framesize;
        vid_format.fmt.pix.field = V4L2_FIELD_NONE;
        vid_format.fmt.pix.colorspace = V4L2_COLORSPACE_SRGB;

        if (ioctl(output, VIDIOC_S_FMT, &vid_format) < 0) {
            std::cerr << "ERROR: unable to set video format!\n"
                      << strerror(errno);
            return absl::InvalidArgumentError("unable to set videoformat");
        }
        // ----------------------------------

    } else {
        cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
    }

    LOG(INFO) << "Start running the calculator auto_frame_graph.";
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller1,
                     auto_frame_graph.AddOutputStreamPoller(kOutputStream));

    auto_frame_graph.ObserveOutputStream(kOutputStream_Roi, roi_callback);
    MP_RETURN_IF_ERROR(auto_frame_graph.StartRun({}));

    LOG(INFO) << "Start grabbing and processing frames.";
    gGrabFrames = true;

    // starting user input thread
    std::thread userInputThread(UserInputReader);
    // userInputThread.join();

    // initializing prev_detetction packet
    gPrevDetection.set_type(mediapipe::CombinedDetection::BBOX);
    auto rect = absl::make_unique<mediapipe::Rect>();
    rect->set_x_center(VID_WIDTH / 2);
    rect->set_y_center(VID_HEIGHT / 2);
    rect->set_width(VID_WIDTH);
    rect->set_height(VID_HEIGHT);
    gPrevDetection.set_allocated_bbox(rect.release());

    while (gGrabFrames) {
        auto start = std::chrono::high_resolution_clock::now();
        // Capture opencv camera or video frame.
        cv::Mat camera_frame_raw;
        capture >> camera_frame_raw;

        if (camera_frame_raw.empty()) {
            LOG(INFO) << "Empty frame, end of video reached.";
            break;
        }
        cv::Mat camera_frame;
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);

        // Wrap Mat into an ImageFrame.
        auto input_frame1 = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary);

        cv::Mat input_frame_mat =
            mediapipe::formats::MatView(input_frame1.get());
        camera_frame.copyTo(input_frame_mat);

        size_t frame_timestamp_us =
            (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;

        MP_RETURN_IF_ERROR(auto_frame_graph.AddPacketToInputStream(
            kInputStream, mediapipe::Adopt(input_frame1.release())
                              .At(mediapipe::Timestamp(frame_timestamp_us))));

        MP_RETURN_IF_ERROR(auto_frame_graph.AddPacketToInputStream(
            kInputSelectStream,
            mediapipe::MakePacket<mediapipe::CombinedDetection::OP_TYPE>(
                gOperation)
                .At(mediapipe::Timestamp(frame_timestamp_us))));

        MP_RETURN_IF_ERROR(auto_frame_graph.AddPacketToInputStream(
            kInputDetectionStream,
            mediapipe::MakePacket<mediapipe::CombinedDetection>(gPrevDetection)
                .At(mediapipe::Timestamp(frame_timestamp_us))));

        // Get the auto_frame_graph result packet, or stop if that fails.
        mediapipe::Packet packet;
        if (!poller1.Next(&packet)) break;

        auto& output_frame = packet.Get<mediapipe::ImageFrame>();

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        // LOG(ERROR) << "time taken to get Frame: " << duration.count();

        // Convert back to opencv for display or saving.
        cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
        static int once = 0;
        if (write_to_v4l2) {
            mediapipe::YUVImage yuv_image;
            mediapipe::image_frame_util::ImageFrameToYUV2Image(output_frame,
                                                               &yuv_image);
            // write frames to v4l2 loopback device
            int written = write(output, yuv_image.data(0), framesize);

            // int written = write(output, output_frame_mat.data, framesize);
            LOG(ERROR) << "written bytes : " << written;
            if (written < 0) {
                std::cerr << "ERROR: could not write to output device!\n";
                close(output);
                break;
            }

            if (once == 0) {
                int fd = open("i420image.yuv", O_CREAT | O_WRONLY);
                int size = yuv_image.width() * yuv_image.height() * 1.5;
                if (fd) {
                    written = write(fd, yuv_image.data(0), size);
                    LOG(ERROR) << "Wrote file i420image.yuv: " << written;
                    close(fd);
                }

                mediapipe::YUVImage yuv_image2;
                mediapipe::image_frame_util::ImageFrameToYUV2Image(output_frame,
                                                                   &yuv_image2);
                size = yuv_image2.width() * yuv_image2.height() * 2;
                fd = open("yuy2image.yuv", O_CREAT | O_WRONLY);
                if (fd) {
                    written = write(fd, yuv_image2.data(0), size);
                    LOG(ERROR) << "Wrote file yuy2image.yuv: " << written;
                    close(fd);
                }
                once += 1;
            }
        } else {
            cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
            cv::imshow(kWindowName, output_frame_mat);

            // Press any key to exit.
            const int pressed_key = cv::waitKey(5);
            // if (pressed_key >= 0 && pressed_key != 255)
            //     gGrabFrames = false;
        }
    }
    if (output) close(output);

    LOG(ERROR) << "Shutting down.";
    MP_RETURN_IF_ERROR(auto_frame_graph.CloseInputStream(kInputStream));
    return auto_frame_graph.WaitUntilDone();
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    absl::ParseCommandLine(argc, argv);
    absl::Status run_status = RunMPPGraph();
    if (!run_status.ok()) {
        LOG(ERROR) << "Failed to run the auto_frame_graph: "
                   << run_status.message();
        return EXIT_FAILURE;
    } else {
        LOG(INFO) << "Success!";
    }
    return EXIT_SUCCESS;
}
