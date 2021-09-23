load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_proto_library(
    name = "detections_to_autoframe_roi_calculator_proto",
    srcs = ["detections_to_autoframe_roi_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util:color_proto",
        "//mediapipe/util:render_data_proto",
    ],
)

cc_library(
    name = "detections_to_autoframe_roi_calculator",
    srcs = [
        "detections_to_autoframe_roi_calculator.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":detections_to_autoframe_roi_calculator_cc_proto",
        "//mediapipe/examples/desktop/auto_frame:autoframe_messages_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/types:optional",
    ],
    alwayslink = 1,
)

cc_library(
    name = "hand_gesture_recognition_calculator",
    srcs = ["hand_gesture_recognition_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/examples/desktop/auto_frame:autoframe_messages_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:ret_check",
    ],
    alwayslink = 1,
)


cc_library(
    name = "put_text_on_image_calculator",
    srcs = ["put_text_on_image_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "detections_to_imageframe_calculator",
    srcs = ["detections_to_imageframe_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/examples/desktop/auto_frame:autoframe_messages_cc_proto",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "image_demux_calculator",
    srcs = ["image_demux_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/examples/desktop/auto_frame:autoframe_messages_cc_proto",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "detection_demux_calculator",
    srcs = ["detection_demux_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/examples/desktop/auto_frame:autoframe_messages_cc_proto",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "nooperation_calculator",
    srcs = ["nooperation_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/examples/desktop/auto_frame:autoframe_messages_cc_proto",
    ],
    alwayslink = 1,
)

cc_library(
    name = "zoomin_calculator",
    srcs = ["zoomin_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/examples/desktop/auto_frame:autoframe_messages_cc_proto",
    ],
    alwayslink = 1,
)


cc_library(
    name = "zoomout_calculator",
    srcs = ["zoomout_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/examples/desktop/auto_frame:autoframe_messages_cc_proto",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mux_calculator",
    srcs = ["mux_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)