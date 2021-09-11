# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library")

proto_library(
    name = "autoframe_messages_proto",
    srcs = ["autoframe_messages.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/formats:rect_proto"
    ],
)

mediapipe_cc_proto_library(
    name = "autoframe_messages_cc_proto",
    srcs = ["autoframe_messages.proto"],
    cc_deps = ["//mediapipe/framework/formats:rect_cc_proto"],
    visibility = [
        "//mediapipe/examples:__subpackages__",
    ],
    deps = [":autoframe_messages_proto"],
)

cc_binary(
    name = "autoframe_cpu",
    srcs = ["zylux_camera.cc"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "//mediapipe/examples/desktop/auto_frame:autoframe_messages_cc_proto",
        "//mediapipe/framework:calculator_framework",        
        "//mediapipe/framework/formats:image_frame",        
        "//mediapipe/framework/formats:yuv_image", 
        "//mediapipe/util:image_frame_util",    
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/calculators/core:mux_calculator",
        "//mediapipe/calculators/core:immediate_mux_calculator",
        "//mediapipe/examples/desktop/auto_frame/calculators:demux_calculator",
        "//mediapipe/examples/desktop/auto_frame/subgraphs:auto_frame_graph",
        "//mediapipe/examples/desktop/auto_frame/subgraphs:hand_gesture_recognition_live",
        "//mediapipe/framework/stream_handler:sync_set_input_stream_handler",
        "//mediapipe/examples/desktop/auto_frame/calculators:detections_to_imageframe_calculator",
    ],
)
