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

cc_binary(
    name = "autoframe_cpu",
    srcs = ["zylux_camera.cc"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "//mediapipe/framework:calculator_framework",        
        "//mediapipe/framework/formats:image_frame",        
        "//mediapipe/framework/formats:yuv_image", 
        "//mediapipe/util:image_frame_util",    
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/calculators/core:mux_calculator",
        "//mediapipe/examples/desktop/auto_frame/calculators:demux_calculator",
        "//mediapipe/examples/desktop/auto_frame/subgraphs:auto_frame_graph",
        "//mediapipe/examples/desktop/auto_frame/subgraphs:hand_gesture_recognition_live",
        "//mediapipe/framework/stream_handler:sync_set_input_stream_handler",
    ],
)
