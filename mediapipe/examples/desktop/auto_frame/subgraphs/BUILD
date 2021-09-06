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

load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_simple_subgraph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])


cc_library(
    name = "desktop_offline_calculators",
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:immediate_mux_calculator",
        "//mediapipe/calculators/core:packet_inner_join_calculator",
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/calculators/video:opencv_video_decoder_calculator",
        "//mediapipe/calculators/video:opencv_video_encoder_calculator",
    ],
)

cc_library(
    name = "desktop_tflite_calculators",
    deps = [
        ":desktop_offline_calculators",
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:merge_calculator",
        "//mediapipe/graphs/hand_tracking/subgraphs:hand_renderer_cpu",
        "//mediapipe/modules/hand_landmark:hand_landmark_tracking_cpu",
    ],
)

mediapipe_simple_subgraph(
    name = "auto_frame_graph",
    graph = "auto_frame_graph.pbtxt",
    register_as = "AutoFrameSubgraph",
    deps = [
        "//mediapipe/examples/desktop/auto_frame/calculators:detections_to_autoframe_roi_calculator",
        "//mediapipe/calculators/image:image_cropping_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/graphs/face_detection:desktop_live_calculators",
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/calculators/core:packet_thinner_calculator",
        "//mediapipe/examples/desktop/auto_frame/calculators:detections_to_imageframe_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "hand_gesture_recognition_live",
    graph = "hand_gesture_recognition_live.pbtxt",
    register_as = "HandGestureRecognitionSubgraph",
    deps = [
        ":desktop_tflite_calculators",
        "//mediapipe/examples/desktop/auto_frame/calculators:hand_gesture_recognition_calculator",
        "//mediapipe/examples/desktop/auto_frame/calculators:put_text_on_image_calculator",
        "//mediapipe/calculators/core:constant_side_packet_calculator",
    ],
)
