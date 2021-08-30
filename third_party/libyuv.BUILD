# Description:
#   The libyuv package provides implementation yuv image conversion, rotation
#   and scaling.

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

cc_library(
    name = "libyuv",
    srcs = glob(
        [
            "source/*.cc",
            "include/libyuv/*.h",
        ],
    ),
    defines = [
        "LIBYUV_DISABLE_NEON"
    ],
    hdrs = [
        "include/libyuv/compare.h",
        "include/libyuv/convert.h",
        "include/libyuv/video_common.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
)
