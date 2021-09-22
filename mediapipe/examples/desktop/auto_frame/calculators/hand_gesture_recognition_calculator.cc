#include <cmath>
#include <vector>

#include "mediapipe/examples/desktop/auto_frame/autoframe_messages.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe {

    namespace {
        constexpr char normRectTag[] = "NORM_RECT";
        constexpr char normalizedLandmarkListTag[] = "NORM_LANDMARKS";
        constexpr char recognizedHandGestureTag[] = "RECOGNIZED_HAND_GESTURE";
        constexpr char normalizedLandmarks[] = "LANDMARKS";
        constexpr char kImageSizeTag[] = "IMAGE_SIZE";

    }  // namespace

    static const int MAX_POINTS_PER_FINGER = 4;
    static const int MAX_FINGERS = 5;

    static const int HAND_LANDMARKS[MAX_FINGERS][MAX_POINTS_PER_FINGER] = {
        {1, 2, 3, 4},      // THUMB_LANDMARKS
        {5, 6, 7, 8},      // INDEX_FINGER_LANDMARKS
        {9, 10, 11, 12},   // MIDDLE_FINGER_LANDMARKS
        {13, 14, 15, 16},  // RING_FINGER_LANDMARKS
        {17, 18, 19, 20},  // PINKY_LANDMARKS
    };

    enum DIRECTION {
        UP = 0,
        DOWN = 1,
        LEFT = 2,
        RIGHT = 3,
        DONT_CARE = 4,
    };

#define PI 3.14159265

    class FingerState {
       private:
       public:
        bool mIsOpen;
        DIRECTION mDir;

        FingerState(bool openState, DIRECTION dir)
            : mIsOpen{openState}, mDir{dir} {}

        FingerState() : mIsOpen{false}, mDir{DONT_CARE} {}

        bool operator==(const FingerState &state) {
            if ((mIsOpen == state.mIsOpen) && (mDir == state.mDir)) return true;

            return false;
        }

        bool operator!=(const FingerState &state) {
            if ((mIsOpen != state.mIsOpen) || (mDir != state.mDir)) return true;

            return false;
        }
    };

    static const int MAX_GESTURES = 13;
    static std::string GESTURE_TEMPLATE_LABEL[MAX_GESTURES] = {
        "OK",          "ONE",          "TWO",        "THREE",     "FOUR",
        "FIVE",        "SIX",          "SPIDER_MAN", "THUMBS_UP", "THUMBS_DOWN",
        "THUMBS_LEFT", "THUMBS_RIGHT", "FIST"};

    // Library of gestures:
    // First element represents if finger is open or closed. TRUE = Finger open
    // , FALSE = Finger closed Second element represents finger direction . It
    // tells us if finger is pointing up/down/left/right NOTE: Finger direction
    // is considered only when it is open Finger positions : THUMB , INDEX,
    // MIDDLE, RING, PINKY
    static const std::vector<FingerState> GESTURE_TEMPLATE[MAX_GESTURES] = {
        // OK Gesture
        {{false, DIRECTION::DONT_CARE},
         {false, DIRECTION::DONT_CARE},
         {true, DIRECTION::UP},
         {true, DIRECTION::UP},
         {true, DIRECTION::UP}},
        // ONE_GESTURE
        {{false, DIRECTION::DONT_CARE},
         {true, DIRECTION::UP},
         {false, DIRECTION::DONT_CARE},
         {false, DIRECTION::DONT_CARE},
         {false, DIRECTION::DONT_CARE}},
        // TWO_GESTURE
        {{false, DIRECTION::DONT_CARE},
         {true, DIRECTION::UP},
         {true, DIRECTION::UP},
         {false, DIRECTION::DONT_CARE},
         {false, DIRECTION::DONT_CARE}},
        // THREE_GESTURE
        {{false, DIRECTION::DONT_CARE},
         {true, DIRECTION::UP},
         {true, DIRECTION::UP},
         {true, DIRECTION::UP},
         {false, DIRECTION::DONT_CARE}},
        // FOUR_GESTURE
        {{false, DIRECTION::DONT_CARE},
         {true, DIRECTION::UP},
         {true, DIRECTION::UP},
         {true, DIRECTION::UP},
         {true, DIRECTION::UP}},
        // FIVE_GESTURE
        {{true, DIRECTION::UP},
         {true, DIRECTION::UP},
         {true, DIRECTION::UP},
         {true, DIRECTION::UP},
         {true, DIRECTION::UP}},
        // SIX_GESTURE
        {{true, DIRECTION::UP},
         {false, DIRECTION::DONT_CARE},
         {false, DIRECTION::DONT_CARE},
         {false, DIRECTION::DONT_CARE},
         {true, DIRECTION::UP}},
        // SPIDERMAN_GESTURE
        {{false, DIRECTION::DONT_CARE},
         {true, DIRECTION::UP},
         {false, DIRECTION::DONT_CARE},
         {false, DIRECTION::DONT_CARE},
         {true, DIRECTION::UP}},
        // THUMBS_UP_GESTURE
        {{true, DIRECTION::UP},
         {false, DIRECTION::DONT_CARE},
         {false, DIRECTION::DONT_CARE},
         {false, DIRECTION::DONT_CARE},
         {false, DIRECTION::DONT_CARE}},
        // THUMBPS_DOWN_GESTURE
        {{true, DIRECTION::DOWN},
         {false, DIRECTION::DONT_CARE},
         {false, DIRECTION::DONT_CARE},
         {false, DIRECTION::DONT_CARE},
         {false, DIRECTION::DONT_CARE}},
        // THUMBPS_LEFT_GESTURE
        {{true, DIRECTION::LEFT},
         {false, DIRECTION::DONT_CARE},
         {false, DIRECTION::DONT_CARE},
         {false, DIRECTION::DONT_CARE},
         {false, DIRECTION::DONT_CARE}},
        // THUMBPS_RIGHT_GESTURE
        {{true, DIRECTION::RIGHT},
         {false, DIRECTION::DONT_CARE},
         {false, DIRECTION::DONT_CARE},
         {false, DIRECTION::DONT_CARE},
         {false, DIRECTION::DONT_CARE}},
        // FIST_GESTURE
        {{false, DIRECTION::DONT_CARE},
         {false, DIRECTION::DONT_CARE},
         {false, DIRECTION::DONT_CARE},
         {false, DIRECTION::DONT_CARE},
         {false, DIRECTION::DONT_CARE}},
    };

    // Graph config:
    //
    // node {
    //   calculator: "HandGestureRecognitionCalculator"
    //   input_stream: "NORM_LANDMARKS:scaled_landmarks"
    //   input_stream: "NORM_RECT:hand_rect_for_next_frame"
    // }
    class HandGestureRecognitionCalculator : public CalculatorBase {
       public:
        static ::mediapipe::Status GetContract(CalculatorContract *cc);
        ::mediapipe::Status Open(CalculatorContext *cc) override;

        ::mediapipe::Status Process(CalculatorContext *cc) override;

       private:
        void InitCombinedDetection(mediapipe::CombinedDetection *detection,
                                   int img_width, int img_height) {
            detection->set_type(mediapipe::CombinedDetection::NONE);

            auto rect = absl::make_unique<Rect>();
            rect->set_x_center(img_width / 2);
            rect->set_y_center(img_height / 2);
            rect->set_width(img_width);
            rect->set_height(img_height);

            detection->set_allocated_bbox(rect.release());
            detection->set_gesture("UNKNOWN");
        }

        float get_Euclidean_DistanceAB(float a_x, float a_y, float b_x,
                                       float b_y) {
            float dist = std::pow(a_x - b_x, 2) + pow(a_y - b_y, 2);
            return std::sqrt(dist);
        }

        bool isThumbNearFirstFinger(NormalizedLandmark point1,
                                    NormalizedLandmark point2) {
            float distance = this->get_Euclidean_DistanceAB(
                point1.x(), point1.y(), point2.x(), point2.y());
            return distance < 0.1;
        }

        float _calculate_distance(float a_x, float a_y, float b_x, float b_y) {
            float dist = std::pow(a_x - b_x, 2) + pow(a_y - b_y, 2);
            return std::sqrt(dist);
        }

        double _calculate_angle(float a_x, float a_y, float b_x, float b_y) {
            double x, y, angle;
            x = a_x - b_x;
            y = a_y - b_y;
            angle = atan2(y, x) * 180 / PI;
            return angle;
        }

        enum DIRECTION _get_finger_direction(float a_x, float a_y, float b_x,
                                             float b_y) {
            double angle = _calculate_angle(a_x, a_y, b_x, b_y);
            if ((angle > 0 && angle < 45) || (angle < 0 && angle > -45)) {
                return DIRECTION::LEFT;
            } else if ((angle > 45) && (angle < 135)) {
                return DIRECTION::UP;
            } else if ((angle > 135 && angle < 180) ||
                       (angle < -135 && angle > -180)) {
                return DIRECTION::RIGHT;
            } else {
                return DIRECTION::DOWN;
            }
        }
        FingerState _calculate_finger_state2(
            NormalizedLandmarkList landmarkList, const int FINGER_landmarks[],
            float *diff_accumulate, float tolerence = 0.0045) {
            FingerState state = {false, DIRECTION::DONT_CARE};
            // the last item in finger will be used to calculate if finger is
            // straight or bent
            float max_sub_finger_distance = 0;
            for (int i = 0; i < MAX_POINTS_PER_FINGER - 1; i++) {
                float distance = _calculate_distance(
                    landmarkList.landmark(0).x(), landmarkList.landmark(0).y(),
                    landmarkList.landmark(FINGER_landmarks[i]).x(),
                    landmarkList.landmark(FINGER_landmarks[i]).y());

                if (max_sub_finger_distance < distance) {
                    max_sub_finger_distance = distance;
                }
            }

            float wrist_to_tip_distance = _calculate_distance(
                landmarkList.landmark(0).x(), landmarkList.landmark(0).y(),
                landmarkList
                    .landmark(FINGER_landmarks[MAX_POINTS_PER_FINGER - 1])
                    .x(),
                landmarkList
                    .landmark(FINGER_landmarks[MAX_POINTS_PER_FINGER - 1])
                    .y());

            if (wrist_to_tip_distance > max_sub_finger_distance) {
                state.mIsOpen = true;
            }

            if (state.mIsOpen) {
                state.mDir = _get_finger_direction(
                    landmarkList.landmark(FINGER_landmarks[0]).x(),
                    landmarkList.landmark(FINGER_landmarks[0]).y(),
                    landmarkList
                        .landmark(FINGER_landmarks[MAX_POINTS_PER_FINGER - 1])
                        .x(),
                    landmarkList
                        .landmark(FINGER_landmarks[MAX_POINTS_PER_FINGER - 1])
                        .y());
            }

            return state;
        }

        FingerState _calculate_finger_state(NormalizedLandmarkList landmarkList,
                                            int finger, float *diff_accumulate,
                                            float tolerence = 0.0009) {
            FingerState state = {false, DIRECTION::DONT_CARE};
            // the last item in finger will be used to calculate if finger is
            // straight or bent
            float finger_length_acc = 0;

            for (int i = 0; i < MAX_POINTS_PER_FINGER - 1; i++) {
                finger_length_acc += _calculate_distance(
                    landmarkList.landmark(HAND_LANDMARKS[finger][i]).x(),
                    landmarkList.landmark(HAND_LANDMARKS[finger][i]).y(),
                    landmarkList.landmark(HAND_LANDMARKS[finger][i + 1]).x(),
                    landmarkList.landmark(HAND_LANDMARKS[finger][i + 1]).y());
            }

            float finger_length = _calculate_distance(
                landmarkList.landmark(HAND_LANDMARKS[finger][0]).x(),
                landmarkList.landmark(HAND_LANDMARKS[finger][0]).y(),
                landmarkList
                    .landmark(HAND_LANDMARKS[finger][MAX_POINTS_PER_FINGER - 1])
                    .x(),
                landmarkList
                    .landmark(HAND_LANDMARKS[finger][MAX_POINTS_PER_FINGER - 1])
                    .y());

            *diff_accumulate += abs(finger_length - finger_length_acc);

            float diff = abs(finger_length - finger_length_acc);

            if (finger == 0) tolerence = 0.007;
            // if joints are aligned the sum of accumation of length of joints
            // should be equal to the length from bottom joint to top most
            // joint.
            if (diff < tolerence) {
                state.mIsOpen = true;
            }

            if (state.mIsOpen) {
                state.mDir = _get_finger_direction(
                    landmarkList.landmark(HAND_LANDMARKS[finger][0]).x(),
                    landmarkList.landmark(HAND_LANDMARKS[finger][0]).y(),
                    landmarkList
                        .landmark(
                            HAND_LANDMARKS[finger][MAX_POINTS_PER_FINGER - 1])
                        .x(),
                    landmarkList
                        .landmark(
                            HAND_LANDMARKS[finger][MAX_POINTS_PER_FINGER - 1])
                        .y());
            }

            return state;
        }

        std::string _check_gesture(std::vector<FingerState> cur_handState) {
            for (int gesture = 0; gesture < MAX_GESTURES; gesture++) {
                std::vector<FingerState> gestureState =
                    GESTURE_TEMPLATE[gesture];
                for (int finger = 0; finger < MAX_FINGERS; finger++) {
                    if (gestureState[finger] == cur_handState[finger]) {
                        if (finger == MAX_FINGERS - 1) {
                            return GESTURE_TEMPLATE_LABEL[gesture];
                        }
                    } else
                        break;
                }
            }
            return "UNKNOWN";
        }
    };

    REGISTER_CALCULATOR(HandGestureRecognitionCalculator);

    ::mediapipe::Status HandGestureRecognitionCalculator::GetContract(
        CalculatorContract *cc) {
        RET_CHECK(cc->Inputs().HasTag(normalizedLandmarks));
        cc->Inputs()
            .Tag(normalizedLandmarks)
            .Set<std::vector<mediapipe::NormalizedLandmarkList>>();

        RET_CHECK(cc->Inputs().HasTag(kImageSizeTag));
        cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();

        RET_CHECK(cc->Outputs().HasTag(recognizedHandGestureTag));
        cc->Outputs()
            .Tag(recognizedHandGestureTag)
            .Set<mediapipe::CombinedDetection>();

        return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status HandGestureRecognitionCalculator::Open(
        CalculatorContext *cc) {
        cc->SetOffset(TimestampDiff(0));
        return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status HandGestureRecognitionCalculator::Process(
        CalculatorContext *cc) {
        // std::cout << "Inside HandGestureRecognitionCalculator" << std::endl;
        if (cc->Inputs().HasTag(normalizedLandmarks) &&
            cc->Inputs().Tag(normalizedLandmarks).IsEmpty()) {
            return absl::OkStatus();
        }
        mediapipe::CombinedDetection *recognized_hand_gesture;

        const auto &landmarks =
            cc->Inputs()
                .Tag(normalizedLandmarks)
                .Get<std::vector<mediapipe::NormalizedLandmarkList>>();

        RET_CHECK_GT(landmarks.size(), 0) << "Input landmarks vector is empty.";

        const auto &landmarkList = landmarks[0];
        RET_CHECK_GT(landmarkList.landmark_size(), 0)
            << "Input landmark vector is empty.";

        // Check the state of each finger in the hand , check if the finger is
        // open or closed and check the direction it is pointing towards
        std::vector<FingerState> handState;
        float fing_diff[MAX_FINGERS] = {0};
        static int count = 0;
        for (int fing = 0; fing < MAX_FINGERS; fing++) {
            FingerState state;
            state =
                _calculate_finger_state(landmarkList, fing, &fing_diff[fing]);
            handState.push_back(state);
        }

        // Extract image size
        absl::optional<std::pair<int, int>> image_size;
        if (HasTagValue(cc->Inputs(), kImageSizeTag)) {
            image_size =
                cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();
        }
        int32 frame_width = image_size->first;
        int32 frame_height = image_size->second;

        std::unique_ptr<mediapipe::CombinedDetection> detection =
            std::make_unique<mediapipe::CombinedDetection>();
        InitCombinedDetection(detection.get(), frame_width, frame_height);

        detection->set_type(mediapipe::CombinedDetection::GESTURE);
        detection->set_gesture(_check_gesture(handState));

        cc->Outputs()
            .Tag(recognizedHandGestureTag)
            .AddPacket(Adopt(detection.release()).At(cc->InputTimestamp()));
        return ::mediapipe::OkStatus();
    }  // namespace mediapipe

}  // namespace mediapipe
