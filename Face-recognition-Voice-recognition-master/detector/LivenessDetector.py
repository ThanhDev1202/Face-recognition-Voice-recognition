import time
import random
import winsound
import math
from collections import deque

import cv2
import numpy as np


class LivenessDetector:
    """
    Liveness Detection dựa trên 5 facial landmarks của SCRFD.

    Challenge:
        LEFT        = quay mặt sang trái
        RIGHT       = quay mặt sang phải
        UP          = ngửa mặt lên
        DOWN        = cúi mặt xuống
        LEFT_RIGHT  = quay trái rồi quay phải

    Logic:
        WAITING -> MOVING -> TARGET -> HOLDING -> PASS

    LEFT_RIGHT:
        LEFT_MOVING -> LEFT_HOLD
                    -> RIGHT_MOVING -> RIGHT_HOLD -> PASS

    Không liên quan đến AntiSpoofing.
    """

    def __init__(
        self,
        movement_threshold=0.045,
        vertical_threshold=0.035,
        tilt_threshold=8.0,
        smile_threshold=0.06,
        confirm_frames=8,
        hold_time=0.8,
        min_frames=10,
        history_size=20,
        smooth_window=5,
        max_frame_jump=0.18,
        timeout=12.0,
    ):
        # Threshold
        self.movement_threshold = movement_threshold
        self.vertical_threshold = vertical_threshold

        # Giữ lại để tương thích code cũ
        self.tilt_threshold = tilt_threshold
        self.smile_threshold = smile_threshold

        # Confirmation
        self.confirm_frames = confirm_frames
        self.hold_time = hold_time
        self.min_frames = min_frames

        # Smoothing
        self.history_size = history_size
        self.smooth_window = smooth_window

        self.feature_history = deque(
            maxlen=self.history_size
        )

        # Frame jump protection
        self.max_frame_jump = max_frame_jump

        # Timeout
        self.timeout = timeout

        # Challenge
        self.challenge_types = [
            "LEFT",
            "RIGHT",
            "UP",
            "DOWN",
            "LEFT_RIGHT",
        ]

        self.sequence = []
        self.current_step = 0

        # Current challenge
        self.challenge_state = "WAITING"
        self.start_features = None
        self.last_position = None

        # Movement counters
        self.motion_frames = 0
        self.confirm_count = 0
        self.hold_start_time = None

        # LEFT_RIGHT
        self.lr_stage = "LEFT_MOVING"

        self.lr_left_features = None
        self.lr_right_features = None

        self.lr_left_motion_frames = 0
        self.lr_right_motion_frames = 0

        self.lr_left_confirm_count = 0
        self.lr_right_confirm_count = 0

        self.lr_hold_start_time = None

        # Global state
        self.started = False
        self.finished = False
        self.passed = False

        self.frame_count = 0
        self.start_time = None

        self.status = "Chưa bắt đầu"

    # ==========================================================
    # GENERATE RANDOM SEQUENCE
    # ==========================================================

    def generate_sequence(self):
        """
        Tạo 3 challenge ngẫu nhiên từ 5 challenge.
        """

        self.sequence = random.sample(
            self.challenge_types,
            3
        )

        self.current_step = 0

        return self.sequence

    # ==========================================================
    # START
    # ==========================================================

    def start(self):
        """
        Bắt đầu liveness.
        """

        self.generate_sequence()

        self.started = True
        self.finished = False
        self.passed = False

        self.frame_count = 0
        self.start_time = time.time()

        self.feature_history.clear()

        self._reset_challenge_state()

        self.status = self._get_challenge_text()

        return self.sequence

    # ==========================================================
    # RESET
    # ==========================================================

    def reset(self):
        """
        Reset toàn bộ detector.
        """

        self.sequence = []
        self.current_step = 0

        self.started = False
        self.finished = False
        self.passed = False

        self.frame_count = 0
        self.start_time = None

        self.feature_history.clear()

        self._reset_challenge_state()

        self.status = "Chưa bắt đầu"

    # ==========================================================
    # RESET CURRENT CHALLENGE
    # ==========================================================

    def _reset_challenge_state(self):
        """
        Reset state của challenge hiện tại.
        """

        self.challenge_state = "WAITING"

        self.start_features = None
        self.last_position = None

        self.motion_frames = 0
        self.confirm_count = 0
        self.hold_start_time = None

        # LEFT_RIGHT
        self.lr_stage = "LEFT_MOVING"

        self.lr_left_features = None
        self.lr_right_features = None

        self.lr_left_motion_frames = 0
        self.lr_right_motion_frames = 0

        self.lr_left_confirm_count = 0
        self.lr_right_confirm_count = 0

        self.lr_hold_start_time = None

        # Challenge mới phải có history mới
        self.feature_history.clear()

    # ==========================================================
    # GET SEQUENCE
    # ==========================================================

    def get_sequence(self):
        return self.sequence

    # ==========================================================
    # GET CURRENT CHALLENGE
    # ==========================================================

    def _get_current_challenge(self):

        if not self.sequence:
            return None

        if self.current_step >= len(self.sequence):
            return None

        return self.sequence[self.current_step]

    # ==========================================================
    # CHALLENGE TEXT
    # ==========================================================

    def _get_challenge_text(self):

        challenge = self._get_current_challenge()

        if challenge == "LEFT":
            return "Quay mặt sang TRÁI"

        if challenge == "RIGHT":
            return "Quay mặt sang PHẢI"

        if challenge == "UP":
            return "Ngửa mặt LÊN"

        if challenge == "DOWN":
            return "Cúi mặt XUỐNG"

        if challenge == "LEFT_RIGHT":

            if self.lr_stage in (
                "LEFT_MOVING",
                "LEFT_HOLD"
            ):
                return "Quay mặt sang TRÁI"

            if self.lr_stage in (
                "RIGHT_MOVING",
                "RIGHT_HOLD"
            ):
                return "Quay mặt sang PHẢI"

        return "Đang xác thực..."

    # ==========================================================
    # EXTRACT FEATURES
    # ==========================================================

    def extract_features(self, landmarks):
        """
        landmarks:
            SCRFD 5 landmarks

        Index:
            0 = left eye
            1 = right eye
            2 = nose
            3 = left mouth
            4 = right mouth

        Trả về:
            [nose_x, nose_y, mouth_x, mouth_y, eye_angle]
        """

        if landmarks is None:
            return None

        landmarks = np.asarray(
            landmarks,
            dtype=np.float32
        )

        if landmarks.shape[0] < 5:
            return None

        try:
            left_eye = landmarks[0]
            right_eye = landmarks[1]

            nose = landmarks[2]

            left_mouth = landmarks[3]
            right_mouth = landmarks[4]

            # Khoảng cách hai mắt
            eye_distance = np.linalg.norm(
                right_eye - left_eye
            )

            if eye_distance < 1e-6:
                return None

            # Nose normalized
            nose_x = (
                nose[0] - left_eye[0]
            ) / eye_distance

            nose_y = (
                nose[1] - left_eye[1]
            ) / eye_distance

            # Mouth center
            mouth_center = (
                left_mouth + right_mouth
            ) / 2.0

            mouth_x = (
                mouth_center[0] - left_eye[0]
            ) / eye_distance

            mouth_y = (
                mouth_center[1] - left_eye[1]
            ) / eye_distance

            # Eye angle
            dx = (
                right_eye[0] - left_eye[0]
            )

            dy = (
                right_eye[1] - left_eye[1]
            )

            eye_angle = math.degrees(
                math.atan2(dy, dx)
            )

            return np.array(
                [
                    nose_x,
                    nose_y,
                    mouth_x,
                    mouth_y,
                    eye_angle,
                ],
                dtype=np.float32
            )

        except Exception:
            return None

    # ==========================================================
    # SMOOTH FEATURES
    # ==========================================================

    def _smooth_features(self, features):
        """
        Moving average để giảm jitter của landmark.
        """

        self.feature_history.append(
            features.copy()
        )

        values = list(
            self.feature_history
        )

        values = values[
            -self.smooth_window:
        ]

        return np.mean(
            values,
            axis=0
        )

    # ==========================================================
    # FRAME JUMP CHECK
    # ==========================================================

    def _check_frame_jump(self, current):
        """
        Kiểm tra landmark có nhảy bất thường giữa 2 frame hay không.
        """

        if self.last_position is None:

            self.last_position = (
                current.copy()
            )

            return False

        jump = np.linalg.norm(
            current[:4]
            - self.last_position[:4]
        )

        self.last_position = (
            current.copy()
        )

        return jump > self.max_frame_jump

    # ==========================================================
    # UPDATE
    # ==========================================================

    def update(self, landmarks):
        """
        Gọi mỗi frame.

        Trả về:
            (passed, status)

        passed:
            True  = liveness PASS
            False = liveness FAIL
            None  = chưa hoàn thành
        """

        # ------------------------------------------------------
        # Chưa start
        # ------------------------------------------------------

        if not self.started:
            return None, "Chưa bắt đầu"

        # ------------------------------------------------------
        # Đã finish
        # ------------------------------------------------------

        if self.finished:
            return self.passed, self.status

        # ------------------------------------------------------
        # Timeout
        # ------------------------------------------------------

        if (
            self.start_time is not None
            and time.time() - self.start_time > self.timeout
        ):
            self.finished = True
            self.passed = False

            self.status = (
                "Liveness thất bại: hết thời gian"
            )

            return False, self.status

        # ------------------------------------------------------
        # Extract features
        # ------------------------------------------------------

        features = self.extract_features(
            landmarks
        )

        if features is None:

            self.status = (
                "Không nhận diện được khuôn mặt"
            )

            return None, self.status

        # ------------------------------------------------------
        # Smooth
        # ------------------------------------------------------

        smooth_features = (
            self._smooth_features(features)
        )

        self.frame_count += 1

        # ------------------------------------------------------
        # Frame jump
        # ------------------------------------------------------

        if self._check_frame_jump(
            smooth_features
        ):

            self.status = (
                "Chuyển động quá nhanh, "
                "giữ khuôn mặt ổn định"
            )

            return None, self.status

        # ------------------------------------------------------
        # First valid frame của challenge
        # ------------------------------------------------------

        if self.start_features is None:

            self.start_features = (
                smooth_features.copy()
            )

            self.last_position = (
                smooth_features.copy()
            )

            self.status = (
                self._get_challenge_text()
                + " - bắt đầu..."
            )

            return None, self.status

        # ------------------------------------------------------
        # Current challenge
        # ------------------------------------------------------

        challenge = self._get_current_challenge()

        if challenge is None:

            self.finished = True
            self.passed = True
            self.status = "Liveness PASS"

            return True, self.status

        # ------------------------------------------------------
        # LEFT
        # ------------------------------------------------------

        if challenge == "LEFT":

            return self._update_left(
                smooth_features
            )

        # ------------------------------------------------------
        # RIGHT
        # ------------------------------------------------------

        if challenge == "RIGHT":

            return self._update_right(
                smooth_features
            )

        # ------------------------------------------------------
        # UP
        # ------------------------------------------------------

        if challenge == "UP":

            return self._update_up(
                smooth_features
            )

        # ------------------------------------------------------
        # DOWN
        # ------------------------------------------------------

        if challenge == "DOWN":

            return self._update_down(
                smooth_features
            )

        # ------------------------------------------------------
        # LEFT_RIGHT
        # ------------------------------------------------------

        if challenge == "LEFT_RIGHT":

            return self._update_left_right(
                smooth_features
            )

        return None, self.status

    # ==========================================================
    # HORIZONTAL DELTA
    # ==========================================================

    def _horizontal_delta(self, features):

        return (
            features[0]
            - self.start_features[0]
        )

    # ==========================================================
    # VERTICAL DELTA
    # ==========================================================

    def _vertical_delta(self, features):

        return (
            features[1]
            - self.start_features[1]
        )

    # ==========================================================
    # LEFT
    # ==========================================================

    def _update_left(self, features):

        dx = self._horizontal_delta(
            features
        )

        # ------------------------------------------------------
        # WAITING / MOVING
        # ------------------------------------------------------

        if self.challenge_state in (
            "WAITING",
            "MOVING"
        ):

            if dx < -self.movement_threshold:

                self.challenge_state = "MOVING"

                self.motion_frames += 1

                self.status = (
                    "Quay mặt sang TRÁI..."
                    f" ({self.motion_frames}/"
                    f"{self.min_frames})"
                )

                if (
                    self.motion_frames
                    >= self.min_frames
                ):

                    self.challenge_state = "TARGET"
                    self.confirm_count = 0
                    self.hold_start_time = None

                    self.status = (
                        "Đã quay TRÁI - giữ nguyên..."
                    )

                return None, self.status

            self.motion_frames = 0

            self.status = (
                "Quay mặt sang TRÁI..."
            )

            return None, self.status

        # ------------------------------------------------------
        # TARGET
        # ------------------------------------------------------

        if self.challenge_state == "TARGET":

            if dx < -self.movement_threshold:

                self.confirm_count += 1

                self.status = (
                    "Xác nhận vị trí TRÁI..."
                    f" ({self.confirm_count}/"
                    f"{self.confirm_frames})"
                )

                if (
                    self.confirm_count
                    >= self.confirm_frames
                ):

                    self.challenge_state = "HOLDING"
                    self.hold_start_time = time.time()

                    self.status = (
                        "Giữ vị trí TRÁI..."
                    )

                return None, self.status

            self.confirm_count = 0
            self.challenge_state = "MOVING"

            self.status = (
                "Quay lại sang TRÁI..."
            )

            return None, self.status

        # ------------------------------------------------------
        # HOLDING
        # ------------------------------------------------------

        if self.challenge_state == "HOLDING":

            if dx < -self.movement_threshold:

                elapsed = (
                    time.time()
                    - self.hold_start_time
                )

                self.status = (
                    "Giữ TRÁI..."
                    f" {elapsed:.1f}/"
                    f"{self.hold_time:.1f}s"
                )

                if elapsed >= self.hold_time:

                    return self._challenge_passed()

                return None, self.status

            self.challenge_state = "MOVING"
            self.motion_frames = 0
            self.confirm_count = 0
            self.hold_start_time = None

            self.status = (
                "Chưa giữ đủ - quay lại TRÁI"
            )

            return None, self.status

        return None, self.status

    # ==========================================================
    # RIGHT
    # ==========================================================

    def _update_right(self, features):

        dx = self._horizontal_delta(
            features
        )

        # ------------------------------------------------------
        # WAITING / MOVING
        # ------------------------------------------------------

        if self.challenge_state in (
            "WAITING",
            "MOVING"
        ):

            if dx > self.movement_threshold:

                self.challenge_state = "MOVING"

                self.motion_frames += 1

                self.status = (
                    "Quay mặt sang PHẢI..."
                    f" ({self.motion_frames}/"
                    f"{self.min_frames})"
                )

                if (
                    self.motion_frames
                    >= self.min_frames
                ):

                    self.challenge_state = "TARGET"
                    self.confirm_count = 0
                    self.hold_start_time = None

                    self.status = (
                        "Đã quay PHẢI - giữ nguyên..."
                    )

                return None, self.status

            self.motion_frames = 0

            self.status = (
                "Quay mặt sang PHẢI..."
            )

            return None, self.status

        # ------------------------------------------------------
        # TARGET
        # ------------------------------------------------------

        if self.challenge_state == "TARGET":

            if dx > self.movement_threshold:

                self.confirm_count += 1

                self.status = (
                    "Xác nhận vị trí PHẢI..."
                    f" ({self.confirm_count}/"
                    f"{self.confirm_frames})"
                )

                if (
                    self.confirm_count
                    >= self.confirm_frames
                ):

                    self.challenge_state = "HOLDING"
                    self.hold_start_time = time.time()

                    self.status = (
                        "Giữ vị trí PHẢI..."
                    )

                return None, self.status

            self.confirm_count = 0
            self.challenge_state = "MOVING"

            self.status = (
                "Quay lại sang PHẢI..."
            )

            return None, self.status

        # ------------------------------------------------------
        # HOLDING
        # ------------------------------------------------------

        if self.challenge_state == "HOLDING":

            if dx > self.movement_threshold:

                elapsed = (
                    time.time()
                    - self.hold_start_time
                )

                self.status = (
                    "Giữ PHẢI..."
                    f" {elapsed:.1f}/"
                    f"{self.hold_time:.1f}s"
                )

                if elapsed >= self.hold_time:

                    return self._challenge_passed()

                return None, self.status

            self.challenge_state = "MOVING"
            self.motion_frames = 0
            self.confirm_count = 0
            self.hold_start_time = None

            self.status = (
                "Chưa giữ đủ - quay lại PHẢI"
            )

            return None, self.status

        return None, self.status

    # ==========================================================
    # UP
    # ==========================================================

    def _update_up(self, features):

        dy = self._vertical_delta(
            features
        )

        # ------------------------------------------------------
        # WAITING / MOVING
        # ------------------------------------------------------

        if self.challenge_state in (
            "WAITING",
            "MOVING"
        ):

            if dy < -self.vertical_threshold:

                self.challenge_state = "MOVING"

                self.motion_frames += 1

                self.status = (
                    "Ngửa mặt LÊN..."
                    f" ({self.motion_frames}/"
                    f"{self.min_frames})"
                )

                if (
                    self.motion_frames
                    >= self.min_frames
                ):

                    self.challenge_state = "TARGET"
                    self.confirm_count = 0
                    self.hold_start_time = None

                    self.status = (
                        "Đã ngửa LÊN - giữ nguyên..."
                    )

                return None, self.status

            self.motion_frames = 0

            self.status = (
                "Ngửa mặt LÊN..."
            )

            return None, self.status

        # ------------------------------------------------------
        # TARGET
        # ------------------------------------------------------

        if self.challenge_state == "TARGET":

            if dy < -self.vertical_threshold:

                self.confirm_count += 1

                self.status = (
                    "Xác nhận vị trí NGỬA..."
                    f" ({self.confirm_count}/"
                    f"{self.confirm_frames})"
                )

                if (
                    self.confirm_count
                    >= self.confirm_frames
                ):

                    self.challenge_state = "HOLDING"
                    self.hold_start_time = time.time()

                    self.status = (
                        "Giữ vị trí NGỬA..."
                    )

                return None, self.status

            self.confirm_count = 0
            self.challenge_state = "MOVING"

            self.status = (
                "Ngửa mặt LÊN rõ hơn..."
            )

            return None, self.status

        # ------------------------------------------------------
        # HOLDING
        # ------------------------------------------------------

        if self.challenge_state == "HOLDING":

            if dy < -self.vertical_threshold:

                elapsed = (
                    time.time()
                    - self.hold_start_time
                )

                self.status = (
                    "Giữ NGỬA..."
                    f" {elapsed:.1f}/"
                    f"{self.hold_time:.1f}s"
                )

                if elapsed >= self.hold_time:

                    return self._challenge_passed()

                return None, self.status

            self.challenge_state = "MOVING"
            self.motion_frames = 0
            self.confirm_count = 0
            self.hold_start_time = None

            self.status = (
                "Chưa giữ đủ - ngửa lại"
            )

            return None, self.status

        return None, self.status

    # ==========================================================
    # DOWN
    # ==========================================================

    def _update_down(self, features):

        dy = self._vertical_delta(
            features
        )

        # ------------------------------------------------------
        # WAITING / MOVING
        # ------------------------------------------------------

        if self.challenge_state in (
            "WAITING",
            "MOVING"
        ):

            if dy > self.vertical_threshold:

                self.challenge_state = "MOVING"

                self.motion_frames += 1

                self.status = (
                    "Cúi mặt XUỐNG..."
                    f" ({self.motion_frames}/"
                    f"{self.min_frames})"
                )

                if (
                    self.motion_frames
                    >= self.min_frames
                ):

                    self.challenge_state = "TARGET"
                    self.confirm_count = 0
                    self.hold_start_time = None

                    self.status = (
                        "Đã cúi XUỐNG - giữ nguyên..."
                    )

                return None, self.status

            self.motion_frames = 0

            self.status = (
                "Cúi mặt XUỐNG..."
            )

            return None, self.status

        # ------------------------------------------------------
        # TARGET
        # ------------------------------------------------------

        if self.challenge_state == "TARGET":

            if dy > self.vertical_threshold:

                self.confirm_count += 1

                self.status = (
                    "Xác nhận vị trí CÚI..."
                    f" ({self.confirm_count}/"
                    f"{self.confirm_frames})"
                )

                if (
                    self.confirm_count
                    >= self.confirm_frames
                ):

                    self.challenge_state = "HOLDING"
                    self.hold_start_time = time.time()

                    self.status = (
                        "Giữ vị trí CÚI..."
                    )

                return None, self.status

            self.confirm_count = 0
            self.challenge_state = "MOVING"

            self.status = (
                "Cúi mặt XUỐNG rõ hơn..."
            )

            return None, self.status

        # ------------------------------------------------------
        # HOLDING
        # ------------------------------------------------------

        if self.challenge_state == "HOLDING":

            if dy > self.vertical_threshold:

                elapsed = (
                    time.time()
                    - self.hold_start_time
                )

                self.status = (
                    "Giữ CÚI..."
                    f" {elapsed:.1f}/"
                    f"{self.hold_time:.1f}s"
                )

                if elapsed >= self.hold_time:

                    return self._challenge_passed()

                return None, self.status

            self.challenge_state = "MOVING"
            self.motion_frames = 0
            self.confirm_count = 0
            self.hold_start_time = None

            self.status = (
                "Chưa giữ đủ - cúi lại"
            )

            return None, self.status

        return None, self.status

    # ==========================================================
    # LEFT RIGHT
    # ==========================================================

    def _update_left_right(self, features):

        # ======================================================
        # LEFT MOVING
        # ======================================================

        if self.lr_stage == "LEFT_MOVING":

            dx = (
                features[0]
                - self.start_features[0]
            )

            if dx < -self.movement_threshold:

                self.lr_left_motion_frames += 1

                self.status = (
                    "Quay TRÁI..."
                    f" ({self.lr_left_motion_frames}/"
                    f"{self.min_frames})"
                )

                if (
                    self.lr_left_motion_frames
                    >= self.min_frames
                ):

                    self.lr_stage = "LEFT_HOLD"
                    self.lr_left_confirm_count = 0
                    self.lr_hold_start_time = None

                    self.status = (
                        "Đã quay TRÁI - giữ nguyên..."
                    )

                return None, self.status

            self.lr_left_motion_frames = 0

            self.status = (
                "Quay mặt sang TRÁI..."
            )

            return None, self.status

        # ======================================================
        # LEFT HOLD
        # ======================================================

        if self.lr_stage == "LEFT_HOLD":

            dx = (
                features[0]
                - self.start_features[0]
            )

            if dx < -self.movement_threshold:

                self.lr_left_confirm_count += 1

                self.status = (
                    "Xác nhận vị trí TRÁI..."
                    f" ({self.lr_left_confirm_count}/"
                    f"{self.confirm_frames})"
                )

                if (
                    self.lr_left_confirm_count
                    >= self.confirm_frames
                ):

                    if self.lr_hold_start_time is None:
                        self.lr_hold_start_time = time.time()

                    elapsed = (
                        time.time()
                        - self.lr_hold_start_time
                    )

                    self.status = (
                        "Giữ TRÁI..."
                        f" {elapsed:.1f}/"
                        f"{self.hold_time:.1f}s"
                    )

                    if elapsed >= self.hold_time:

                        self.lr_left_features = (
                            features.copy()
                        )

                        self.lr_stage = "RIGHT_MOVING"

                        self.lr_right_motion_frames = 0
                        self.lr_right_confirm_count = 0
                        self.lr_hold_start_time = None

                        self.status = (
                            "Đã giữ TRÁI - "
                            "bây giờ quay PHẢI"
                        )

                        return None, self.status

                return None, self.status

            self.lr_left_confirm_count = 0
            self.lr_hold_start_time = None

            self.status = (
                "Giữ TRÁI ổn định..."
            )

            return None, self.status

        # ======================================================
        # RIGHT MOVING
        # ======================================================

        if self.lr_stage == "RIGHT_MOVING":

            if self.lr_left_features is None:

                self.lr_stage = "LEFT_MOVING"

                self.status = (
                    "Lỗi trạng thái - quay lại TRÁI"
                )

                return None, self.status

            # Tính từ vị trí TRÁI
            dx = (
                features[0]
                - self.lr_left_features[0]
            )

            if dx > self.movement_threshold:

                self.lr_right_motion_frames += 1

                self.status = (
                    "Quay từ TRÁI sang PHẢI..."
                    f" ({self.lr_right_motion_frames}/"
                    f"{self.min_frames})"
                )

                if (
                    self.lr_right_motion_frames
                    >= self.min_frames
                ):

                    self.lr_stage = "RIGHT_HOLD"
                    self.lr_right_confirm_count = 0
                    self.lr_hold_start_time = None

                    self.status = (
                        "Đã quay PHẢI - giữ nguyên..."
                    )

                return None, self.status

            self.lr_right_motion_frames = 0

            self.status = (
                "Bây giờ quay sang PHẢI..."
            )

            return None, self.status

        # ======================================================
        # RIGHT HOLD
        # ======================================================

        if self.lr_stage == "RIGHT_HOLD":

            if self.lr_left_features is None:

                self.lr_stage = "LEFT_MOVING"

                self.status = (
                    "Lỗi trạng thái - quay lại TRÁI"
                )

                return None, self.status

            dx = (
                features[0]
                - self.lr_left_features[0]
            )

            if dx > self.movement_threshold:

                self.lr_right_confirm_count += 1

                self.status = (
                    "Xác nhận vị trí PHẢI..."
                    f" ({self.lr_right_confirm_count}/"
                    f"{self.confirm_frames})"
                )

                if (
                    self.lr_right_confirm_count
                    >= self.confirm_frames
                ):

                    if self.lr_hold_start_time is None:
                        self.lr_hold_start_time = time.time()

                    elapsed = (
                        time.time()
                        - self.lr_hold_start_time
                    )

                    self.status = (
                        "Giữ PHẢI..."
                        f" {elapsed:.1f}/"
                        f"{self.hold_time:.1f}s"
                    )

                    if elapsed >= self.hold_time:

                        self.lr_right_features = (
                            features.copy()
                        )

                        return self._challenge_passed()

                return None, self.status

            self.lr_right_confirm_count = 0
            self.lr_hold_start_time = None

            self.status = (
                "Giữ PHẢI ổn định..."
            )

            return None, self.status

        # ======================================================
        # FALLBACK
        # ======================================================

        self.lr_stage = "LEFT_MOVING"

        self.lr_left_motion_frames = 0
        self.lr_right_motion_frames = 0

        self.lr_left_confirm_count = 0
        self.lr_right_confirm_count = 0

        self.lr_hold_start_time = None

        self.status = (
            "Quay mặt sang TRÁI..."
        )

        return None, self.status

    # ==========================================================
    # CHALLENGE PASSED
    # ==========================================================

    def _challenge_passed(self):
         # Âm thanh báo challenge vừa hoàn thành
        winsound.Beep(1000, 150)
        current_challenge = (
            self._get_current_challenge()
        )

        self.current_step += 1

        # ------------------------------------------------------
        # Hoàn thành toàn bộ
        # ------------------------------------------------------

        if (
            self.current_step
            >= len(self.sequence)
        ):

            self.finished = True
            self.passed = True

            self.status = (
                "Liveness PASS"
            )
            # Âm thanh báo hoàn thành toàn bộ
            winsound.Beep(1200, 200)
            winsound.Beep(1500, 250)
            return True, self.status

        # ------------------------------------------------------
        # Reset challenge tiếp theo
        # ------------------------------------------------------

        self._reset_challenge_state()

        self.status = (
            f"{current_challenge} PASS - "
            f"Challenge {self.current_step + 1}/"
            f"{len(self.sequence)}: "
            f"{self._get_challenge_text()}"
        )

        return None, self.status

    # ==========================================================
    # DRAW STATUS
    # ==========================================================

    def draw_status(
        self,
        frame,
        position=(20, 40),
        color=(0, 255, 0),
        thickness=2,
    ):
        """
        Vẽ status lên frame.

        Giữ nguyên API để FaceVerifyWindow.py
        tiếp tục gọi.
        """

        if frame is None:
            return frame

        cv2.putText(
            frame,
            self.status,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            thickness,
            cv2.LINE_AA,
        )

        # ------------------------------------------------------
        # Progress
        # ------------------------------------------------------

        if self.sequence:

            progress = (
                f"Step "
                f"{min(self.current_step + 1, len(self.sequence))}/"
                f"{len(self.sequence)}"
            )

            cv2.putText(
                frame,
                progress,
                (
                    position[0],
                    position[1] + 35
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
                cv2.LINE_AA,
            )

        return frame