# importing necessary libraries i.e... opencv,mediapipe
import cv2  # open cv
import mediapipe as mp  # pose estimation

mp_drawing = mp.solutions.drawing_utils  # pose landmarks
mp_pose = mp.solutions.pose  # performing pose estimation

# Initialize the push-up and pull-up counters and detection flags
pushup_counter = 0  # track number of push-ups
is_pushup = False  # detect whether push-up is being performed or not

pullup_counter = 0  # track number of pull-ups
is_pullup = False  # detect whether pull-up is being performed or not

# Function to detect push-ups and pull-ups
def detect_exercises(image):
    global pushup_counter, is_pushup, pullup_counter, is_pullup

    # Convert the image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect the pose landmarks in the image
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(image_rgb)

        # Draw the pose landmarks on the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

        # Check if the left and right wrists are above the corresponding shoulders
        if results.pose_landmarks:
            left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y

            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

            # Check if wrists are above shoulders
            if left_wrist < left_shoulder and right_wrist < right_shoulder:
                # If wrists are above the shoulders, it's a potential push-up or pull-up
                if not is_pushup and not is_pullup:
                    if push_up_condition_met(results.pose_landmarks):
                        pushup_counter += 1
                        is_pushup = True
                    elif pull_up_condition_met(results.pose_landmarks):
                        pullup_counter += 1
                        is_pullup = True
            else:
                is_pushup = False
                is_pullup = False

    # Display the push-up and pull-up counts on the image
    cv2.putText(image, f"Push-ups: {pushup_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Pull-ups: {pullup_counter}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return image


def push_up_condition_met(landmarks):
    # Check if specific pose/movement for push-ups is met
    # Here's an example condition for push-ups:
    # If the elbows are lower than the shoulders, consider it a push-up
    left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y

    right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

    return left_elbow > left_shoulder and right_elbow > right_shoulder


def pull_up_condition_met(landmarks):
    # Check if specific pose/movement for pull-ups is met
    # Here's an example condition for pull-ups:
    # If the elbows are higher than the shoulders, consider it a pull-up
    left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y

    right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

    return left_elbow < left_shoulder and right_elbow < right_shoulder


# Start push-up and pull-up detection

cap = cv2.VideoCapture(0)  # Change the parameter to the desired camera index if needed

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for mirror effect

    # Detect and track push-ups and pull-ups
    frame = detect_exercises(frame)

    # Display the resulting frame
    cv2.imshow('Exercise Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
