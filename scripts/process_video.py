import math
import json
import cv2
from matplotlib import pyplot as plt
import mediapipe as mp

# from mediapipe.framework.formats import landmark_pb2

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "xtick.labelbottom": False,
        "xtick.bottom": False,
        "ytick.labelleft": False,
        "ytick.left": False,
        "xtick.labeltop": False,
        "xtick.top": False,
        "ytick.labelright": False,
        "ytick.right": False,
    }
)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def display_one_image(image, title, subplot, titlesize=16):
    """Displays one image along with the predicted category name and score."""
    plt.subplot(*subplot)
    plt.imshow(image)
    if len(title) > 0:
        plt.title(
            title,
            fontsize=int(titlesize),
            color="black",
            fontdict={"verticalalignment": "center"},
            pad=int(titlesize / 1.5),
        )
    return (subplot[0], subplot[1], subplot[2] + 1)


def display_batch_of_images_with_gestures_and_hand_landmarks(images, results):
    """Displays a batch of images with the gesture category
    and its score along with the hand landmarks."""
    # Images and labels.
    images = [image.numpy_view() for image in images]
    timestamps = []
    gestures = []
    for result in results:
        timestamps.append(result["frame_timestamp_ms"])
        gestures.append(result["gestures"])

    # Auto-squaring: this will drop data that does not fit into square or square-ish rectangle.
    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows

    print(f"Displaying {rows}x{cols} images.")

    # Size and spacing.
    FIGSIZE = 5 * max(rows, cols)
    SPACING = 0.1
    subplot = (rows, cols, 1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE, FIGSIZE / cols * rows))
    else:
        plt.figure(figsize=(FIGSIZE / rows * cols, FIGSIZE))

    # Display gestures and hand landmarks.
    for i, (image, gesture) in enumerate(
        zip(images[: rows * cols], gestures[: rows * cols])
    ):

        title = f"Time: {timestamps[i]} ms"
        for gesture_obj in gesture:
            title += f"\n{gesture_obj['gesture_category']} ({gesture_obj['gesture_score']:.2f}), "
            title += f"{gesture_obj['handedness_category']} ({gesture_obj['handedness_score']:.2f}); "

        print(title)

        dynamic_titlesize = FIGSIZE * SPACING / max(rows, cols) * 40 + 3
        annotated_image = image.copy()

        # hand_landmarks_list = [gesture_obj["hand_landmarks"] for gesture_obj in gesture]
        # for hand_landmarks in hand_landmarks_list:
        #     hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        #     hand_landmarks_proto.landmark.extend(
        #         [
        #             landmark_pb2.NormalizedLandmark(
        #                 x=landmark.x, y=landmark.y, z=landmark.z
        #             )
        #             for landmark in hand_landmarks
        #         ]
        #     )

        #     mp_drawing.draw_landmarks(
        #         annotated_image,
        #         hand_landmarks_proto,
        #         mp_hands.HAND_CONNECTIONS,
        #         mp_drawing_styles.get_default_hand_landmarks_style(),
        #         mp_drawing_styles.get_default_hand_connections_style(),
        #     )

        subplot = display_one_image(
            annotated_image, title, subplot, titlesize=dynamic_titlesize
        )

    # Layout.
    plt.tight_layout()
    plt.subplots_adjust(wspace=SPACING, hspace=SPACING)

    print("Displaying the images with the predicted gesture category and score.")
    # save the plot
    # plt.show()
    plt.savefig("../demo_1204_R_17/gesture-ziyu-noisy.png")


# Create a gesture recognizer instance with the live stream mode:
def print_result(result):
    print("gesture recognition result: {}".format(result))


def process_video_file(video_file_name, recognizer):
    """Processes a video file and returns the results."""

    try:
        print(f"Processing video file: {video_file_name}")

        # Use OpenCV’s VideoCapture to load the input video.
        cap = cv2.VideoCapture(video_file_name)
        fps = cap.get(cv2.CAP_PROP_FPS)

        print("Summary of the video file:")
        print(f"  FPS: {fps}")
        print(f"  Number of frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
        print(
            f"  Duration (ms): {int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * (1000 / fps))}"
        )

        # Loop through each frame in the video using VideoCapture#read()
        frame_index = 0
        mp_images = []
        mp_results = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate the timestamp for the current frame.
            frame_timestamp_ms = int(frame_index * (1000 / fps))

            # Convert the frame received from OpenCV to a MediaPipe’s Image object.
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )

            # Perform gesture recognition on the provided single image.
            gesture_recognition_result = recognizer.recognize_for_video(
                mp_image, frame_timestamp_ms
            )

            gestures = []
            if gesture_recognition_result.gestures:
                for gesture_obj in gesture_recognition_result.gestures:
                    gesture = gesture_obj[0]
                    if gesture != "None":
                        print(
                            f"Frame index: {frame_index}, "
                            f"Frame timestamp: {frame_timestamp_ms} ms, "
                            f"Gesture: {gesture.category_name} ({gesture.score:.2f}), "
                        )
                        gestures.append(
                            {
                                "category_name": gesture.category_name,
                                "score": gesture.score,
                            }
                        )

            handednesses = []
            if gesture_recognition_result.handedness:
                for handedness_obj in gesture_recognition_result.handedness:
                    handedness = handedness_obj[0]
                    print(
                        f"Frame index: {frame_index}, "
                        f"Frame timestamp: {frame_timestamp_ms} ms, "
                        f"Handedness: {handedness.category_name} ({handedness.score:.2f}), "
                    )
                    handednesses.append(
                        {
                            "category_name": handedness.category_name,
                            "score": handedness.score,
                        }
                    )

            if (
                gestures and handednesses
            ):  # and gesture_recognition_result.hand_landmarks:

                assert len(gestures) == len(handednesses)
                # assert len(gestures) == len(gesture_recognition_result.hand_landmarks)

                has_gesture = False
                gesture_list = []
                for gesture, handedness in zip(gestures, handednesses):
                    if gesture["category_name"] != "None":
                        has_gesture = True
                        gesture_list.append(
                            {
                                "gesture_category": gesture["category_name"],
                                "gesture_score": gesture["score"],
                                "handedness_category": handedness["category_name"],
                                "handedness_score": handedness["score"],
                                # "hand_landmarks": hand_landmarks,
                            }
                        )
                if has_gesture:
                    mp_images.append(mp_image)
                    mp_results.append(
                        {
                            "frame_timestamp_ms": frame_timestamp_ms,
                            "gestures": gesture_list,
                        }
                    )
                    print(
                        f"Succesfully processed frame {frame_index}, timestamp: {frame_timestamp_ms} ms"
                    )

            # Increment the frame index.
            frame_index += 1

        cap.release()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return mp_images, mp_results


if __name__ == "__main__":

    print("MediaPipe version:", mp.__version__)

    # Define the MediaPipe gesture recognizer classes and types.
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
    VisionRunningMode = mp.tasks.vision.RunningMode

    print("MediaPipe gesture recognizer classes and types defined.")

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path="../gesture_recognizer.task"),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,  # The maximum number of hands to detect.
        # result_callback=print_result, # only for live stream
    )
    recognizer = GestureRecognizer.create_from_options(options)

    print("MediaPipe gesture recognizer instance created.")

    VIDEO_FILENAMES = ["../demo_1204_R_17/R_17-ziyu-noisy.mp4"]
    for video_file_name in VIDEO_FILENAMES:
        mp_images, mp_results = process_video_file(video_file_name, recognizer)

        # print the results
        json_results = json.dumps({"results": mp_results}, indent=4)
        print(json_results)

        # save the results
        with open(video_file_name.replace(".mp4", "_ges.json"), "w") as f:
            f.write(json_results)

        if mp_images and mp_results:
            assert len(mp_images) == len(mp_results)
            display_batch_of_images_with_gestures_and_hand_landmarks(
                mp_images[:50], mp_results[:50]
            )
        else:
            print(f"No gesture recognition results found in: {video_file_name}")
