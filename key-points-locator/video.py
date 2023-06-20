import cv2

def sift_keypoint_detection_video(video_file):
    """Detects SIFT keypoints in a video.

    Args:
        video_file: The path to the video file.

    Returns:
        A list of SIFT keypoints.
    """

    # Create a SIFT object.
    sift = cv2.xfeatures2d.SIFT_create()

    # Create a video capture object.
    video_capture = cv2.VideoCapture(video_file)

    # Loop over the frames in the video.
    while True:
        # Grab the next frame.
        success, frame = video_capture.read()

        # If the frame is not grabbed, break from the loop.
        if not success:
            break

        # Convert the frame to grayscale.
        # input image is converted to gray scale image
        imagegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # using the SIRF algorithm to detect key
        # points in the image
        features = cv2.SIFT_create()

        keypoints = features.detect(imagegray, None)

        # drawKeypoints function is used to draw keypoints
        output_image = cv2.drawKeypoints(imagegray, keypoints, 0, (0, 0, 255),
                                        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        # displaying the image with keypoints as the
        # output on the screen

        cv2.imshow("SIFT Keypoints", output_image)
        cv2.waitKey(1)

    # Release the video capture object.
    video_capture.release()

if __name__ == "__main__":
    # Path to the video file.
    video_file = "video.mp4"

    # Detect SIFT keypoints in the video.
    sift_keypoint_detection_video(video_file)
