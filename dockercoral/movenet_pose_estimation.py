import argparse
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter

_NUM_KEYPOINTS = 17

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m', '--model', required=True, help='File path of .tflite file.')
    args = parser.parse_args()

    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()

    cap = cv2.VideoCapture(0)  # Open the default camera (usually the first camera)

    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            print("Failed to capture image")
            break

        # Convert the frame from BGR to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to PIL Image
        img = Image.fromarray(frame_rgb)
        resized_img = img.resize(common.input_size(interpreter), Image.LANCZOS)
        common.set_input(interpreter, resized_img)

        interpreter.invoke()

        pose = common.output_tensor(interpreter, 0).copy().reshape(_NUM_KEYPOINTS, 3)

        # Draw keypoints on the original frame
        draw = ImageDraw.Draw(img)
        width, height = img.size
        for i in range(0, _NUM_KEYPOINTS):
            draw.ellipse(
                xy=[
                    pose[i][1] * width - 2, pose[i][0] * height - 2,
                    pose[i][1] * width + 2, pose[i][0] * height + 2
                ],
                fill=(255, 0, 0))

        # Convert the PIL Image back to OpenCV format (BGR)
        frame_with_keypoints = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Display the resulting frame
        cv2.imshow('MoveNet Pose Estimation', frame_with_keypoints)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
