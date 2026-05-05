"""
webcam_detect.py — Live webcam water level detector.
Author: Iyad Laphir

Captures frames from the webcam and passes them to predict.py which
runs WaterSegNet and returns the annotated image + level.

Controls:
  SPACE — capture current frame and analyse
  Q     — quit

Usage:
  python webcam_detect.py
  python webcam_detect.py --save output.jpg
  python webcam_detect.py --camera 1
"""

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import predict


def main():
    parser = argparse.ArgumentParser(description="Webcam water level detector")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default 0)")
    parser.add_argument("--save",   type=str, default=None,
                        help="Save result image to this path")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {args.camera}")
        sys.exit(1)

    print("Camera open.")
    print("  SPACE — capture and analyse")
    print("  Q     — quit\n")

    last_level = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame.")
            break

        preview = frame.copy()
        if last_level is not None:
            cv2.putText(preview, f"Last level: {last_level}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(preview, "SPACE: analyse   Q: quit",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow("Water Level Detector — Live Preview", preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break

        elif key == ord(" "):
            print("Analysing...")
            result     = predict(frame)
            last_level = result.level_name

            print(f"  Level      : {result.level_name}  (0=empty, 4=full)")
            print(f"  Confidence : {result.confidence:.2%}")
            if result.water_box is not None:
                print(f"  Water box  : {result.water_box}")
            else:
                print("  Water box  : not detected")
            print()

            cv2.imshow("Water Level Detector — Result", result.image)

            if args.save:
                cv2.imwrite(args.save, result.image)
                print(f"  Saved to {args.save}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
