#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test keyboard key codes"""
import cv2
import numpy as np

print("Press arrow keys to see their codes. Press ESC to exit.")

# Create a simple window
img = np.zeros((300, 500, 3), dtype=np.uint8)
cv2.namedWindow("Key Tester", cv2.WINDOW_NORMAL)

while True:
    img[:] = 0
    cv2.putText(img, "Press arrow keys", (50, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "ESC to exit", (50, 150),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    cv2.imshow("Key Tester", img)

    key = cv2.waitKey(0)
    key_masked = key & 0xFF

    print(f"Key pressed: raw={key}, masked={key_masked}")

    if key == 27 or key_masked == 27:  # ESC
        break

    # Check which arrow
    if key == 2424832 or key_masked == 81:
        print("  -> LEFT arrow detected")
    elif key == 2555904 or key_masked == 83:
        print("  -> RIGHT arrow detected")
    elif key == 2490368 or key_masked == 82:
        print("  -> UP arrow detected")
    elif key == 2621440 or key_masked == 84:
        print("  -> DOWN arrow detected")

cv2.destroyAllWindows()
print("\nDone!")
