import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # need 2 hands for some signs

offset = 20
imgSize = 300

folder = "C:\SL_Detector\TEST_THREE\DataNew\Beef"
counter = 0

last_save = 0
delay = 0.06 #how long before autocapture 2 handed signs

"""IF no folder exists, create one"""
if not os.path.exists(folder):
    os.makedirs(folder)

print("\n" + "=" * 60)
print("TWO-HAND DATA COLLECTION")
print("=" * 60)
print("Press 's' to save")
print("Press 'q' to quit")
print("=" * 60 + "\n")

while True:
    success, img = cap.read()

    if not success:
        break

    hands, img = detector.findHands(img)

    imgHeight, imgWidth, _ = img.shape
    imgWhite = None
    num_hands = len(hands) if hands else 0

    if hands:
        if num_hands == 1:
            # ========== SINGLE HAND ==========
            hand = hands[0]
            x, y, w, h = hand['bbox']

            y1 = max(0, y - offset)
            y2 = min(imgHeight, y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(imgWidth, x + w + offset)

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size > 0:
                aspectRatio = h / w

                try:
                    if aspectRatio > 1:
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        if wCal > 0:
                            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                            wGap = math.ceil((imgSize - wCal) / 2)
                            imgWhite[:, wGap:wCal + wGap] = imgResize
                    else:
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        if hCal > 0:
                            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                            hGap = math.ceil((imgSize - hCal) / 2)
                            imgWhite[hGap:hCal + hGap, :] = imgResize

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                except Exception as e:
                    print(f"Error processing: {e}")

        elif num_hands == 2:
            # ========== TWO HANDS - CAPTURE BOTH! ==========

            hand1 = hands[0]
            hand2 = hands[1]

            x1, y1, w1, h1 = hand1['bbox']
            x2, y2, w2, h2 = hand2['bbox']

            # Safe boundaries for hand 1
            y1_1 = max(0, y1 - offset)
            y1_2 = min(imgHeight, y1 + h1 + offset)
            x1_1 = max(0, x1 - offset)
            x1_2 = min(imgWidth, x1 + w1 + offset)

            # Safe boundaries for hand 2
            y2_1 = max(0, y2 - offset)
            y2_2 = min(imgHeight, y2 + h2 + offset)
            x2_1 = max(0, x2 - offset)
            x2_2 = min(imgWidth, x2 + w2 + offset)

            imgCrop1 = img[y1_1:y1_2, x1_1:x1_2]
            imgCrop2 = img[y2_1:y2_2, x2_1:x2_2]

            if imgCrop1.size > 0 and imgCrop2.size > 0:
                try:
                    # ✅ FIX: Resize both crops to same height before stacking
                    h_crop1, w_crop1 = imgCrop1.shape[:2]
                    h_crop2, w_crop2 = imgCrop2.shape[:2]

                    # Use the maximum height
                    max_height = max(h_crop1, h_crop2)

                    # Resize crop1 to max_height (maintain aspect ratio)
                    scale1 = max_height / h_crop1
                    new_w1 = int(w_crop1 * scale1)
                    imgCrop1_resized = cv2.resize(imgCrop1, (new_w1, max_height))

                    # Resize crop2 to max_height (maintain aspect ratio)
                    scale2 = max_height / h_crop2
                    new_w2 = int(w_crop2 * scale2)
                    imgCrop2_resized = cv2.resize(imgCrop2, (new_w2, max_height))

                    # Now they have the same height - can stack horizontally!
                    imgCrop_combined = np.hstack([imgCrop1_resized, imgCrop2_resized])

                    # Create WIDER white background for TWO hands (300x600)
                    imgWhite = np.ones((imgSize, imgSize * 2, 3), np.uint8) * 255

                    # Process HAND 1 (left side of image)
                    aspectRatio1 = h1 / w1
                    if aspectRatio1 > 1:
                        k = imgSize / h1
                        wCal = math.ceil(k * w1)
                        if wCal > 0:
                            imgResize1 = cv2.resize(imgCrop1, (wCal, imgSize))
                            wGap = math.ceil((imgSize - wCal) / 2)
                            imgWhite[:, wGap:wCal + wGap] = imgResize1
                    else:
                        k = imgSize / w1
                        hCal = math.ceil(k * h1)
                        if hCal > 0:
                            imgResize1 = cv2.resize(imgCrop1, (imgSize, hCal))
                            hGap = math.ceil((imgSize - hCal) / 2)
                            imgWhite[hGap:hCal + hGap, :imgSize] = imgResize1

                    # Process HAND 2 (right side of image)
                    aspectRatio2 = h2 / w2
                    if aspectRatio2 > 1:
                        k = imgSize / h2
                        wCal = math.ceil(k * w2)
                        if wCal > 0:
                            imgResize2 = cv2.resize(imgCrop2, (wCal, imgSize))
                            wGap = math.ceil((imgSize - wCal) / 2)
                            imgWhite[:, imgSize + wGap:imgSize + wCal + wGap] = imgResize2
                    else:
                        k = imgSize / w2
                        hCal = math.ceil(k * h2)
                        if hCal > 0:
                            imgResize2 = cv2.resize(imgCrop2, (imgSize, hCal))
                            hGap = math.ceil((imgSize - hCal) / 2)
                            imgWhite[hGap:hCal + hGap, imgSize:] = imgResize2

                    # Show combined crops (now same height!)
                    cv2.imshow("ImageCrop", imgCrop_combined)
                    cv2.imshow("ImageWhite", imgWhite)

                    current_time = time.time()
                    if num_hands == 2 and imgWhite is not None:
                        if current_time - last_save > delay:
                            counter += 1
                            filename = f'{folder}/Image_{time.time()}.jpg'
                            cv2.imwrite(filename, imgWhite)
                            last_save = current_time
                            print(f"✓ Auto-saved #{counter} (two-hand sign)")

                except Exception as e:
                    print(f"Error processing two hands: {e}")

    # Display hand count on screen
    cv2.putText(img, f'Hands: {num_hands}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f'Saved: {counter}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        if num_hands == 1 and imgWhite is not None:
            counter += 1
            filename = f'{folder}/Image_{time.time()}.jpg'
            cv2.imwrite(filename, imgWhite)
            print(f"✓ Saved #{counter} (one-hand sign)")
        else:
            print("⚠ Press 'S' only for ONE-hand signs")

    elif key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nTotal images saved: {counter}")
print(f"Saved to: {os.path.abspath(folder)}")