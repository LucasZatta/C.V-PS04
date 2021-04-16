import numpy as np
import cv2
cap = cv2.VideoCapture('freeway.mp4')
video_mode = False
print("Choose the app mode:")
print("1: Image Mode Tracing")
print("2: Real Time Video Mode Tracing")
x = int(input())
if x == 2:
    video_mode = True

feature_params = dict(maxCorners=100,
                      qualityLevel=0.1,
                      minDistance=7,
                      blockSize=7)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
generate_image = 0
while(1):
    ret, frame = cap.read()
    if generate_image == 0 and video_mode == False:
        cv2.imwrite('initial_frame.png', frame)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), (255, 255, 255), 2)
        frame = cv2.circle(frame, (a, b), 1, (255, 255, 255), -1)
    img = cv2.add(frame, mask)
    if video_mode == True:
        cv2.imshow('frame', img)
    generate_image += 1
    if generate_image == 24 and video_mode == False:
        cv2.imwrite('final_frame.png', img)
        cv2.imshow('Motion Capture', img)
        cv2.waitKey()
        break
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
cv2.destroyAllWindows()
cap.release()
