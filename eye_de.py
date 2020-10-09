import cv2
import numpy as np
import dlib
from keras.models import load_model
from scipy.spatial import distance as dist
from imutils import face_utils
import pyautogui
import tensorflow as tf


# 중간 좌표 계산
def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


# 최대 좌표 계산
def maxpoint(p1, p2):
    return (int(p1.x if p1.x > p2.x else p2.x), int(p1.y if p1.y > p2.y else p2.y))


# 최소 좌표 계산
def minpoint(p1, p2):
    return (int(p1.x if p1.x < p2.x else p2.x), int(p1.y if p1.y < p2.y else p2.y))


# 최종 좌표의 변화량을 완화하기 위한 평균값 산출
def get_mem_avg(arr):
    x = sum_total = 0
    for i in range(len(arr)):
        x = x + (i + 1)
        sum_total = sum_total + arr[i] * (i + 1)
    return sum_total / x


# 동공 contour에 따른 thresh 자동 조정
def auto_thresh(cnt, radius, drt, thr, w):
    minCircleArea = radius * radius * 3.141592
    blobArea = cv2.contourArea(cnt)

    if (blobArea / minCircleArea < 0.40):
        if (drt == "left"):
            if (landmarks.part(45).x - landmarks.part(42).x > 2 * w - 10):
                thr = thr + 3
            elif (landmarks.part(45).x - landmarks.part(42).x < 2 * w):
                thr = thr - 3
        else:  # "right"
            if (landmarks.part(39).x - landmarks.part(36).x > 2 * w - 10):
                thr = thr + 3
            elif (landmarks.part(39).x - landmarks.part(36).x < 2 * w):
                thr = thr - 3
    if (thr == 0):
        thr = 40

    return thr


# 동공 contour에 따른 밝기(동공으로 인식하는 명도 값) 자동 조정
def auto_brightness(roi):
    # avg_brt should be more than 0.35 and less than 0.70
    cols, rows = roi.shape
    brightness = np.sum(roi) / (255 * cols * rows)

    minimum_brightness = 0.45
    maximum_brightness = 0.70

    ratio = brightness / minimum_brightness
    if ratio <= 1:
        roi = cv2.convertScaleAbs(roi, alpha=1 / ratio, beta=0)

    ratio = brightness / maximum_brightness
    if ratio >= 1:
        roi = cv2.convertScaleAbs(roi, alpha=1 / ratio, beta=0)

    return roi


# 동공 인식 및 표시
def pupil(roi, thr, drt):
    global opt
    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.bilateralFilter(gray_roi, 10, 75, 75)
    gray_roi = auto_brightness(gray_roi)

    _, threshold = cv2.threshold(gray_roi, thr, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    '''
    if len(contours) == 0:
        if(thr > 130):
            thr = 60
        else:
            thr = thr + 5
     '''
    eye_x = eye_y = 0
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))

        cx = x
        cy = y
        M = cv2.moments(cnt)
        if (M['m00'] != 0):
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.line(roi, (int(cx), 0), (int(cx), rows), (255, 255, 255), 1)
            cv2.line(roi, (0, int(cy)), (cols, int(cy)), (255, 255, 255), 1)

        eye_x = (x + cx) / 2
        eye_y = (y + cy) / 2
        cv2.circle(roi, center, int(radius), (255, 0, 0), 1)
        cv2.line(roi, (int(x), 0), (int(x), rows), (0, 255, 0), 1)
        cv2.line(roi, (0, int(y)), (cols, int(y)), (0, 255, 0), 1)

        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 1)

        if (opt == 0):
            if (auto_toggle == 1):
                thr = auto_thresh(cnt, radius, drt, thr, w)

        break
    return roi, thr, int(eye_x), int(eye_y);


# 초기 값 설정 UI
def showimg(num):
    if (num > 0):
        cv2.destroyWindow("showimg")

    showimg = cv2.imread("showimg.png", cv2.IMREAD_COLOR)
    cv2.namedWindow("showimg", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("showimg", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.putText(showimg, "look at number " + str(num) + " then press enter", (int(sw / 6), int(sh / 3)),
                cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("showimg", showimg)

    if (num == 4):
        cv2.destroyWindow("showimg")


# 눈 깜빡임 감지
def eye_blink_detection(left_roi, right_roi):
    global close_counter, blinks, mem_counter, state, state_tag

    left_eye_image = cv2.cvtColor(left_roi, cv2.COLOR_BGR2GRAY)
    right_eye_image = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY)

    left_eye_image = cv2.resize(left_eye_image, (34, 26))
    right_eye_image = cv2.resize(right_eye_image, (34, 26))
    right_eye_image = cv2.flip(right_eye_image, 1)

    l_prediction = model.predict(cnnPreprocess(left_eye_image))
    r_prediction = model.predict(cnnPreprocess(right_eye_image))

    print(l_prediction, r_prediction)

    prec = 0.005
    if r_prediction < prec and l_prediction < prec:
        state = 'blink'
        close_counter = 0
    elif l_prediction < prec:
        state = 'left close'
        state_tag = 'left'
        close_counter += 1
    elif r_prediction < prec:
        state = 'right close'
        state_tag = 'right'
        close_counter += 1
    elif l_prediction > 0.1 and r_prediction > 0.1:
        state = 'open'
        close_counter = 0

    if state == 'open' and mem_counter > 2:
        if state_tag == 'left':
            pyautogui.click()
        if state_tag == 'right':
            pyautogui.rightClick()
        blinks += 1

    mem_counter = close_counter

    cv2.putText(frame, "Blinks: {}".format(blinks), (400, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Counter: {}".format(close_counter), (400, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "State: {}".format(state), (400, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return


# 얼굴 좌우상하 움직임 감지
def head_flick_detection():
    global f_state, f_state_tag, f_counter, mem_f_counter, flicks

    nose = landmarks.part(30)
    f_height = landmarks.part(8).y - landmarks.part(27).y
    f_width = landmarks.part(14).x - landmarks.part(2).x

    up = landmarks.part(30).y - landmarks.part(27).y
    down = landmarks.part(8).y - landmarks.part(30).y

    up_ratio = up / f_height
    down_ratio = down / f_height

    left = landmarks.part(14).x - landmarks.part(30).x
    right = landmarks.part(30).x - landmarks.part(2).x

    left_ratio = left / f_width
    right_ratio = right / f_width

    if left_ratio < 0.3:
        f_state = 'left'
        f_state_tag = 'back'
        f_counter += 1
    elif right_ratio < 0.3:
        f_state = 'right'
        f_state_tag = 'next'
        f_counter += 1
    elif not (f_state == 'left' or f_state == 'right') and up_ratio < head_up_prec:
        f_state = 'up'
        f_state_tag = 'scrollup'
        f_counter += 1
    elif not (f_state == 'left' or f_state == 'right') and down_ratio < head_down_prec:
        f_state = 'down'
        f_state_tag = 'scrolldown'
        f_counter += 1
    elif left_ratio > 0.4 and right_ratio > 0.4:
        f_state = 'still'
        f_counter = 0
    if mem_f_counter > 1:
        if f_state == 'still' and mem_f_counter < 10:
            if f_state_tag == 'back':
                pyautogui.hotkey('alt', 'left')
            if f_state_tag == 'next':
                pyautogui.hotkey('alt', 'right')
            flicks += 1
        else:
            if f_state_tag == 'scrollup':
                pyautogui.scroll(100)
            if f_state_tag == 'scrolldown':
                pyautogui.scroll(-100)
    mem_f_counter = f_counter

    cv2.putText(frame, "Flinks: {}".format(flicks), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Counter: {}".format(f_counter), (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "State: {}".format(f_state), (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return


# CNN 처리 단계
def cnnPreprocess(img):
    img = img.astype('float32')
    img /= 255
    img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)
    return img


''' ↑    function    ↑ '''
''' ↓      main      ↓ '''

close_counter = blinks = mem_counter = 0
state = 'open'
state_tag = ''
f_state = 'still'
f_state_tag = ''
f_counter = mem_f_counter = flicks = 0
mem_x_arr = []
mem_y_arr = []
mem_x_sum = mem_y_sum = 0
detect_motion = 0
head_up_prec = 0.3
head_down_prec = 0.6

blank = np.zeros((500, 500, 3), np.uint8)

right_thr = left_thr = 30  # set basic thresh
left_eye_list = []
right_eye_list = []
auto_toggle = 0
setnum = 0
pyautogui.FAILSAFE = False
sw, sh = pyautogui.size()
sx = sw / 2
sy = sh / 2

opt = 1  # optimizing per n times loop

box_x1 = box_y1 = box_x2 = box_y2 = box_x3 = box_y3 = box_x4 = box_y4 = 0
box_x = box_y = box_w = box_h = 0
input_x = input_y = 0

overlay_toggle = 0
overlay = cv2.imread('overlay.png')

cap = cv2.VideoCapture(1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

model = tf.keras.models.load_model('./blinkModel.hdf5', custom_objects=None, compile=True)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    opt = (opt + 1) % 21

    faces = detector(gray)
    if (len(faces) == 0):
        continue
    num = index = max_area = 0
    for face in faces:
        if (face.area() > max_area):
            max_area = face.area()
            index = num
        num = num + 1

    num = 0
    for temp in faces:
        face = temp
        if (num == index):
            break
        num = num + 1

    global landmarks
    landmarks = predictor(gray, face)

    right_roi = frame[minpoint(landmarks.part(36), landmarks.part(39))[1] - 10:
                      maxpoint(landmarks.part(36), landmarks.part(39))[1] + 10,
                landmarks.part(36).x - 5: landmarks.part(39).x + 5]
    left_roi = frame[minpoint(landmarks.part(42), landmarks.part(45))[1] - 10:
                     maxpoint(landmarks.part(42), landmarks.part(45))[1] + 10,
               landmarks.part(42).x - 5: landmarks.part(45).x + 5]

    if detect_motion == 1:
        eye_blink_detection(left_roi, right_roi)
        head_flick_detection()

    magnify = 10
    rows, cols, _ = right_roi.shape
    right_roi = cv2.resize(right_roi, (int(cols * magnify), int(rows * magnify)))
    cv2.putText(right_roi, "Right", (0, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(right_roi, "thr = " + str(right_thr), (0, int(cols * 1.5)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255),
                2, cv2.LINE_AA)

    rows, cols, _ = left_roi.shape
    left_roi = cv2.resize(left_roi, (int(cols * magnify), int(rows * magnify)))
    cv2.putText(left_roi, "Left", (0, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(left_roi, "thr = " + str(left_thr), (0, int(cols * 1.5)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2,
                cv2.LINE_AA)

    temp = right_thr
    right_roi, right_thr, right_eye_x, right_eye_y = pupil(right_roi, right_thr, "right")

    temp = left_thr
    left_roi, left_thr, left_eye_x, left_eye_y = pupil(left_roi, left_thr, "left")

    if len(mem_x_arr) >= 5:
        mem_x_arr.pop(0)
    mem_x_arr.append(int((right_eye_x + left_eye_x) / 2))
    if len(mem_y_arr) >= 5:
        mem_y_arr.pop(0)
    mem_y_arr.append(int((right_eye_y + left_eye_y) / 2))

    cv2.namedWindow("right_Eye")
    cv2.moveWindow("right_Eye", 1000, 30)

    cv2.namedWindow("left_Eye")
    cv2.moveWindow("left_Eye", 300, 30)

    cv2.imshow("right_Eye", right_roi)
    cv2.imshow("left_Eye", left_roi)

    cv2.imshow("setting", blank)

    if overlay_toggle == 1:
        frame = cv2.addWeighted(frame, 1, overlay, 0.5, 0)
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  # esc
        break
    if key == 113:  # q : 왼쪽 눈 임계값 감소
        left_thr = left_thr - 2
    if key == 119:  # w : 왼쪽 눈 임계값 증가
        left_thr = left_thr + 2
    if key == 91:  # [ : 오른쪽 눈 임계값 감소
        right_thr = right_thr - 2
    if key == 93:  # ] : 오른쪽 눈 임계값 증가
        right_thr = right_thr + 2
    if key == 112:  # p : pause
        key2 = cv2.waitKey(0)
    if key == ord('u'):
        if auto_toggle == 0:
            auto_toggle = 1
        else:
            auto_toggle = 0
    if key == 114:  # r : re-init setting
        right_eye_list = []
        left_eye_list = []
        showimg(0)
        setnum = 0
    if key == 13:  # enter
        if (setnum < 4):
            right_eye_list.append([right_eye_x, right_eye_y])
            left_eye_list.append([left_eye_x, left_eye_y])
            setnum = setnum + 1
            showimg(setnum)

            if (setnum == 4):
                for i in range(setnum):
                    print("right_eye[", i, "](x,y) : ", right_eye_list[i])

                print("")
                for i in range(setnum):
                    print("left_eye[", i, "](x,y) : ", left_eye_list[i])
    ### 동공 좌표 디버깅
    if key == ord('s'):  # s : 눈 좌표 표시
        cv2.circle(blank, (int((right_eye_x + left_eye_x) / 2), int((right_eye_y + left_eye_y) / 2)), 3, (255, 255, 0),
                   -1)
    if key == ord('z'):  # z : 상 좌표 저장
        box_x1 = (right_eye_x + left_eye_x) / 2
        box_y1 = (right_eye_y + left_eye_y) / 2
        cv2.circle(blank, (int((right_eye_x + left_eye_x) / 2), int((right_eye_y + left_eye_y) / 2)), 3, (255, 0, 0),
                   -1)
    if key == ord('x'):  # x : 하 좌표 저장
        box_x2 = (right_eye_x + left_eye_x) / 2
        box_y2 = (right_eye_y + left_eye_y) / 2
        cv2.circle(blank, (int((right_eye_x + left_eye_x) / 2), int((right_eye_y + left_eye_y) / 2)), 3, (255, 0, 0),
                   -1)
    if key == ord('c'):  # c : 우 좌표 저장
        box_x3 = (right_eye_x + left_eye_x) / 2
        box_y3 = (right_eye_y + left_eye_y) / 2
        cv2.circle(blank, (int((right_eye_x + left_eye_x) / 2), int((right_eye_y + left_eye_y) / 2)), 3, (255, 0, 0),
                   -1)
    if key == ord('v'):  # v : 좌 좌표 저장
        box_x4 = (right_eye_x + left_eye_x) / 2
        box_y4 = (right_eye_y + left_eye_y) / 2
        cv2.circle(blank, (int((right_eye_x + left_eye_x) / 2), int((right_eye_y + left_eye_y) / 2)), 3, (255, 0, 0),
                   -1)
    if key == ord('a'):  # a : 눈 box 설정
        box_x = box_x4 - box_x3
        box_y = box_y2 - box_y1
        cv2.rectangle(blank, (int(box_x3), int(box_y1)), (int(box_x4), int(box_y2)), (0, 255, 255), 2)
    if key == ord('d'):  # d : x, y 마우스 이동
        ratio_x = (get_mem_avg(mem_x_arr) - box_x3) / box_x
        ratio_y = (get_mem_avg(mem_y_arr) - box_y1) / box_y
        print("x, y=[{} : {}]".format(round(ratio_x, 2), round(ratio_y, 2)))
        pyautogui.moveTo(sw - ratio_x * sw, ratio_y * sh, 0.5)
    if key == ord('f'):  # f : x 마우스 이동
        ratio_x = (get_mem_avg(mem_x_arr) - box_x3) / box_x
        print("x={}".format(round(ratio_x, 2)))
        pyautogui.moveTo(sw - ratio_x * sw, 540, 0.5)
    if key == ord('g'):  # g : y 마우스 이동
        ratio_y = (get_mem_avg(mem_y_arr) - box_y1) / box_y
        print("y={}".format(round(ratio_y, 2)))
        pyautogui.moveTo(960, ratio_y * sh, 0.5)
    if key == ord('t'):  # t : 필드 리셋
        blank = np.zeros((500, 500, 3), np.uint8)
    if key == ord('i'):  # t : 필드 리셋
        if detect_motion == 0:
            detect_motion = 1
        else:
            detect_motion = 0
    if key == ord('o'):  # t : 필드 리셋
        if overlay_toggle == 0:
            overlay_toggle = 1
        else:
            overlay_toggle = 0
    if key == ord('='):
        head_up_prec = head_up_prec + 0.01
    if key == ord('-'):
        head_up_prec = head_up_prec - 0.01
    if key == ord('0'):
        head_down_prec = head_down_prec + 0.01
    if key == ord('9'):
        head_down_prec = head_down_prec - 0.01

cap.release()
cv2.destroyAllWindows()