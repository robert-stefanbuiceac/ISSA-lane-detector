import cv2
import numpy as np


last_left_top = (0, 0)
last_left_bottom = (0, 0)
last_right_top = (0, 0)
last_right_bottom = (0, 0)

cam = cv2.VideoCapture('Lane Detection Test Video 01.mp4')

while True:

    ret, frame = cam.read()

    # ret (bool): Return code of the `read` operation. Did we get an image or not?
    #             (if not maybe the camera is not detected/connected etc.)

    # frame (array): The actual frame as an array.
    #                Height x Width x 3 (3 colors, BGR) if color image.
    #                Height x Width if Grayscale
    #                Each element is 0-255.
    #                You can slice it, reassign elements to change pixels, etc.

    if not ret:
        break

    height,width=frame.shape[:2]
    new_height=int(height/3)
    new_width=int(width/3)

    resized_frame = cv2.resize(frame,(new_width,new_height))

    resized_frame=cv2.cvtColor(resized_frame,cv2.COLOR_BGR2GRAY)

    upper_left=(int(new_width*0.45),int(new_height*0.76))
    upper_right=(int(new_width*0.55),int(new_height*0.76))
    lower_left=(0,new_height)
    lower_right=(new_width,new_height)

    margini_trapez=np.array([upper_right,upper_left,lower_left,lower_right],dtype=np.int32)
    matrice_de_zero=np.zeros((new_height,new_width),dtype=np.uint8)
    cv2.fillConvexPoly(matrice_de_zero, margini_trapez, 1)

    drum=resized_frame*matrice_de_zero

    s_upper_left=(0,0)
    s_upper_right=(int(new_width),0)
    s_lower_left=(0,int(new_height))
    s_lower_right=(int(new_width),int(new_height))
    margini_ecran=np.array([s_upper_right,s_upper_left,s_lower_left,s_lower_right],dtype=np.int32)

    margini_trapez=np.float32(margini_trapez)
    margini_ecran=np.float32(margini_ecran)

    magic_matrix=cv2.getPerspectiveTransform(margini_trapez,margini_ecran)
    stretched_image=cv2.warpPerspective(drum,magic_matrix,(new_width,new_height))
    blurred_image=cv2.blur(stretched_image,(5,5))

    sobel_vertical=np.float32([[-1,-2,-1],
                              [0,0,0],
                              [+1,+2,+1]])
    sobel_horizontal=np.transpose(sobel_vertical)

    blurred_image=np.float32(blurred_image)
    filtru_v=cv2.filter2D(blurred_image,-1,sobel_vertical)
    filtru_h=cv2.filter2D(blurred_image,-1,sobel_horizontal)

    #frame_de_afisat_filtru_h=cv2.convertScaleAbs(filtru_h)
    #frame_de_afisat_filtru_v=cv2.convertScaleAbs(filtru_v)
    filtru_final=np.sqrt(filtru_h**2+filtru_v**2)
    filtru_de_afisat=cv2.convertScaleAbs(filtru_final)
    threshold = int(255 / 2)
    _, binarized_frame = cv2.threshold(filtru_de_afisat, threshold, 255, cv2.THRESH_BINARY)

    copie=binarized_frame.copy()
    copie[:,:(int(new_width*0.05))]=0
    copie[:,-(int(new_width*0.05)):]=0

    half_width=new_width//2
    left_half=copie[:,:half_width]
    right_half=copie[:,half_width:]

    left_points=np.argwhere(left_half>0)
    right_points=np.argwhere(right_half>0)

    left_xs = left_points[:, 1] if len(left_points) > 0 else np.array([])
    left_ys = left_points[:, 0] if len(left_points) > 0 else np.array([])
    right_xs = right_points[:, 1] + half_width if len(right_points) > 0 else np.array([])
    right_ys = right_points[:, 0] if len(right_points) > 0 else np.array([])

    if len(left_xs) > 0 and len(left_ys) > 0:
        dreapta_st = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)
        if dreapta_st[1] != 0:
            top_x = int((0 - dreapta_st[0]) / dreapta_st[1])
            bottom_x = int((new_height - dreapta_st[0]) / dreapta_st[1])
            if 0 <= top_x <= half_width and 0 <= bottom_x <= half_width:
                last_left_top = (top_x, 0)
                last_left_bottom = (bottom_x, new_height)

    if len(right_xs) > 0 and len(right_ys) > 0:
        dreapta_dr = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)
        if dreapta_dr[1] != 0:
            top_x = int((0 - dreapta_dr[0]) / dreapta_dr[1])
            bottom_x = int((new_height - dreapta_dr[0]) / dreapta_dr[1])
            if half_width <= top_x <= new_width and half_width <= bottom_x <= new_width:
                last_right_top = (top_x, 0)
                last_right_bottom = (bottom_x, new_height)

    cv2.line(binarized_frame,last_left_top,last_left_bottom,(100,0,0),5)
    cv2.line(binarized_frame,last_right_top,last_right_bottom,(200,0,0),5)

    cv2.imshow('Original', binarized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()