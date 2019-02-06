import numpy as np
import cv2
import os
from datetime import datetime
from pynput.mouse import Button, Controller

# za kontrolu misa
mouse = Controller()

# inicijalizacija kamere
cap = cv2.VideoCapture(0)

# ucitavanje haar cascade za lice
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# faza 0 dok se ne odredi koza - faza 1 pracenje
faze = 0
# timer za prelazak iz faza
last_detected = datetime.now()
# promenljiva koja pamti poslednju detekciju / detekcija se ne menja dok se bar ne drzi 0.2 sekunde
old_detection = 0
new_detection = 0

# inicijalna pozicija sa koje se uzima boja koze i odakle krece pracenje oko nje se formira 
handCenter = (500, 225)
palm_radius_history = []  # not implemented
avg_color_history = []  #implemented

# da li da doda racunanje maske preko histograma
hist = False
hist_mask = []

# funkcija za vracanje tacke na pocetak
def reset_to_start():
    return [(500,225), 0]

# inicijalizacija table za crtanje
board = np.zeros((480,640,3), np.uint8)

# dimenzija okvira od centra tacke u kojoj se ocekuje da bude ruka
radius3pola = 125
def surroundFrame(point, dist):
    w_r = (point[0]+dist,640)[point[0]+dist>640]
    w_l = (point[0]-dist,0)[point[0]-dist<0]
    h_t = (point[1]+dist,480)[point[1]+dist>480]
    h_b = (point[1]-dist,0)[point[1]-dist<0]
    return {'x':w_l, 'xw':w_r, 'y':h_b, 'yh':h_t}

# funkcija za racunanje srednje boje oko zadate tacke
def calculateSkin(frame):
    point = tuple(int(x/2) for x in frame.shape[:2])
    surr_frame = surroundFrame(point, 20)
    hsvtarget = frame[surr_frame['y']:surr_frame['yh'], surr_frame['x']:surr_frame['xw']]
    hsvtarget[hsvtarget>170] = 0
    avg_color = [np.mean(hsvtarget[:,:,0]), np.mean(hsvtarget[:,:,1])]
    # print(avg_color)

    # racunanje histograma na osnovu boje centra sake
    if hist:
        roihist = cv2.calcHist([hsvtarget],[0, 1], None, [180, 256], [0, 180, 0, 256] )
        cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
        hist_tresh = cv2.calcBackProject([frame],[0,1],roihist,[0,180,0,256],1)
        _, hist_tresh = cv2.threshold(hist_tresh, 10, 255, cv2.THRESH_BINARY)
        cv2.imshow("hist tresh", hist_tresh)
        return avg_color, hist_tresh

    return avg_color, []

while(True):
    # ucitavanje frejma i njegovo okretanje
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # uklanjanje lica
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x,y,w,h) in faces:
        frame[y:y+h, x:x+w] = 0

    # lokalizacija sake
    surr_frame = surroundFrame(handCenter, radius3pola)
    frameCroped = frame[surr_frame['y']:surr_frame['yh'], surr_frame['x']:surr_frame['xw']]
    
    # blurujemo pocetnu sliku da bi izgubili neke sumove i pretvorimo je u hsv
    blure = cv2.medianBlur(frameCroped, 15)
    blure = cv2.medianBlur(blure,7)

    edges = cv2.Canny(blure,25,75)
    kernel = np.zeros((3, 3), np.uint8)
    kernel[2, :] = 1
    edges = cv2.dilate(edges, kernel, iterations=2)
    cv2.imshow('edges', edges)

    # pretvorimo blurovanu sliku u HSV
    hsv = cv2.cvtColor(blure, cv2.COLOR_BGR2HSV)

    # iscrtavamo centar ruke / mesto odakle krece aplikacija
    cv2.circle(frame,handCenter,5,[0,0,255],-1)
    avg_color, hist_mask = calculateSkin(hsv)

    # ako je ukljuceno crtanje proveri da li je stara boja koze slicna onoj oko nove tacke ako nije doslo je do skoka
    if faze == 1 or faze == 2:
        diff = abs(avg_color_history[0]-avg_color[0])
        if diff > 20 and diff < 160:
            print(avg_color)
            print(avg_color_history)
            handCenter, faze = reset_to_start()
    avg_color_history = avg_color

    # pravimo masku na osnovu uzorka boje koze
    lower_skin = np.array([0,avg_color[1]-20,70])
    upper_skin = np.array([avg_color[0]+5,avg_color[1]+20,255])
    mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
    
    bug_lower = np.array([175,avg_color[1]-20,70])
    bug_upper = np.array([180,avg_color[1]+100,255])
    mask2 = cv2.inRange(hsv, bug_lower, bug_upper)

    mask = mask1 + mask2
    cv2.imshow('mask', mask)
    if len(hist_mask)!=0 and hist:
        mask = mask + hist_mask
        # cv2.imshow('mask1', mask)
    mask = cv2.medianBlur(mask, 7)
    cv2.imshow("maska2", mask)

    # trazi se najveca kontura u maski
    # ndajena kontura se uzima bez hijerarhije tako da ako su kojim slucajevima postojale
    # rupe unutar povrsine sake one ce biti popunjene
    # bitno zato sto distanceTransform je jako osetljiv na rupe unutar objekata
    # vracena nova maska sa najvecom konturom i bez rupa
    img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        handCenter, faze = reset_to_start()
        continue
    cntsSorted = sorted(contours, key=lambda x: -cv2.contourArea(x)) 
    mask = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(mask, [cntsSorted[0]], -1, (255,255,255), -1)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

    # trazi se najdublja tacka u konturi (tacka najudaljenija od ivica konture)
    # ta tacka se nalazi u centru dlana po morfologiji ruke
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    minV, maxV, minL, maxL = cv2.minMaxLoc(np.asarray(dist))
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

    # posto je pronadjena tacka centra dlana prelazimo u polarni kordinatni sistem
    # takodje deo gde se nalazi zglob i nadlaktica izbacimo, to jeste prvih 140 stepeni (odokativno)
    # sve u svemu nasa maska zapravo uzima podatke od -20 do 200 stepeni prvobitne polarno transformisane maske
    # slika se takodje skrati na pola da bi se izbacila potpuno dlan
    polar_img = cv2.linearPolar(mask, maxL, int(3*maxV), cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
    cv2.imshow('polar_img', polar_img)
    polar_img = polar_img[100:, :]
    cv2.imshow('polar_img_cut', polar_img)

    polar_widened = np.zeros([polar_img.shape[0]*2,polar_img.shape[1]])
    polar_widened[0::2] = polar_img
    polar_widened[1::2] = polar_img
    # polar_widened[2::3] = polar_img
    cv2.imshow('widend', polar_widened)
    kernel = np.zeros((1, polar_img.shape[1]), np.uint8)

    # vrsi se erozija da se prodube udubljenja izmedju prstiju, i da se odvoje prsti ako su bili spojeni
    kernel = np.zeros((7, 7), np.uint8)
    kernel[ 3, 3:] = 1

    x_shortend = polar_img.shape[1] - 75
    polar_img[:,:x_shortend] = cv2.erode(polar_img[:, :x_shortend], kernel, iterations=35)
    cv2.imshow('polar_img_cut_erode', polar_img)

    x_shortend = polar_img.shape[1] - 75
    polar_widened[:,:x_shortend] = cv2.erode(polar_widened[:, :x_shortend], kernel, iterations=40)
    cv2.imshow('polar_img_cut_erodefkjdfbdasf', polar_widened)

    fingers = polar_widened.sum(axis=1)
    fingers = np.ma.make_mask(fingers)
    polar_widened[fingers, :] = 1
    cv2.imshow('random', polar_widened)

    # traze se konture u maski i broje se, koliko kontura toliko prstiju na osnovu toga se menja boja
    _, contours, hierarchy = cv2.findContours(polar_img[:,20:100], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    number_of_fingers = len(contours)
    
    # app
    if faze == 0 and number_of_fingers != 4 and number_of_fingers != 5:
        cv2.putText(frame, str(number_of_fingers), (600,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,105), 1, cv2.LINE_AA)
        cv2.putText(frame, "Put 4 or 5 fingers hold for 2 sec to start.", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,105), 1, cv2.LINE_AA)
        last_detected = datetime.now()
    elif faze == 0 and number_of_fingers == 4:
        cv2.putText(frame, str((datetime.now() - last_detected).total_seconds()), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,105), 1, cv2.LINE_AA)
        if (datetime.now() - last_detected).total_seconds() > 2:
            faze = 1
            new_detection = 4
            old_detection = 4
    elif faze == 0 and number_of_fingers == 5:
        cv2.putText(frame, str((datetime.now() - last_detected).total_seconds()), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,105), 1, cv2.LINE_AA)
        if (datetime.now() - last_detected).total_seconds() > 2:
            faze = 2
            new_detection = 5
            old_detection = 5
    elif faze == 1 or faze == 2:
        # pozicija centra sake u odnosu na isceni deo kamere se se preracunava na poziciju na celoj kameri
        center_position = (surr_frame['x']+maxL[0], surr_frame['y']+maxL[1])

        # deo koji sprecava nagle skokove zbog suma, ako dodje do ostavi tacku gde jeste i proveri da li u njenoj okolini ima boje koze (to se radi gore)
        if abs(handCenter[0]-center_position[0]) <= 30 and abs(handCenter[1]-center_position[1]) <= 30:
            # deo za upravljanje misem
            if new_detection == 5 and faze == 2:
                mx = -(handCenter[0]-center_position[0])
                # mx = (((mx - (-30)) * (130 - (-130))) / (30 - (-30))) + (-130)
                my = -(handCenter[1]-center_position[1])
                # my = (((my - (-30)) * (130 - (-130))) / (30 - (-30))) + (-130)
                mouse.move(mx, my)
            if new_detection == 2 and old_detection != 2:
                mouse.click(Button.left, 2)
            handCenter = center_position

        # iscrtavanja krugova oko sake
        cv2.putText(frame, "Crtas sa 5 prstiju - crta se centrom dlana.", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,105), 1, cv2.LINE_AA)
        cv2.putText(frame, str(new_detection), (600,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,105), 1, cv2.LINE_AA)
        cv2.circle(frame,center_position,5,[0,255,0],-1)
        cv2.circle(frame,center_position,int(maxV),[0,255,0],1)
        cv2.circle(frame,center_position,int(3*maxV),[0,255,0],1)
        if new_detection == 5:
            cv2.circle(board,handCenter,5,[100,100,100],-1)

        if new_detection == 0:
            board = np.zeros((480,640,3), np.uint8)
        
        if number_of_fingers != old_detection:
            old_detection = number_of_fingers
            last_detected = datetime.now()

        if number_of_fingers != new_detection and (datetime.now() - last_detected).total_seconds() > 0.2:
            new_detection = number_of_fingers



  
    # cv2.imshow('frame',frame)
    cv2.imshow('together',np.hstack((frame,board)))
    blure_hsv = np.hstack((blure,hsv))
    cv2.imshow('blure_hsv',blure_hsv)
    mask_dist = np.hstack((mask,dist))
    cv2.imshow('mask_dist',mask_dist)
    key = cv2.waitKey(10) & 0b11111111 
    if key == ord('q'):
        break
    elif key == ord('h'):
        if hist:
            hist = False
        else:
            hist = True

cap.release()
cv2.destroyAllWindows()