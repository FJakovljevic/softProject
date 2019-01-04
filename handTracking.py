import numpy as np
import cv2
import os

#funkcija koja vraca prosecnu boju sa slike iz datog kvadrata
def calculateSkin(frame):
    hsvtarget = frame[100:150, 100:150]
    avg_color_per_row = np.mean(hsvtarget, axis=0)
    avg_color = np.mean(avg_color_per_row, axis=0)
    print(avg_color)
    return [1, avg_color]

# funkcije zaduzene da iseceni kvadrat bude u okvirima kamere
def toRangeW(x):
    if x > 640:
        return 640
    elif x < 0:
        return 0
    else:
        return x

def toRangeH(x):
    if x > 480:
        return 480
    elif x < 0:
        return 0
    else:
        return x

# inicijalizacija kamere
cap = cv2.VideoCapture(0)
# ucitavanje haar cascade za lice
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# faza 0 pre odredjivanja boje koze pritiskom na tipku d prelazi se na fazu 1 pracenja ruke
faze = 0

# inicijalna pozicija sa koje se uzima boja koze i odakle krece pracenje oko nje se formira 
# kvarat dimenzija 250x250
handCenter = (525, 225)
radius3pola = 125

# promenljiva za boje
# 2 - crveno
# 3 - zuto
# 4 - zeleno
color = [0,150,150]
def colorChoose(x, color):
    if x == 2:
        return [0,0,200]
    elif x == 3:
        return [0,150,150]
    elif x == 4:
        return [0,200,0]
    else:
        return color

# inicijalizacija table za crtanje
board = np.zeros((480,640,3), np.uint8)

while(True):
    # ucitavanje frejma i njegovo okretanje
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # uklanjanje lica
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x,y,w,h) in faces:
        frame[y:y+h, x:x+w] = 0

    # lokalizacija sake
    y = toRangeH(handCenter[1]-radius3pola)
    yh = toRangeH(handCenter[1]+radius3pola)
    x = toRangeW(handCenter[0]-radius3pola)
    xw = toRangeW(handCenter[0]+radius3pola)
    frameCroped = frame[int(y):int(yh), int(x):int(xw)]
    
    # blurujemo pocetnu sliku da bi izgubili neke sumove i pretvorimo je u hsv
    blure = cv2.medianBlur(frameCroped, 15)
    # blure = cv2.medianBlur(blure,13)
    blure = cv2.medianBlur(blure,7)

    # pretvorimo blurovanu sliku u HSV
    hsv = cv2.cvtColor(blure, cv2.COLOR_BGR2HSV)

    if faze==0:
        cv2.rectangle(frame, (498, 198), (552, 252), (0, 0, 255), 2)
    else:
        # pravimo masku na osnovu uzorka boje koze
        lower_skin = np.array([0,70.1,70])
        upper_skin = np.array([avg_color[0]+5,255,255])
        mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
        
        bug_lower = np.array([170,avg_color[1]-20,100])
        bug_upper = np.array([180,avg_color[1]+100,255])
        mask2 = cv2.inRange(hsv, bug_lower, bug_upper)

        mask = mask1 #+ mask2
        # mask = cv2.medianBlur(mask, 17)
        # mask = cv2.medianBlur(mask, 11)     
        mask = cv2.medianBlur(mask, 7)

        # trazi se najveca kontura u maski
        # ndajena kontura se uzima bez hijerarhije tako da ako su kojim slucajevima postojale
        # rupe unutar povrsine sake one ce biti popunjene
        # bitno zato sto distanceTransform je jako osetljiv na rupe unutar objekata
        # vracena nova maska sa najvecom konturom i bez rupa
        img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntsSorted = sorted(contours, key=lambda x: -cv2.contourArea(x)) 
        mask = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(mask, [cntsSorted[0]], -1, (255,255,255), -1)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        cv2.imshow('mask', mask)

        # trazi se najdublja tacka u konturi (tacka najudaljenija od ivica konture)
        # ta tacka se nalazi u centru dlana po morfologiji ruke
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        minV, maxV, minL, maxL = cv2.minMaxLoc(np.asarray(dist))

        # posto je pronadjena tacka centra dlana prelazimo u polarni kordinatni sistem
        # kako kako se obicno palac nalazi na 0 stepeni dobijenu polarnu masku transliramo za nekih 21,6 stepeni
        # da bi cela ruka bila spojena
        # takodje deo gde se nalazi zglob i nadlaktica izbacimo, to jeste prvih 140 stepeni (odokativno)
        # sve u svemu nasa maska zapravo uzima podatke od -20 do 200 stepeni prvobitne polarno transformisane maske
        # slika se takodje skrati na pola da bi se izbacila potpuno dlan
        newImg = cv2.linearPolar(mask, maxL, int(3*maxV), cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
        cv2.imshow('newImg', newImg[:, 125:])
        image = np.vstack((newImg[15:, 117:], newImg[:15, 117:]))
        image[:100, :] = 0
        
        # vrsi se erozija da se prodube udubljenja izmedju prstiju, i da se odvoje prsti ako su bili spojeni
        kernel = np.zeros((7, 7), np.uint8)
        kernel[ 3, 3:] = 1
        image = cv2.erode(image, kernel, iterations=13)
        cv2.imshow('deNewImage', image)

        # traze se konture u maski i broje se, koliko kontura toliko prstiju na osnovu toga se menja boja
        _, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.putText(frame, str(len(contours)), (600,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,105), 1, cv2.LINE_AA)
        color = colorChoose(len(contours), color)

        # pozicija centra sake u odnosu na isceni deo kamere se se preracunava na poziciju na celoj kameri
        handCenter = (x+maxL[0], y+maxL[1])

        # deo za crtanje po tabli ako se ne drzi ispruzen ni jedan prst onda se iscrtavaju tacke
        if len(contours) == 0:
             cv2.circle(board,handCenter,5,color,-1)
        cv2.imshow('board', board)
        
        # iscrtavanje informacija na ekranu
        maxL = (x+maxL[0], y+maxL[1])
        cv2.circle(frame,maxL,5,[0,0,255],-1)
        cv2.circle(frame,maxL,int(maxV),[0,0,255],1)
        cv2.circle(frame,maxL,int(3*maxV),[0,0,255],1)
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        cv2.imshow('Distance Transform Image', dist)
        
    cv2.imshow('frame',frame)
    cv2.imshow('blure',blure)
    cv2.imshow('hsv',hsv)
    key = cv2.waitKey(27) & 0b11111111 
    if key == ord('q'):
        break
    elif key == ord('d'):
        faze, avg_color = calculateSkin(hsv)

cap.release()
cv2.destroyAllWindows()