# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 10:56:06 2018

@author: DAVIAUD
"""
import cv2
import imutils
import numpy as np
import re
import pandas as pd
import datetime as dt 
import os, os.path
from tqdm import tqdm
import fitz
import math
import pytesseract
# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = r'D:\Profiles\daviaud\Documents\Travail\Chantier Data analytics\Python modules\Tesseract-OCR\tesseract'

now = dt.datetime.now()

def load_images_from_file(path, file):
    images = {}
    valid_images = [".jpg",".jpeg",".png"]
    pdf = [".pdf"]
    ext = os.path.splitext(file)[1]
    if ext.lower() in pdf:
        images.update(extract_images_from_pdf(path,file))
    elif ext.lower() in valid_images:
        images.update(read_image(path,file))       
    return images

def read_image(path, filename):
    try:
        image = cv2.imread(os.path.join(path,filename),0)
        return {filename:image}
    except:
        return {filename:"Erreur : image illisible"}

def extract_images_from_pdf(path, filename):
    newpath = path + "images_from_pdf"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    PNGs = {}
    try:
        doc = fitz.open(os.path.join(path,filename))
        for i in range(len(doc)):
            imgs = doc.getPageImageList(i)
            if len(imgs) == 1:
                for img in imgs:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    try:       # this is GRAY or RGB
                        pix.writePNG(os.path.join(newpath,filename+"p%s.png" % i))
                    except:               # CMYK: convert to RGB first
                        pix1 = fitz.Pixmap(fitz.csRGB, pix)
                        pix1.writePNG(os.path.join(newpath,filename+"p%s.png" % i))
                        pix1 = None
            else:
                page=doc[i]
                pix = page.getPixmap(colorspace = fitz.csGRAY)
                try:       # this is GRAY or RGB
                    pix.writePNG(os.path.join(newpath,filename+"p%s.png" % i))
                except:               # CMYK: convert to RGB first
                    pix1 = fitz.Pixmap(fitz.csRGB, pix)
                    pix1.writePNG(os.path.join(newpath,filename+"p%s.png" % i))
                    pix1 = None
            png = read_image(newpath,filename+"p%s.png" % i)
            PNGs.update(png)
            pix = None
    except:
        PNGs = {filename:"Erreur : PDF non convertible"}
    return PNGs 

def resize_img(image, size, side):
    height, width = image.shape[:2]
    if side == "h":
        factor = size/height
    elif side == "w":
        factor = size/width
    else:
        factor = 1
    image = cv2.resize(image, (int(math.ceil(factor*width)), int(math.ceil(factor*height))), interpolation = cv2.INTER_CUBIC) 
    return (image, factor)

def auto_canny(image, sigma=0.33):
	v = np.median(image)
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	return edged

def rotate(image):
    try:
        (h0, w0) = image.shape[:2]
        if h0 == min(h0, w0):
            side = "h"
        else:
            side = "w"
        kernel = np.ones((3,3),np.uint8)
        img = image.copy()
        (img, factor) = resize_img(image, 400, side)
        height, width = img.shape[:2]
        edges = auto_canny(img)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=width / 2.0, maxLineGap=20)
        angles = np.array([])
        for x in range(0, len(lines)):
            angle = 0.0
            for x1,y1,x2,y2 in lines[x]:
                angle += np.arctan2(y2 - y1, x2 - x1)
                angles = np.append(angles, angle)
        angle_median = np.median(angles) * 180 / np.pi
        rotated = imutils.rotate_bound(image, -angle_median)
        return rotated
    except:
        return image

def rotate_image_and_coord(image, angle, startingPoint):
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    rotated_image = cv2.warpAffine(image, M, (nW, nH))
    rotated_startingPoints = cv2.transform(startingPoint, M)
    rotation = [rotated_image, rotated_startingPoints]
    return rotation

def show(titre, image):
    cv2.imshow(titre,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def get_Contour_Properties(contour, factor):
    (x_cnt, y_cnt, w_cnt, h_cnt) = cv2.boundingRect(contour)
    approx = cv2.approxPolyDP(contour,0.02*cv2.arcLength(contour,True),True)
    rect = cv2.minAreaRect(contour)
    skew = rect[2]
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    Xa = box[0][0]
    Ya = box[0][1]
    Xb = box[1][0]
    Yb = box[1][1]
    Xc = box[2][0]
    Yc = box[2][1]
    Xd = box[3][0]
    Yd = box[3][1]
    boundingRect = [x_cnt, y_cnt, w_cnt, h_cnt]
    cnt_points = [[int(round(Xa/factor)), int(round(Ya/factor))], [int(round(Xb/factor)), int(round(Yb/factor))], [int(round(Xc/factor)), int(round(Yc/factor))], [int(round(Xd/factor)), int(round(Yd/factor))]]
    L1 = math.sqrt(math.pow((Xa-Xb),2)+math.pow((Ya-Yb),2))
    L2 = math.sqrt(math.pow((Xc-Xd),2)+math.pow((Yc-Yd),2))
    l1 = math.sqrt(math.pow((Xa-Xd),2)+math.pow((Ya-Yd),2))
    l2 = math.sqrt(math.pow((Xb-Xc),2)+math.pow((Yb-Yc),2))
    min_cote = min(L1, L2, l1, l2)
    max_cote = max(L1, L2, l1, l2)
    perimeter = (2*min_cote)+(2*max_cote)
    if h_cnt > w_cnt:
        if abs(skew) > 45:
            skew_cnt = max(abs(skew), abs(90+skew))
        else:
            skew_cnt = -1 * max(abs(skew), abs(90+skew))
    else:
        if abs(skew) > 45:
            skew_cnt = -1 * min(abs(skew), abs(90+skew))
        else:
            skew_cnt = min(abs(skew), abs(90+skew))
    contour_properties = [boundingRect, cnt_points, min_cote, max_cote, perimeter, len(approx), skew_cnt, factor]
    return contour_properties

def is_Similar(element1, element2):
    is_similar = False
    (x1, y1, w1, h1) = element1[0]
    (x2, y2, w2, h2) = element2[0]
    x1 = x1/element1[7]
    y1 = y1/element1[7]
    x2 = x2/element2[7]
    y2 = y2/element2[7]
    espace = (50/element1[7])
    if (abs(x1 - x2) < espace) and (abs(y1 - y2) < espace):
        is_similar = True
    return is_similar

def extract_contour(image, contour_to_extract, detected_bande):
    (x_cnt, y_cnt, w_cnt, h_cnt) = contour_to_extract[0]
    factor = contour_to_extract[7]
    w_cnt = int(math.ceil(w_cnt/factor))
    h_cnt = int(math.ceil(h_cnt/factor))
    startingPoints = np.array([contour_to_extract[1]])
    min_cote = int(math.ceil(contour_to_extract[2]/factor))
    max_cote = int(math.ceil(contour_to_extract[3]/factor))
    (rotated_image, rotated_startingPoints) = rotate_image_and_coord(image, contour_to_extract[6], startingPoints)
    (height, width) = rotated_image.shape[:2]
    paddingX = 2*int(math.ceil(max_cote/36))
    if detected_bande == 1:
        paddingY = 5*int(math.ceil(10/factor))
    elif detected_bande == 2:
        paddingY = int(math.ceil(min_cote/2))
    X = max(min(rotated_startingPoints[0][0][0], rotated_startingPoints[0][1][0], rotated_startingPoints[0][2][0], rotated_startingPoints[0][3][0]), 0)
    Y = max(min(rotated_startingPoints[0][0][1], rotated_startingPoints[0][1][1], rotated_startingPoints[0][2][1], rotated_startingPoints[0][3][1]), 0)
    startX = max(X - paddingX, 0)
    endX = min(X + max_cote + paddingX, width)
    startY = max(Y - paddingY, 0)
    endY = min(Y + min_cote + paddingY, height)
    roi = rotated_image[startY:endY, startX:endX].copy()
    return roi   

def prep_MRZ(roi):
    kernel2 = np.ones((2,2),np.uint8)
    kernel3 = np.ones((3,3),np.uint8)
    roi = cv2.fastNlMeansDenoising(roi, roi, 7, 14)     
    (h_roi,w_roi) = roi.shape[:2]
    if w_roi < 400:
        (roi, factor) = resize_img(roi, 800, "w")
        roi = cv2.adaptiveThreshold(roi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,2)
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel3)
        roi = cv2.dilate(roi, kernel2, iterations = 1)
    elif w_roi < 700:
        (roi, factor) = resize_img(roi, 800, "w")
        roi = cv2.adaptiveThreshold(roi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2)
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel3) 
    show('roi', roi)
    return roi

def read_MRZ(image):
    lang = 'ocrb'
    text = ''
    tessdata_dir_config = "--psm 11 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<> -c load_system_dawg=F -c load_freq_dawg=F -c load_system_dawg=F -c load_freq_dawg=F -c load_punc_dawg=F -c load_number_dawg=F -c load_unambig_dawg=F -c load_bigram_dawg=F -c load_fixed_length_dawgs=F"
    try:
        text = pytesseract.image_to_string(image,lang=lang,config=tessdata_dir_config)
        if '>>' in text:
            image = imutils.rotate_bound(image, angle=-180)
            text = pytesseract.image_to_string(image,lang=lang,config=tessdata_dir_config)
    except:
        text = 'Aucune bande MRZ détéctée'
    return text 
     
def read_if_MRZ_and_new(liste_MRZ, contour, img, factor, filename):
    detected_bande = 0
    (h_img, w_img) = img.shape[:2]
    h_img = int(round(h_img*factor))
    w_img = int(round(w_img*factor))
    new_contour = get_Contour_Properties(contour, factor)
    if (new_contour[2]/max(new_contour[3], 1) < 0.025) and (new_contour[2]/max(new_contour[3], 1) > 0.005) and (new_contour[5] == 2) and (new_contour[3]/min(h_img, w_img) > 0.35) and (new_contour[4] > 700):
        detected_bande = 1
    elif (new_contour[2]/max(new_contour[3], 1) < 0.15) and (new_contour[2]/max(new_contour[3], 1) > 0.05) and (new_contour[5] == 4) and (new_contour[3]/min(h_img, w_img) > 0.35) and (new_contour[4] > 700):
        detected_bande = 2
    if detected_bande > 0:
        is_New = True
        for element in liste_MRZ:
            if is_Similar(element[1], new_contour):
                is_New = False
                break
        if is_New == True:
            roi = extract_contour(img, new_contour, detected_bande)
            prepared_roi = prep_MRZ(roi)
            texte_MRZ = read_MRZ(prepared_roi)
            if '<<' in texte_MRZ:
                print(texte_MRZ)
                result = process_MRZ(texte_MRZ)
                if result is not None:
                    info_MRZ = {'filename':filename, 'resultat':'Bande MRZ détéctée', 'type':result['type'], 'code':result['code'], 'nom':result['nom'], 'prenom':result['prenom'], 'date_naissance':result['date_naissance'], 'nationalite':result['nationalite'], 'sexe':result['sexe'], 'date_validite':result['date_validite'], 'statut':result['statut'], 'MRZ':result['MRZ']}
                    new_MRZ = [info_MRZ, new_contour]
                    liste_MRZ.append(new_MRZ)
    return liste_MRZ

def pattern_matching(regex,string):
        match = re.search(regex, string)
        return match.groups() if match else None 
    
def replace_car(text, isNumeric):
    Number_to_letter = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '8': 'B', '<': ' '}
    Letter_to_number = {'B': '8', 'C': '0', 'D': '0', 'G': '6', 'I': '1', 'O': '0', 'Q': '0', 'S': '5', 'Z': '2'}
    if isNumeric == True:
        for key in Letter_to_number.keys():
            text = text.replace(key, Letter_to_number[key])
    else:
        for key in Number_to_letter.keys():
            text = text.replace(key, Number_to_letter[key])
    return text
        
def process_MRZ(texte):
    clean_MRZ = ""
    bandes = texte.split("\n")
    for bande in bandes:
        if "<" in bande:
            if clean_MRZ == "":
                clean_MRZ = bande
            else:
                clean_MRZ = clean_MRZ + "\n" + bande
    clean_MRZ = clean_MRZ.replace(" ", "")
    regex_cni_fra = ".*([I|L|T|Z|U][O|D|B|P])([F|P|E|M][R|P|-|B][A|8])(?P<last_name>.+?(?=<<))(.*)\\n+(?P<card_number>.{13})(?P<first_name>.+?(?=<<|\w+))<+(?P<first_name2>.*?(?=<|\d))?<*(?P<first_name3>[A-Z]*?(?=<|\d))?<*?(?P<birth_date_year>.{2})?(?P<birth_date_month>.{2})?(?P<birth_date_day>.{2})([0-9]*)(?P<sex>[MF]{1})?."
    regex_pass_fra = ".*([P|D|F]<)([F|P|E][R|P][A|8])(?P<last_name>.+?(?=<<))<<(?P<first_name>.+?(?=<))<(?P<second_name>.+?(?=<))<(?P<third_name>.+?(?=<))(.*)\\n(.*)[F|P|E][R|P][A|8](?P<birth_year>.{2})(?P<birth_month>.{2})(?P<birth_day>.{2}).{1}(?P<sex>.{1})(?P<expiration_year>.{2})(?P<expiration_month>.{2})(?P<expiration_day>.{2}).*"
    result = {}
    result['MRZ'] = clean_MRZ
    pattern_cni_fra = pattern_matching(regex_cni_fra,clean_MRZ)
    pattern_pass_fra = pattern_matching(regex_pass_fra,clean_MRZ)
    if pattern_cni_fra:
        result['type'] = 'ID'
        result['nationalite'] = 'FRA'
        result['nom'] = replace_car(str(pattern_cni_fra[2]), False)
        result['code'] = str(pattern_cni_fra[4][:-1])
        result['prenom'] = replace_car(str(pattern_cni_fra[5]), False)
        result['sexe'] = str(pattern_cni_fra[12]).replace("W", "M")
        result['date_naissance'] = replace_car(str(pattern_cni_fra[10]) + "." + str(pattern_cni_fra[9]) + "." + str(pattern_cni_fra[8]), True)
        result['date_validite'] = replace_car(str(pattern_cni_fra[4][2:4]) + "." + str(pattern_cni_fra[4][0:2]), True)        
        try:
            if (int(str(now.year)[2:]) - int(replace_car(pattern_cni_fra[4][0:2], True)) > 15) or (int(str(now.year)[2:]) - int(replace_car(pattern_cni_fra[4][0:2], True)) < 0) :
                result['statut'] = 'EXPIREE'
            elif (int(str(now.year)[2:]) - int(replace_car(pattern_cni_fra[4][0:2], True)) == 15) and (int(str(now.month)) - int(replace_car(pattern_cni_fra[4][2:4], True)) < 0):
                result['statut'] = 'EXPIREE'
            else:
                result['statut'] = 'VALIDE'
        except ValueError:
                result['statut'] = 'INCONNU'
    elif pattern_pass_fra:
        result['type'] = "P"
        result['nationalite'] = "FRA"
        result['nom'] = replace_car(str(pattern_pass_fra[2]), False)
        result['prenom'] = replace_car(str(pattern_pass_fra[3]), False)
        result['code'] = replace_car(str(pattern_pass_fra[7]), True)
        result['sexe'] = str(pattern_pass_fra[11]).replace("W", "M")
        result['date_naissance'] = replace_car(str(pattern_pass_fra[10]) + "." + str(pattern_pass_fra[9]) + "." + str(pattern_pass_fra[8]), True)
        result['date_validite'] = replace_car(str(pattern_pass_fra[14]) + "." + str(pattern_pass_fra[13]) + "." + str(pattern_pass_fra[12]), True)        
        try:
            if (int(str(now.year)[2:]) - int(replace_car(str(pattern_pass_fra[12]), True)) > 0) or (int(str(now.year)[2:]) - int(replace_car(str(pattern_pass_fra[12]), True)) < -10) :
                result['statut'] = 'EXPIREE'
            elif (int(str(now.year)[2:]) - int(replace_car(str(pattern_pass_fra[12]), True)) == 0) and (int(str(now.month)) - int(replace_car(str(pattern_pass_fra[13]), True)) < 0):
                result['statut'] = 'EXPIREE'
            else:
                result['statut'] = 'VALIDE'
        except:
                result['statut'] = 'INCONNU'
    else:
        if (len(clean_MRZ)) > 70 and (len(clean_MRZ) < 100):
            result['nationalite'] = None
            result['type'] = None
            result['nationalite'] = None
            result['nom'] = None
            result['prenom'] = None
            result['code'] = None
            result['sexe'] = None
            result['date_naissance'] = None
            result['date_validite'] = None  
            result['statut'] = None
        else:
           result = None 
    return result

def get_MRZ(filename, img):
    liste_MRZ = []
    (h0, w0) = img.shape[:2]
    if h0 == min(h0, w0):
        side = "h"
    else:
        side = "w"
    for size in range(300, 1300, 50):
        (image, factor) = resize_img(img, size, side)
        (h, w) = image.shape[:2]
        blur = cv2.GaussianBlur(image, (3,3), 0)
        sqKernel0 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        sqKernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        sqKernel2 = np.ones((3,3),np.uint8)
        blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, sqKernel0)
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, sqKernel0)  
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel1)
        thresh = cv2.erode(thresh, sqKernel2, iterations=4)
        p = int(image.shape[1] * 0.05)
        thresh[:, 0:p] = 0
        thresh[:, image.shape[1] - p:] = 0
        show("thresh", thresh)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            liste_MRZ = read_if_MRZ_and_new(liste_MRZ, c, img, factor, filename)              
    return liste_MRZ

def process_file(file):
    images = load_images_from_file(path_to_files, file)
    for key, value in images.items():
        if type(value) is not str:
            try:
                liste_MRZ = get_MRZ(key, value)
                if len(liste_MRZ) > 0:
                    for MRZ in liste_MRZ:
                       mrz_complet.append(MRZ[0])
                else:
                    liste_MRZ = get_MRZ(key, rotate(value))
                    if len(liste_MRZ) > 0:
                        for MRZ in liste_MRZ:
                            mrz_complet.append(MRZ[0])
                    else:
                        temp = {'filename':key, 'resultat':'Aucune bande MRZ détéctée', 'type':None, 'code':None, 'nom':None, 'prenom':None, 'date_naissance':None, 'nationalite':None, 'sexe':None, 'date_validite':None, 'statut':None, 'MRZ':None}
                        mrz_complet.append(temp)
            except:
                temp = {'filename':key, 'resultat':'Erreur image illisible', 'type':None, 'code':None, 'nom':None, 'prenom':None, 'date_naissance':None, 'nationalite':None, 'sexe':None, 'date_validite':None, 'statut':None, 'MRZ':None}
                mrz_complet.append(temp)
        else:
            temp = {'filename':key, 'resultat':value, 'type':None, 'code':None, 'nom':None, 'prenom':None, 'date_naissance':None, 'nationalite':None, 'sexe':None, 'date_validite':None, 'statut':None, 'MRZ':None}
            mrz_complet.append(temp)
    
if __name__ == "__main__":
    mrz_complet = []
    path_to_files = ('test/')
    files = os.listdir(path_to_files)
    for file in tqdm(files):
        process_file(file)
    df_MRZ = pd.DataFrame(mrz_complet)
#    df_MRZ.to_csv("result_Payplug.csv", sep=";", encoding="latin-1")
    