import cv2
import numpy as np


def paddingSquare(im):
    shp = im.shape
    rows, cols = shp[0], shp[1]
    m = max(rows,cols)
    im2 = np.empty((), dtype=np.uint8)
    if len(shp) != 2:
        im2 = np.zeros((m,m,shp[2]), dtype=np.uint8)
        im2[:rows,:cols,:] = im
    else:
        im2 = np.zeros((m,m), dtype=np.uint8)
        im2[:rows,:cols] = im
    return im2


def verticalSplit(im):
    shp = im.shape
    mid = shp[1] // 2
    im2 = np.empty((), dtype=np.uint8)
    im3 = np.empty((), dtype=np.uint8)
    if len(shp) != 2:
        im2 = im[:,:mid,:]
        im3 = im[:,mid:,:]
    else:
        im2 = im[:,:mid]
        im3 = im[:,mid:]
    im2 = cv2.flip(im2, 1) # vertical flip
    return im2, im3


def derotate(im, angle=0, vadd=0):
    shp = np.array(im.shape)
    shp[0] += vadd
    im2 = np.zeros(shp, dtype=np.uint8)
    im2[vadd:,...] = im
    im2 = rotate(im2, -angle)
    return im2


def cropTop(im, vcut):
    return im[vcut:,...]


def getMinBoxCoords(im):
    im2 = im.astype(np.uint8)
    shp = np.array(im.shape)
    if len(shp) != 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(
        im2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    rect = cv2.minAreaRect(cnt)
    boxm = cv2.boxPoints(rect)
    # box = np.int0(boxm)
    return boxm


def getHighestPointWithNeighbors(box):
    hp = np.argmin(box[:, 1])  # highest point
    # hp += 4
    hpm = hp - 1
    hpp = hp + 1
    # hp = hp % 4
    hpm = hpm % 4
    hpp = hpp % 4
    return hp, hpm, hpp


def getInitialAngleEstimate(box):
    hp, hpm, hpp = getHighestPointWithNeighbors(box)
    # Compare lengths of sides
    if np.linalg.norm(box[hp] - box[hpm]) > np.linalg.norm(box[hp] - box[hpp]):
        if abs(box[hp][0]-box[hpm][0]) < 0.0001:  # to avoid division by zero
            km = 10000
        else:
            km = (box[hpm][1]-box[hp][1])/(box[hp][0]-box[hpm][0])
    else:
        if abs(box[hp][0]-box[hpp][0]) < 0.0001:  # to avoid division by zero
            km = 10000
        else:
            km = (box[hpp][1]-box[hp][1])/(box[hp][0]-box[hpp][0])
    angl = -1*np.arctan(km)*180/np.pi
    return angl


# Only call this method after rotateAndAdjust()
def rotate(im, angle):
    shp = np.array(im.shape)
    rows, cols = shp[0], shp[1]
    # image_center = tuple(np.array(im.shape[1::-1]) // 2)
    rot_mat = cv2.getRotationMatrix2D((rows//2, cols//2), angle, 1.0)
    # rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    im2 = cv2.warpAffine(im, rot_mat, (rows, cols), flags=cv2.INTER_LINEAR)
    # im2 = cv2.warpAffine(im, rot_mat, im.shape[1::-1], flags=cv2.INTER_LINEAR)
    # im2 = cv2.warpAffine(im2, rot_mat, im.shape[1::-1], flags=cv2.INTER_LINEAR)
    return im2


# This should be the first method to call, it will rotate the image and return the angle
# Other images/masks with the same rotation angle can then be modified with rotate()
def rotateAndAdjust(im):
    shp = np.array(im.shape)
    box = getMinBoxCoords(im)
    angle = getInitialAngleEstimate(box)
    im2 = rotate(im, angle)
    # image_center = tuple(np.array(im.shape[1::-1]) / 2)
    # Take a slice from top middle of the image, if contains non-black pixels, rotate 180 degrees
    if len(shp) != 2:
        if np.sum(im2[:im.shape[0]//8, im.shape[1]//3:2*im.shape[1]//3, ...] != 0):
            im2 = rotate(im2, 180)
            angle += 180
    else:
        if np.sum(im2[:im.shape[0]//8, im.shape[1]//3:2*im.shape[1]//3] != 0):
            im2 = rotate(im2, 180)
            angle += 180
    return im2, angle
