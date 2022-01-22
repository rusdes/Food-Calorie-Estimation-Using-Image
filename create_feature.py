from feature_moments import getShapeFeatures
from feature_gabor import *
from feature_color import getColorFeature
from img_seg import *

def createFeature(img):
    '''
    Creates the feature vector of the image using the three features -
    color, texture, and shape features
    '''
    feature = []
    areaFruit, binaryImg, colourImg, areaSkin, fruitContour, pix_to_cm_multiplier = getAreaOfFood(img)
    
    color = getColorFeature(colourImg)
    texture = getTextureFeature(colourImg)
    shape = getShapeFeatures(binaryImg)

    for i in color:
        feature.append(i)
    for i in texture:
        feature.append(i)
    for i in shape:
        feature.append(i)

    feature = list(map(lambda x: x * 2, feature))
    mean=np.mean(feature)
    dev=np.std(feature)
    feature = (feature - mean)/dev

    return feature, areaFruit, areaSkin, fruitContour, pix_to_cm_multiplier

def readFeatureImg(filename):
    '''
    Reads an input image when the filename is given,
    and creates the feature vector of the image.
    '''
    img = cv2.imread(filename)
    feature, areaFruit, areaSkin, fruitContour, pix_to_cm_multiplier = createFeature(img)
    return feature, areaFruit, areaSkin, fruitContour, pix_to_cm_multiplier

if __name__ == '__main__':
    feature, areaFruit, areaSkin, fruitContour, pix_to_cm_multiplier = readFeatureImg("./Dataset/images/All_Images/1_12.jpg")
    print(areaFruit, areaSkin)
    print(fruitContour)
    print(pix_to_cm_multiplier)