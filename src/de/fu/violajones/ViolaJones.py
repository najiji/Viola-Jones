from de.fu.violajones import AdaBoost
from de.fu.violajones.IntegralImage import IntegralImage
import os
import pickle
import random

def load_images(path, label):
    images = []
    for _file in os.listdir(path):
        if _file.endswith('.pgm'):
            images.append(IntegralImage(os.path.join(path, _file), label))
    return images

if __name__ == "__main__":
    
    # TODO: select optimal threshold for each feature
    # TODO: attentional cascading
    faces = []
    non_faces = []
    print('Loading faces..')
#    faces = load_images('train/face', 1)
    print('..done. ' + str(len(faces)) + ' faces loaded.\n\nLoading non faces..')
#    non_faces = load_images('train/non-face', -1)
    print('..done. ' + str(len(non_faces)) + ' non faces loaded.\n')
    
    T = 3000
    classifiers = AdaBoost.learn(faces, non_faces, T)
