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

def classify(classifiers, image):
    return 1 if sum([c[0].get_vote(image) * c[1] for c in classifiers]) >= 0 else -1
    
        

if __name__ == "__main__":
    
    # TODO: select optimal threshold for each feature
    # TODO: attentional cascading
    
    print('Loading faces..')
    faces = load_images('train/face', 1)
    print('..done. ' + str(len(faces)) + ' faces loaded.\n\nLoading non faces..')
    non_faces = load_images('train/non-face', -1)
    print('..done. ' + str(len(non_faces)) + ' non faces loaded.\n')
    
    T = 3000
    classifiers = AdaBoost.learn(faces, non_faces, T)
    
    print('Loading test faces..')
    faces = load_images('test/face', 1)
    print('..done. ' + str(len(faces)) + ' faces loaded.\n\nLoading test non faces..')
    non_faces = load_images('test/non-face', -1)
    print('..done. ' + str(len(non_faces)) + ' non faces loaded.\n')

    with open('classifiers_' + str(T) + '_' + hex(random.getrandbits(16)) + '.pckl', 'wb') as file:
        pickle.dump(classifiers, file)

    print('Validating selected classifiers..')
    correct_faces = 0
    correct_non_faces = 0
    for image in faces + non_faces:
        result = classify(classifiers, image)
        if image.label == 1 and result == 1:
            correct_faces += 1
        if image.label == -1 and result == -1:
            correct_non_faces += 1
            
    print('..done. Result:\n  Faces: ' + str(correct_faces) + '/' + str(len(faces)) + '\n  non-Faces: ' + str(correct_non_faces) + '/' + str(len(non_faces)))
