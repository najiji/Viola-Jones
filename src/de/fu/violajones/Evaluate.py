import os
import pickle
from de.fu.violajones.IntegralImage import IntegralImage


def load_images(path, reduction, label):
    images = []
    i = 0
    for _file in os.listdir(path):
        if _file.endswith('.pgm'):
            i += 1
            if i > reduction:
                break
            images.append(IntegralImage(os.path.join(path, _file), label))
    return images


def classify(classifiers, image):
    evidence = sum([c[0].get_vote(image) * c[1] for c in classifiers])
    weight_sum = sum([c[1] for c in classifiers])

    return 1 if float(evidence) >= weight_sum/2 else 0


def main():
    print('Loading classifiers')
    if os.path.isfile('classifiers.pkl'):
        with open('classifiers.pkl', 'rb') as file:
            classifiers = pickle.load(file)
            print('loaded %d classifiers'%len(classifiers))
    else:
        print('No classifiers file found')
        return

    RED = 1000
    print('Loading test faces..')
    faces = load_images('test/face', RED, 1)
    print('..done. ' + str(len(faces)) + ' faces loaded.\n\nLoading test non faces..')
    non_faces = load_images('test/non-face', RED, 0)
    print('..done. ' + str(len(non_faces)) + ' non faces loaded.\n')


    print('Validating selected classifiers..')
    correct_faces = 0
    correct_non_faces = 0
    for image in faces + non_faces:
        result = classify(classifiers, image)
        if image.label == 1 and result == 1:
            correct_faces += 1
        if image.label == 0 and result == 0:
            correct_non_faces += 1

    print('..done. Result:\n  Faces: ' + str(correct_faces) + '/' + str(len(faces)) + '\n  non-Faces: ' + str(
        correct_non_faces) + '/' + str(len(non_faces)))

    TP = correct_faces
    FN = len(faces) - correct_faces
    TN = correct_non_faces
    FP = len(non_faces) - correct_non_faces

    CR = (TP + TN) / (TP + TN + FP +FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    print('-------------------------------')
    print('TP: %d, FP: %d, TN: %d, FN: %d' % (TP, FP, TN, FN))
    print('classfication rate: %.5f' % CR)
    print('precision: %.5f (If we say true, how likely is it to be the case)' % precision)
    print('recall: %.5f (If an example is true, how likely are we are going to find it)' % recall)
    print('-------------------------------')



if __name__ == '__main__':
    main()

