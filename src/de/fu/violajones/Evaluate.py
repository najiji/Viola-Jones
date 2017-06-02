from de.fu.violajones.ViolaJones import load_images
import os
import pickle


def classify(classifiers, image):
    evidence = sum([max(c[0].get_vote(image), 0.) * c[1] for c in classifiers])
    weight_sum = sum([c[1] for c in classifiers])

    return 1 if evidence >= weight_sum/2 else -1


def main():
    print('Loading classifiers')
    if os.path.isfile('classifiers.pckl'):
        with open('classifiers.pckl', 'rb') as file:
            classifiers = pickle.load(file)
            print('loaded %d classifiers'%len(classifiers))
    else:
        print('No classifiers file found')
        return

    print('Loading test faces..')
    faces = load_images('test/face', 1)
    print('..done. ' + str(len(faces)) + ' faces loaded.\n\nLoading test non faces..')
    non_faces = load_images('test/non-face', -1)
    print('..done. ' + str(len(non_faces)) + ' non faces loaded.\n')

    print('Validating selected classifiers..')
    correct_faces = 0
    correct_non_faces = 0
    for image in faces + non_faces:
        result = classify(classifiers, image)
        if image.label == 1 and result == 1:
            correct_faces += 1
        if image.label == -1 and result == -1:
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
    print('classfication rate: %.5f' % CR)
    print('precision: %.5f (If we say true, how likely is it to be the case)' % precision)
    print('recall: %.5f (If an example is true, how likely are we are going to find it)' % recall)
    print('-------------------------------')



if __name__ == '__main__':
    main()

