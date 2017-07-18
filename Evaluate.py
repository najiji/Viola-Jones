import os
import pickle
from IntegralImage import IntegralImage
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, cpu_count
import numpy as np

N_CPUS = cpu_count()

def print_classifier(classifier):
    print('========================================')
    height = classifier[0].height
    width = classifier[0].width
    tlx, tly = classifier[0].top_left
    mask = np.zeros((19, 19))
    if classifier[0].type == (1,2):
        for x in range(tlx, tlx+width):
            for y in range(tly, tly+height//2):
                mask[x,y] = 1 - 2*classifier[0].polarity
        for x in range(tlx, tlx+width):
            for y in range(tly+height//2, tly+height):
                mask[x,y] = -1 + 2*classifier[0].polarity

    elif classifier[0].type == (2,1):
        for x in range(tlx, tlx+width//2):
            for y in range(tly, tly+height):
                mask[x,y] = 1 - 2*classifier[0].polarity
        for x in range(tlx+width//2, tlx+width):
            for y in range(tly, tly+height):
                mask[x,y] = -1 + 2*classifier[0].polarity

    elif classifier[0].type == (3,1):
        for x in range(tlx, tlx+width//3):
            for y in range(tly, tly+height):
                mask[x,y] = 1 - 2*classifier[0].polarity
        for x in range(tlx+width//3, tlx+2*width//3):
            for y in range(tly, tly+height):
                mask[x,y] = -1 + 2*classifier[0].polarity
        for x in range(tlx+2*width//3, tlx+width):
            for y in range(tly, tly+height):
                mask[x,y] = 1 - 2*classifier[0].polarity

    elif classifier[0].type == (1,3):
        for x in range(tlx, tlx+width):
            for y in range(tly, tly+height//3):
                mask[x,y] = 1 - 2*classifier[0].polarity
        for x in range(tlx, tlx+width):
            for y in range(tly+height//3, tly+2*height//3):
                mask[x,y] = -1 + 2*classifier[0].polarity
        for x in range(tlx, tlx + width):
            for y in range(tly + 2*height//3, tly+height):
                mask[x, y] = 1

    elif classifier[0].type == (2,2):
        for x in range(tlx, tlx+width//2):
            for y in range(tly, tly+height//2):
                mask[x,y] = 1 - 2*classifier[0].polarity
        for x in range(tlx+width//2, tlx+width):
            for y in range(tly, tly+height//2):
                mask[x,y] = -1 + 2*classifier[0].polarity

        for x in range(tlx, tlx + width//2):
            for y in range(tly+height//2, tly+height):
                mask[x, y] = -1 + 2*classifier[0].polarity
        for x in range(tlx+width//2, tlx + width):
            for y in range(tly + height//2, tly+height):
                mask[x, y] = 1 - 2*classifier[0].polarity

    for y in range(19):
        for x in range(19):
            char = '.'
            if mask[x, y] == 1:
                char = '+'
            elif mask[x, y] == -1:
                char = '-'
            print(char, end='')
        print('')
    print('========================================')


def print_best_classifiers(classifiers, n):
    for i in range(n):
        print_classifier(classifiers[i])


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


def get_performance(classifiers):
    correct_faces = 0
    correct_non_faces = 0
    for image in faces + non_faces:
        result = classify(classifiers, image)
        if image.label == 1 and result == 1:
            correct_faces += 1
        if image.label == 0 and result == 0:
            correct_non_faces += 1

    TP = correct_faces
    FN = len(faces) - correct_faces
    TN = correct_non_faces
    FP = len(non_faces) - correct_non_faces

    CR = (TP + TN) / (TP + TN + FP +FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return TP, FN, TN, FP, CR, precision, recall



def main():
    cls_file = 'classifiers_120_sym.pkl'
    print('Loading classifiers')
    if os.path.isfile(cls_file):
        with open(cls_file, 'rb') as file:
            classifiers = pickle.load(file)
            print('loaded %d classifiers'%len(classifiers))
            classifiers.sort(key=lambda x: x[1], reverse=True)
    else:
        print('No classifiers file found')
        return

    print_best_classifiers(classifiers, 10)

    RED = 100000
    print('Loading test faces..')
    global faces, non_faces
    faces = load_images('train/face', RED, 1)
    print('..done. ' + str(len(faces)) + ' faces loaded.\n\nLoading test non faces..')
    non_faces = load_images('train/non-face', RED, 0)
    print('..done. ' + str(len(non_faces)) + ' non faces loaded.\n')


    print('Validating selected classifiers..')
    TP, FN, TN, FP, CR, precision, recall = get_performance(classifiers[:500])

    # TP, FN, TN, FP, CR, precision, recall

    print('-------------------------------')
    print('TP: %d, FP: %d, TN: %d, FN: %d' % (TP, FP, TN, FN))
    print('classfication rate: %.5f' % CR)
    print('precision: %.5f (If we say true, how likely is it to be the case)' % precision)
    print('recall: %.5f (If an example is true, how likely are we are going to find it)' % recall)
    print('-------------------------------')

    perf_values = Parallel(n_jobs=N_CPUS, verbose=1)(delayed(get_performance)(classifiers[:i]) for i in range(1, len(classifiers), 1))
    perf_values = list(map(list, zip(*perf_values)))

    plt.plot(perf_values[4], label='Classification rate')
    plt.plot(perf_values[5], label='Precision')
    plt.plot(perf_values[6], label='Recall')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

