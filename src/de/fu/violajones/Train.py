from de.fu.violajones.IntegralImage import IntegralImage
from de.fu.violajones.HaarLikeFeature import FeatureTypes, HaarLikeFeature
import numpy as np
from joblib import Parallel, delayed, cpu_count
from joblib.pool import has_shareable_memory
import os
import copy
import pickle

N_CPUS = cpu_count()

def load_images(path, label):
    images = []
    for _file in os.listdir(path):
        if _file.endswith('.pgm'):
            images.append(IntegralImage(os.path.join(path, _file), label))
    return images


def generate_features(size):
    print('Generating Features')
    features = []
    for f in FeatureTypes:
        for width in range(f[0], size, f[0]):
            for height in range(f[1], size, f[1]):
                for x in range(size - width):
                    for y in range(size - height):
                        features.append(HaarLikeFeature(f, (x, y), width, height, 0, 1))
    print('..done.\n' + str(len(features)) + ' features created.\n')
    return features



# find / update threshold and coeff for each feature
def feature_eval(feature, used):
    if used:
        return feature, np.inf

    # find the scores for the images
    n_img = len(images)
    scores = np.zeros(n_img)
    for i, img in enumerate(images):
        scores[i] = feature.get_score(img)
    sorted_img_args = np.argsort(scores)
    Sp = np.zeros(n_img)  # sum weights for positive examples below current img
    Sn = np.zeros(n_img)  # sum weights for negative examples below current img
    Tp = 0
    Tn = 0
    for img_arg in np.nditer(sorted_img_args):
        if labels[img_arg] == 0:
            Tn += w[img_arg]
        else:
            Tp += w[img_arg]
        Sn[img_arg] = Tn
        Sp[img_arg] = Tp

    # compute the formula for the threshold
    nerror = Sp + (Tn - Sn)  # error of classifying everything negative below threshold
    perror = Sn + (Tp - Sp)  # error of classifying everything positive below threshold
    error = np.minimum(perror, nerror)  # find minimum
    best_threshold_img = np.argmin(error)  # find the image with the threshold
    best_local_error = error[best_threshold_img]
    feature.threshold = scores[best_threshold_img]  # use the score we estimated for the image as new threshold
    # assign new polarity, based on above calculations
    feature.polarity = 1 if nerror[best_threshold_img] < perror[best_threshold_img] else -1

    # store the error to find best feature
    return feature, best_local_error


def train(positive, negative, T):
    # count
    n_pos = len(positive)
    n_neg = len(negative)
    n_img = n_pos + n_neg
    used = []
    final_classifiers = []

    # gather images, extract labels
    global images, labels, w
    images = np.hstack((np.array(positive), np.array(negative)))
    labels = [img.label for img in images]

    # assign initial weights
    w = np.zeros(n_img)
    w[:n_pos] = 1./(2*n_pos)
    w[n_pos:] = 1./(2*n_neg)

    # generate features
    features = generate_features(20)

    # generate error list
#    n_features = len(features)
#    errors = np.zeros(n_features)

    # ADABoost loop -------------------
    for round in range(T):
        print('Feature selection round # %d'%(round+1))

        # normalize the weights
        w = w/np.sum(w)

        result = list(map(list, zip(*(Parallel(n_jobs=N_CPUS, max_nbytes=1e6, verbose=1)(delayed(feature_eval)
                                                              (feature, feature_nr in used)
                                                              for feature_nr, feature in enumerate(features))))))
        features = result[0]
        errors = np.array(result[1])

        # select best feature
        best_feature_arg = np.argmin(errors)
        best_feature = features[np.asscalar(best_feature_arg)]
        best_error = errors[best_feature_arg]
        used.append(best_feature_arg)

        # compute beta
        beta = best_error/(1.-best_error)

        # update weights
        for i, img in enumerate(images):
            if best_feature.get_vote(img) == labels[i]:
                w[i] = w[i] * beta

        # compute alpha
        alpha = np.log(1./beta)

        # store final classifier
        final_classifiers.append((copy.deepcopy(best_feature), alpha))

        if (round+1) % 10 == 0:
            with open('classifiers.pkl', 'wb') as file:
                pickle.dump(final_classifiers)


def main():
    RED = 1000

    print('Loading faces..')
    faces = load_images('train/face', 1)
    print('..done. %d faces loaded, reducing to %d' % (len(faces), RED))
    faces = faces[:RED]
    print('Loading non-faces')
    non_faces = load_images('train/non-face', 0)
    print('..done. %d non faces loaded, reducing to %d' % (len(non_faces), RED))
    non_faces = non_faces[:RED]
    T = 300
    train(faces, non_faces, T)

if __name__ == '__main__':
    main()