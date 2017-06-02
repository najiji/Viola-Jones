import numpy as np
from HaarLikeFeature import FeatureType
from HaarLikeFeature import HaarLikeFeature
from de.fu.violajones.HaarLikeFeature import FeatureTypes
import sys
import pickle
import os
from pympler import summary, muppy


class AdaBoost(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
    
def learn(positives, negatives, T):
    if os.path.isfile('votes.pkl'):
        images = []
        print('loading preprocessed votes..')
        with open('votes.pkl', 'rb') as file:
            votes = pickle.load(file)
            f_votes = next(iter(votes.values())).tolist()
            for img, _ in f_votes:
                images.append(img)
        images = np.array(images)


    else:
        print('Generating data from scratch')
        # construct initial weights
        pos_weight = 1. / (2 * len(positives))
        neg_weight = 1. / (2 * len(negatives))
        for p in positives:
            p.set_weight(pos_weight)
        for n in negatives:
            n.set_weight(neg_weight)
    
        # create column vector
        images = np.hstack((positives, negatives))
    
        print('Creating haar like features..')
        features = []
        for f in FeatureTypes:
            for width in range(f[0],20,f[0]):
                for height in range(f[1],20,f[1]):
                    for x in range(20-width):
                        for y in range(20-height):
                            features.append(HaarLikeFeature(f, (x,y), width, height, 0, 1))
        print('..done.\n' + str(len(features)) + ' features created.\n')
    
        print('Calculating scores for features..')
        # dictionary of feature -> list of vote for each image: matrix[image, weight, vote])
        votes = dict()

        i = 0

        for feature in features:
            # calculate score for each image, also associate the image
            votes[feature] = np.array(list(map(lambda im: [im, feature.get_vote(im)], images)))

            i += 1
            if i % 1000 == 0:
                print(str(i) + ' features of ' + str(len(features)) + ' done')

        # pickle our work from before
        print('storing generated votes..')
        with open('votes.pkl', 'wb') as file:
            pickle.dump(votes, file)

    print('..done.\n')



    
    # select classifiers
    classifiers = []
    used = []
    n_features = len(votes)

    print('Selecting classifiers..')

    for i in range(T):
        print('picking feature # %d ..'%(i+1))
        classification_errors = dict()

        # normalize weights
        norm_factor = 1. / sum(map(lambda im: im.weight, images))
        for image in images:
            image.set_weight(image.weight * norm_factor)

        # compute information gains of the classifiers over the images
        i_feature = 1
        for feature, feature_votes in votes.items():
            
            if feature in used:
                continue

            error = sum(map(lambda im, vote: im.weight if im.label != vote else 0, feature_votes[:,0], feature_votes[:,1]))
            # map error -> feature, use error as key to select feature with
            # smallest error later
            classification_errors[error] = feature
            if i_feature % 1000 == 0:
                print('[ %d of %d ]\r'%(i_feature, n_features))

            i_feature += 1

        print("")
        # get best feature, i.e. with smallest error
        errors = list(classification_errors.keys())
        best_error = np.min(errors)
        feature = classification_errors[best_error]
        used.append(feature)
        feature_weight = 0.5 * np.log((1-best_error)/best_error)
        
        classifiers.append((feature, feature_weight))
        
        # update image weights
        best_feature_votes = votes[feature]
        for feature_vote in best_feature_votes:
            im = feature_vote[0]
            vote = feature_vote[1]
            if im.label != vote:
                im.set_weight(im.weight * np.sqrt((1-best_error)/best_error))
            else:
                im.set_weight(im.weight * np.sqrt(best_error/(1-best_error)))

        if (i+1) % 10 == 0:
            with open('classifiers.pckl', 'wb') as file:
                pickle.dump(classifiers, file)
    
    return classifiers
        