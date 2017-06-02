from HaarLikeFeature import HaarLikeFeature
import pickle
import numpy as np


# scaling iterations, each iteration scales by 2, so 3 is equivalent to a scale of 8
scaling_LUT = {
    1: 0,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 2,
    7: 3,
    8: 3,
    9: 3,
    10: 3,
    11: 3,
    12: 3
}


def load_classifiers():
    with open('classifiers.pckl', 'rb') as file:
        classifiers = pickle.load(file)
        print('loaded %d classifiers' % len(classifiers))
        for c, cw in classifiers:
            print('type: (%d, %d), w: %d, h: %d'%(c.type[0], c.type[1], c.width, c.height))
        return classifiers


def find_pairs(classifiers):
    # find identical feats in different positions
    pairs = []
    used = []
    for c1 in classifiers:
        l = [c1]
        if c1 in used:
            continue
        for c2 in classifiers:
            if c2 not in used and c1 != c2 and c1[0].width == c2[0].width and c1[0].height == c2[0].height and c1[0].type == c2[0].type:
                l.append(c2)
        used = used+l
        pairs.append(l)
    return pairs


def strategy(dir, dist):
    # scaling
    scaling_iter = scaling_LUT[dist]
    s = []
    for i in range(scaling_iter):
        s.append('E = div2(D)') if i % 2 == 0 else s.append('D = div2(E)')
    if scaling_iter % 2 != 0:
        s.append('D = copy(E)')

    # adding
    shifter_size = 1
    s.append('C = copy(D)')

    for i in range(shifter_size):
        s.append('D = south(D)')
    s.append('C = add(C, D)')

    # decide if we keep current shifter size, or if we increase


def get_full_sum_sequence(cls):
    area = cls.width * cls.height
    s = []
    if cls.type == (1,2): # TWO_VERTICAL
        if cls.height <= 1:
            print('[WARNING] too low height VERTICAL feature')
        s = s + strategy('vert', cls.height//2)


    elif cls.type == (2,1): # TWO_HORIZONTAL
        pass
    elif cls.type == (1,3): # THREE_VERTICAL
        pass
    elif cls.type == (3,1): # THREE_HORIZONTAL
        pass
    elif cls.type == (2,2): # FOUR
        pass


def pair_instruction_sequence(pairs):
    for pair in pairs:
        # get first classifier
        fc, fcw = pair[0]
        sequence = get_full_sum_sequence(fc)


def main():
    classifiers = load_classifiers()
    pairs = find_pairs(classifiers)
    print(len(pairs))


if __name__ == '__main__':
    main()