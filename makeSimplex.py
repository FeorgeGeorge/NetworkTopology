import os

from keras import layers
from keras import models
import numpy as np
import copy
import itertools
import pickle

# model parameters
first = 10
second = 10
third = 10
output = 1

# model file name
modelDir = "./models/"

# the list of filtration. Parallel process execution is possible by distributing this list.
filList = range(1, 65)


def get_model_name(params, cut):
    return "{}_cut_{}_{}l1_{}l2_{}l3".format(cut, **params)


def simplexDir(cut): return "cut" + str(cut)


def get_relevance(model, outputSize=1):
    layers = model.layers

    print(len(layers))
    weights = [layer.get_weights()[0] for layer in layers]

    print(len(weights))

    sizes = [len(weight) for weight in weights] + [outputSize]

    offset = 0
    size = sum(sizes)
    relevance = np.identity(size)
    for layer_num in range(len(sizes) - 1, 0, -1):
        old_offset = offset
        offset += sizes[layer_num]
        for j in range(0, sizes[layer_num]):
            weight = weights[layer_num - 1]
            weightPlus = weight * (weight > 0)

            normalizeFactor = 0
            for i in range(sizes[layer_num - 1]):
                normalizeFactor += weightPlus[i][j]
            for i in range(sizes[layer_num - 1]):
                x, y = i + offset, j + old_offset
                if weightPlus[i][j] != 0:
                    relevance[x][y] = weightPlus[i][j] / normalizeFactor

    print("Total relevance length: {}".format(len(relevance)))
    return np.array(relevance)


def comb(sequence):
    result = []
    for L in range(1, len(sequence)+1):
        for subset in itertools.combinations(sequence, L):
            result.append(list(subset))
    return result


def getSimplex(matrix, pointSequence, threshold):
    matrixSize = len(matrix)

    relevance = 1.0
    result = []
    # startPointからのRelevanceを計算する
    startPoint = pointSequence[0]
    for pointNumber in pointSequence:
        relevance = relevance * matrix[startPoint][pointNumber]
        startPoint = pointNumber
    # relevanceがthreshold以上だったらここまでの経路を追加する

    if relevance >= threshold:
        for e in comb(pointSequence):
            result.append(e)
        # 最後の要素からの連結要素について再帰的にチェックする
        lastPoint = pointSequence[-1]
        for i in range(matrixSize):
            if matrix[lastPoint][i] > 0 and i != lastPoint:
                tempPointSequence = copy.deepcopy(pointSequence)
                tempPointSequence.append(i)
                # 再帰呼び出し
                temp = getSimplex(matrix, tempPointSequence, threshold)
                # 結果をresultに追加
                for e in temp:
                    for ee in comb(e):
                        result.append(ee)
    return list(map(list, set(map(tuple, result))))


def registerSimplexOutput(filList, cut=1, name=None):
    if name == None:
        model = models.load_model(get_model_name(cut))
    else:
        model = models.load_model("./models/" + name)
    matrix = get_relevance(model)
    matrixSize = len(matrix)
    r = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2,
         1.0e-1, 0.9e-1, 0.8e-1, 0.7e-1, 0.6e-1, 0.5e-1, 0.4e-1, 0.3e-1, 0.2e-1,
         1.0e-2, 0.9e-2, 0.8e-2, 0.7e-2, 0.6e-2, 0.5e-2, 0.4e-2, 0.3e-2, 0.2e-2,
         1.0e-3, 0.9e-3, 0.8e-3, 0.7e-3, 0.6e-3, 0.5e-3, 0.4e-3, 0.3e-3, 0.2e-3,
         1.0e-4, 0.9e-4, 0.8e-4, 0.7e-4, 0.6e-4, 0.5e-4, 0.4e-4, 0.3e-4, 0.2e-4,
         1.0e-5, 0.9e-5, 0.8e-5, 0.7e-5, 0.6e-5, 0.5e-5, 0.4e-5, 0.3e-5, 0.2e-5,
         1.0e-6, 0.9e-6, 0.8e-6, 0.7e-6, 0.6e-6, 0.5e-6, 0.4e-6, 0.3e-6, 0.2e-6,
         1.0e-7]
    print("Filtration: ", end="")
    for fil in filList:
        number = r[fil - 1]
        if name == None:
            filename = simplexDir(cut) + "/Simplex" + str(fil)
        else:
            filename = name + "/Simplex" + str(fil)

        print(str(fil) + ", ", end="")

        saveSimplex = []
        for startPoint in range(0, matrixSize):
            simplex = getSimplex(matrix, [startPoint], number)
            saveSimplex.extend(simplex)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        saveFile = open(filename, 'wb')
        pickle.dump(saveSimplex, saveFile)
        saveFile.close
