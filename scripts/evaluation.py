import argparse
import glob
import os
import numpy as np


# Stores the information of a symmetry plane
class SymmetryPlaneEvaluation:
    def __init__(self, point, normal, confidence):
        # 3D coords of a canonical plane (for drawing)
        # self.coordsBase = np.array([[0,-1,-1],[0,1,-1],[0,1,1],[0,-1,1]], dtype=np.float32)
        # Indices for the canonical plane
        # self.trianglesBase = np.array([[0,1,3],[3,1,2]], dtype=np.int32)

        # The plane is determined by a normal vector and a point
        self.point = point.astype(np.float32)
        self.normal = normal
        self.normal = self.normal / np.linalg.norm(self.normal)
        self.confidence = confidence

    def __repr__(self):
        return np.array2string(self.normal, precision=2, separator=',') + ',' + np.array2string(self.point, precision=2,
                                                                                                separator=',') + ',' + str(
            self.confidence)


def computeMetricsPerModel(inputRes, value, opt):
    # Read symmetries result
    # print(inputRes)
    symmetry_list = []

    # Read symmetries
    with open(inputRes) as f:
        num_symmetries = int(f.readline().strip())
        # print(num_symmetries)

        for i in range(num_symmetries):
            L = f.readline().strip().split()
            L = [float(x) for x in L]
            symmetry_list.append(
                SymmetryPlaneEvaluation(point=np.array(L[3:6]), normal=np.array(L[0:3]), confidence=L[6]))

        symmetryL = sorted(symmetry_list, key=lambda x: x.confidence, reverse=True)

    # Read the groundtruth
    nameG = inputRes.split('/')[-1].split('_')[0] + '_sym.txt'
    nameP = inputRes.split('/')[-1].split('_')[0] + '.txt'

    P = np.loadtxt(os.path.join(opt.groundtruth_shape, nameP))
    mi = P.min(axis=0)
    ma = P.max(axis=0)
    diag = np.linalg.norm(ma - mi)

    groundt = os.path.join(opt.groundtruth, nameG)

    symmetryG = []
    with open(groundt) as f:
        num_symmetries = int(f.readline().strip())
        # print(groundt)

        for i in range(num_symmetries):
            L = f.readline().strip().split()
            L = [float(x) for x in L]
            symmetryG.append(SymmetryPlaneEvaluation(point=np.array(L[3:6]), normal=np.array(L[0:3]), confidence=1.0))

    numModelsGroundtruth = len(symmetryG)

    # Create match set between results and groundtruth
    retList = []
    count = 0
    for symR in symmetryL:  # For each symmetry in the result
        listDist = []
        for symG in symmetryG:  # Scan the list of groundtruth
            distAngle = 1 - np.abs(np.dot(symR.normal, symG.normal))
            # distPoint = np.linalg.norm(symR.point - symG.point)
            normal = symG.normal
            normal = normal / np.linalg.norm(normal)
            distPoint = np.abs(np.dot(normal, symR.point) - np.dot(normal, symG.point))

            if distAngle <= opt.threshold_angle and distPoint <= opt.threshold_dist * diag:
                listDist.append(distAngle + distPoint / diag)
            else:
                listDist.append(np.inf)

        if len(listDist) == 0:
            retList.append(0)
            continue

        minDist = min(listDist)
        posMin = listDist.index(minDist)

        if np.isinf(minDist):  # There was no a match
            retList.append(0)
        else:
            retList.append(1)
            symmetryG.pop(posMin)
            count = count + 1

    # print(f'{numModelsGroundtruth} - {retList}')

    if count != 0:
        precisionRecallTable = np.zeros((count, 2), dtype=np.float32)
        numRetrieved = 0
        relevantRetrieved = 0

        NN = 0.0
        MAP = 0.0

        for ret in retList:
            if ret == 1:
                if numRetrieved == 0:
                    NN = 1.0

                rec = (relevantRetrieved + 1) / numModelsGroundtruth
                prec = (relevantRetrieved + 1) / (numRetrieved + 1)

                precisionRecallTable[relevantRetrieved, 0] = rec * 100
                precisionRecallTable[relevantRetrieved, 1] = prec
                MAP = MAP + prec
                relevantRetrieved = relevantRetrieved + 1

            numRetrieved = numRetrieved + 1
        MAP = MAP / count

        recall_precision = [0 for i in range(value + 1)]  # Array with precision values per recall

        # Interpolation procedure
        index = count - 2
        recallValues = 100 - (100 / value)
        maxim = precisionRecallTable[index + 1][1]
        pos = value
        recall_precision[pos] = maxim

        pos = pos - 1

        while index >= 0:
            if int(precisionRecallTable[index][0]) >= recallValues:
                if precisionRecallTable[index][1] > maxim:
                    maxim = precisionRecallTable[index][1]
                index = index - 1
            else:
                recall_precision[pos] = maxim
                recallValues = recallValues - 100 / value
                pos = pos - 1

        while pos >= 0:
            recall_precision[pos] = maxim
            pos = pos - 1

        result = [recall_precision[i] for i in range(value + 1)]
    else:
        result = [0.0 for i in range(value + 1)]
        NN = 0.0
        MAP = 0.0

    resultModel = dict()
    resultModel["pr"] = result
    resultModel["NN"] = NN
    resultModel["MAP"] = MAP

    return resultModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundtruth", type=str, default='', help='')
    parser.add_argument("--groundtruth_shape", type=str, default='', help='')
    parser.add_argument("--input_folder", type=str, default='', help='')
    parser.add_argument("--threshold_angle", type=float, default=0.0, help='')
    parser.add_argument("--threshold_dist", type=float, default=0.0, help='')
    parser.add_argument("--bins", type=int, default=10, help='')
    parser.add_argument("--set", type=str, default='all', help='')

    opt = parser.parse_args()

    print(f'################### Set {opt.set} ###################')

    if opt.set == "all":
        # Result analysis folder
        input_list = glob.glob(os.path.join(opt.input_folder, '*_res.txt'))

        if len(input_list) != 9000:
            print(f"The folder with results is incomplete: {len(input_list)}/9000")
            # exit()

        gr_list = glob.glob(os.path.join(opt.groundtruth, '*_sym.txt'))

        if len(gr_list) != 9000:
            print(f"The folder with results is incomplete: {len(gr_list)}/9000")
            exit()
    else:
        set_file = os.path.join(opt.groundtruth, opt.set + ".txt")
        if not os.path.exists(set_file):
            print(f'Set file does not exist:{set_file}')
            exit()

        with open(set_file, 'r') as f:
            input_list = [os.path.join(opt.input_folder, "points" + x.strip() + "_res.txt") for x in f.readlines()]

    # Starts evaluation
    MAP = 0
    NN = 0
    value = opt.bins
    pr = [0.0 for i in range(value + 1)]

    for inputRes in input_list:
        resultModel = computeMetricsPerModel(inputRes, opt.bins, opt)

        pr = [pr[i] + resultModel["pr"][i] for i in range(value + 1)]
        MAP = MAP + resultModel["MAP"]
        NN = NN + resultModel["NN"]

    pr = [pr[i] / len(input_list) for i in range(value + 1)]
    MAP = MAP / len(input_list)
    NN = NN / len(input_list)

    for x in pr:
        print(str(x).replace(".", ","))

    print()
    print('MAP:', str(MAP).replace(".", ","))
    print('PHC:', str(NN).replace(".", ","))


















