import numpy as np
import pandas as pd
import scipy.spatial as spa
import scipy.stats as sta
import scipy.special as spe
import random
import sys


def pointFurthestAwayFrom(point, points):
    dist = -sys.maxsize - 1
    found = None
    for p in points:
        nDist = spa.distance.euclidean(point, p)
        if (nDist > dist):
            dist = nDist
            found = p

    return found


def emInit(points, k):
    rIdx = random.randint(0, len(points) - 1)
    mus = [points[rIdx]]
    covs = k * [np.identity(len(points[rIdx]))]
    priors = k * [1 / k]
    for i in range(k):
        p = mus[-1]
        mus.append(pointFurthestAwayFrom(p, points))

    return [mus, covs, priors]


def isConverged(mus, prev_mus, k, eps):
    total = 0
    for i in range(k):
        total += np.linalg.norm((mus[i] - prev_mus[i]))

    print(total)

    if (total <= eps):
        return True
    else:
        return False


def expectationMaximization(points, k, eps):
    t = 0
    n = len(points)  # number of points
    # inital paramters
    mus, covs, priors = emInit(points, k)
    p_mu = len(points[0]) * [0]
    prev_mus = k * [p_mu.copy()]
    finalPosteriorProbabilities = []
    converged = False
    while (not converged):
        t = t + 1
        posteriorProbabilities = []
        mvns = []
        for i in range(k):
            mvns.append((sta.multivariate_normal.logpdf(
                points, mean=mus[i], cov=covs[i], allow_singular=True)) *
                        priors[i])
        mvns_sum = np.sum(np.array(mvns))

        # expectation
        for i in range(k):
            classPosteriorProbabilities = []
            mvn = mvns[i]
            for j in range(n):
                numer = spe.logsumexp(mvn[j])
                denom = spe.logsumexp(mvns_sum)

                posteriorProbability = numer / denom  # for point j in class i
                classPosteriorProbabilities.append(posteriorProbability)
            posteriorProbabilities.append(classPosteriorProbabilities)

        print(posteriorProbabilities)
        # maximization
        prev_mus = mus.copy()
        mus = []
        covs = []
        priors = []

        for i in range(k):
            muNumerTotal = np.array(p_mu.copy())
            covNumerTotal = np.zeros(shape=(len(points[0]), len(points[0])))
            denomTotal = 0.0
            for j in range(n):
                muNumerTotal = muNumerTotal + posteriorProbabilities[i][
                    j] * points[j]
                denomTotal = denomTotal + posteriorProbabilities[i][j]

                covNumerTotal = covNumerTotal + (
                    posteriorProbabilities[i][j] *
                    np.dot(points[j] - prev_mus[i],
                           np.transpose(points[j] - prev_mus[i])))

            mu = muNumerTotal / denomTotal
            cov = covNumerTotal / denomTotal
            prior = denomTotal / n

            mus.append(mu)
            covs.append(cov)
            priors.append(prior)

        # check convergence
        converged = isConverged(mus, prev_mus, k, eps)
        finalPosteriorProbabilities = np.transpose(
            np.array(posteriorProbabilities))
    return [mus, covs, priors, finalPosteriorProbabilities]


def generateClusterLabels(clusterLabelData):

    clusterLabels = []
    for value in clusterLabelData:
        if (value <= 40):
            clusterLabels.append(0)
        elif (value <= 60):
            clusterLabels.append(1)
        elif (value <= 100):
            clusterLabels.append(2)
        else:
            clusterLabels.append(3)

    return clusterLabels


def parseFile(filename, k):
    headers = list(pd.read_csv(filename, nrows=0))
    # remove data/time and rv2 attributes
    headers.pop(len(headers) - 1)
    headers.pop(0)

    data = pd.read_csv(filename, usecols=headers).to_numpy()
    data_T = np.transpose(data)
    applianceAttributeData = data_T[0]  # appliances attribute data
    trueClusterLabels = generateClusterLabels(applianceAttributeData)

    return [data, data_T, trueClusterLabels, headers]


def assignPoints(k, points, posteriors):

    clusters = []

    for i in range(len(points)):
        post = posteriors[i].tolist()
        maxPost = max(post)
        maxPostIdx = post.index(maxPost)
        clusters.append(maxPostIdx)

    return clusters


def purityScore(points, k, clusters, trueClusterLabels, trueNumberOfClusters):
    score = 0
    n = len(trueClusterLabels)

    for i in range(len(trueClusterLabels)):
        if (clusters[i] == trueClusterLabels[i]):
            score += 1
    return score / n


if __name__ == "__main__":

    arguments = sys.argv
    filename = arguments[1]
    k = int(arguments[2])  # num of clusters
    eps = float(arguments[3])  # convergence threshold
    ridge = float(
        arguments[4]
    )  #not using ridge, going to use pdf with accept singular matricies
    max_iter = int(arguments[5])  # max iterations

    # parse
    point_view, col_view, trueClusterLabels, headers = parseFile(filename, k)

    mus, covs, priors, posteriors = expectationMaximization(point_view, k, eps)

    clusters = assignPoints(k, point_view, posteriors)

    c = {}
    for i in clusters:
        if i not in c:
            c[i] = 1
        else:
            c[i] += 1
    print("CLUSTER INFORMATION")
    for i in range(k):
        if i in c:
            print("----CLUSTER", i, "----")
            print("MEAN:", mus[i])
            print("COVARIANCE:", covs[i])
            print("SIZE:", c[i])

    print("PURITY SCORE: ",
          purityScore(point_view, k, clusters, trueClusterLabels, 4))
