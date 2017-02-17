import numpy as np

def getMinimumUniqueSum(arr):

    ref = list(np.zeros(len(arr)))
    print(list(np.zeros(len(arr))))

    for i in range(0, len(arr)-1):
        ref[i] = arr[i+1]-(arr[i]+1)
    ref[len(arr)] = 1

    for i in range(0, len(arr)):
        if ref[i] == -1: # duplicate detected
            # search for the next gap


print(getMinimumUniqueSum([2,2,4,5]))