'''
Created on Sep 18, 2014

@author: bjw
'''

def parseFile(filename):
    f = open(filename)
    text = f.readlines()
    f.close()
    data = []
    for line in text:
        row = line.split()
        data.append([float(x) for x in row])
    return data
