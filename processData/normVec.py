import math
normfile = open('trainRegionFeaNorm.txt', 'w')
count = 0
linelist = []
for line in open('trainRegionFea.txt'):
    linelist.append(line)
outlist = []
for line in linelist:
    if count % 100000 == 0:
        print count
    count += 1
    lines = line.split('\t')
    qfea = lines[0].split()
    afea = lines[1].split()
    feaSum = 0
    outtemp = ''
    # l2 norm to quesion feature and answer feature
    for fea in qfea:
        feaSum += float(fea) * float(fea)
    feaSum = math.sqrt(feaSum)
    for i in range(len(qfea)):
        qfea[i] = float(qfea[i])/feaSum
        outtemp += str(qfea[i])+' '
    outtemp+='\t'
    feaSum = 0
    for fea in afea:
        feaSum += float(fea) * float(fea)
    feaSum = math.sqrt(feaSum)
    for i in range(len(afea)):
        afea[i] = float(afea[i])/feaSum
        outtemp+=str(afea[i]) + ' '
    outtemp+='\t'
    #print len(lines)
    outtemp+=lines[2]+'\t'+lines[3]+'\t'+lines[4]+'\t'+lines[5]
    outlist.append(outtemp)
for line in outlist:
    normfile.write(line)






