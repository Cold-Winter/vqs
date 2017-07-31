import json
from nltk.tokenize import TweetTokenizer
from gensim.models import word2vec
tokenzer = TweetTokenizer()
s0 = "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
model = word2vec.Word2Vec.load_word2vec_format('model.bin', binary=True)
print tokenzer.tokenize(s0)
with open('mscoco_train2014_annotations.json', 'r') as f:
    dataAnno = json.load(f)
with open('MultipleChoice_mscoco_train2014_questions.json', 'r') as f:
    dataQuestion = json.load(f)

feaFile = open('trainRegionFea.txt','w')
choicelist = {}
answerlist = {}
# print dataQuestion['num_choices']
for question in dataAnno['annotations']:
    choicelist[question['question_id']] = question['multiple_choice_answer']
    #print choicelist[question['question_id']]
errorword = 0
erroranswer = 0
errorquestion = 0
count = 0
for questionEn in dataQuestion['questions']:
    if count % 10000 == 0:
        print 'process questions count '+ str(count)
    count += 1
    # answerlist[question['question_id']] = question['multiple_choices']
    #print questionEn
    question = questionEn['question'][:-1].lower()
    qwordlist = tokenzer.tokenize(question)
    qFea = []
    for i in range(300):
        qFea.append(0)
    qwCount = 0
    for word in qwordlist:
        try:
            wordFeaTemp = model[word]
            for i in range(300):
                qFea[i] += wordFeaTemp[i]
        except:
            errorword += 1
        else:
            qwCount += 1
    if qwCount == 0:
        errorquestion += 1
        continue
    for i in range(300):
        qFea[i] /= qwCount
    choices = questionEn["multiple_choices"]
    for choice in choices:
        if not choice == choicelist[questionEn['question_id']]:
            cFea = []
            cwordlist = tokenzer.tokenize(choice.lower())
            for i in range(300):
                cFea.append(0)
            cwCount = 0
            for word in cwordlist:
                try:
                    wordFeaTemp = model[word]
                    for i in range(300):
                        cFea[i] += wordFeaTemp[i]
                except:
                    errorword += 1
                else:
                    cwCount += 1
            if cwCount != 0:
                for i in range(300):
                    cFea[i] /= cwCount
                for value in qFea:
                    feaFile.write(str(value)+' ')
                feaFile.write('\t')
                for value in cFea:
                    feaFile.write(str(value)+' ')
                feaFile.write('\t0\t')
                feaFile.write(str(questionEn['image_id'])+ '\t' + questionEn['question'] + '\t' + str(questionEn['question_id']) + '\n')
            else:
                erroranswer += 1
    awordlist=tokenzer.tokenize(choicelist[questionEn['question_id']].lower())
    aFea = []
    for i in range(300):
        aFea.append(0)
    awCount = 0
    for word in awordlist:
        try:
            wordFeaTemp = model[word]
            for i in range(300):
                aFea[i] += wordFeaTemp[i]
        except:
            # print 'no ' + word + ' found'
            errorword += 1
        else:
            awCount += 1
    if awCount != 0:
        for i in range(300):
            aFea[i] /= awCount
        for value in qFea:
            feaFile.write(str(value) + ' ')
        feaFile.write('\t')
        for value in aFea:
            feaFile.write(str(value) + ' ')
        feaFile.write('\t1\t')
        feaFile.write(str(questionEn['image_id'])+ '\t' + questionEn['question'] + '\t' + str(questionEn['question_id']) + '\n')
    else:
        # print 'one zero found'
        erroranswer += 1
feaFile.close()

