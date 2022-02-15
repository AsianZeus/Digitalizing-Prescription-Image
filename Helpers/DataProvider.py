import os
import numpy as np
import cv2


class DataProvider():
	"this class creates machine-written text for a word list. TODO: change getNext() to return your samples."

	def __init__(self, wordList,wordImage):
		self.wordList = wordList
		self.wordImage = wordImage
		self.idx = 0

	def hasNext(self):
		"are there still samples to process?"
		return self.idx < len(self.wordList)

	def getNext(self):
		"TODO: return a sample from your data as a tuple containing the text and the image"
		img = self.wordImage[self.idx]
		word = self.wordList[self.idx]
		self.idx += 1
		return (word, img)


def createIAMCompatibleDataset(dataProvider):
	"this function converts the passed dataset to an IAM compatible dataset"
	
	path='X/XX/XXXX/XXX/'
	# create files and directories
	f = open(path+'words.txt', 'w+')
	if not os.path.exists(path+'sub'):
		os.makedirs(path+'sub')
	if not os.path.exists(path+'sub/sub-sub'):
		os.makedirs(path+'sub/sub-sub')

	# go through data and convert it to IAM format
	ctr = 0
	while dataProvider.hasNext():
		sample = dataProvider.getNext()
		
		# write img
		print(ctr,sample[0])
		cv2.imwrite(f"{path}sub/sub-sub/sub-sub-{ctr}.png", sample[1])
		
		# write filename, dummy-values and text
		line = 'sub-sub-%d'%ctr + ' X X X X X X X ' + sample[0] + '\n'
		f.write(line)
		
		ctr += 1


if __name__ == '__main__':
	wordslist=[]
	wordsimage=[]
	path=os.getcwd()+'/Desktop/Grayscale/'
	filename= os.listdir(path)
	imgcounter=0
	for namex in filename:
		name=namex.split('_')[0]
		imagex=cv2.imread(path+namex)
		wordslist.append(name)
		wordsimage.append(imagex)
		imgcounter+=1
	print(wordslist)
	dataProvider = DataProvider(wordslist,wordsimage)
	createIAMCompatibleDataset(dataProvider)