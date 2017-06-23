import csv
import re
import time
class DataExtraction:    
    def preprocessing(self,path):
        corpus=[]
        
        with open(path,'r') as input_file:
            curr_fields=[];
        
            for line in input_file:
                    line=re.sub('"','',line)
                    curr_fields.append(line);
                    if( not line.strip()):
                        corpus.append(curr_fields)
                        curr_fields=[];
        return corpus
        

start=time.time()

extract=DataExtraction()
corpus_file_path=raw_input('Enter the Path to the Corpus\n')

raw_corpus=extract.preprocessing(corpus_file_path)
print 'The corpus contains  ',len(raw_corpus),' articles'
c = csv.writer(open("corpus.csv", "w"))
c.writerow(["body","tag"])

print '\nProcessing the corpus and writing it to CSV...'

for row in raw_corpus:
        c.writerow(row)
print '\nThe training data has been loaded into corpus.csv'
print '\nTime to extract data was:',time.time()-start,' seconds'