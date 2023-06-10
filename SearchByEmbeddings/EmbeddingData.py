from CollectDataFromWord import CollectData,CollectHeading
from os import path

def EmbeddingData(filepath):
    embedpath='embeddings'
    inputfilename=path.basename(filepath)
    outfilename=inputfilename.replace("docx","csv")

    iscreated=path.exists(path.join(embedpath,outfilename))
    if not  iscreated:
        CollectData(filepath,path.join(embedpath,outfilename)).embedchunks()
        return True
    else:
        return True

#EmbeddingData('data\密云区矿区转型绿色数字产业示范园区可行性研究报告.docx')
filepath=r'data\码头项目可行性研究报告.docx'

title=CollectHeading(filepath,"").getheadings()
print(title)