from CollectDataFromWord import CollectData
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

EmbeddingData('data\城市建筑信息在城市作战中的应用.docx')