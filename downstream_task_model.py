from statistics import mode
import torch.nn as nn
#模型定义
from fastNLP.models import BiLSTMCRF


#数据加载以获得词嵌入
from data_dataset_process_loader import myWeiboNERDataSet

data_bundle=myWeiboNERDataSet().get_data_bundle()

#embedding加载
from fastNLP.embeddings import StaticEmbedding
from fastNLP.embeddings import BertEmbedding

embed_no= StaticEmbedding(vocab=data_bundle.get_vocab('words'),model_dir_or_name=None,embedding_dim=768)
embed_word2vec = StaticEmbedding(vocab=data_bundle.get_vocab('words'), model_dir_or_name='cn-char-fastnlp-100d')
embed_Bert = BertEmbedding(vocab=data_bundle.get_vocab('words'), model_dir_or_name='cn')
embed_Bert.requires_grad = True

def getModel_with_no():
    model= BiLSTMCRF(embed=embed_no, 
        num_classes=len(data_bundle.get_vocab('target')), 
        num_layers=1, 
        hidden_size=200, 
        dropout=0.5,
        target_vocab=data_bundle.get_vocab('target'))
    return model
def getModel_with_word2vec():
    model= BiLSTMCRF(embed=embed_word2vec, 
        num_classes=len(data_bundle.get_vocab('target')), 
        num_layers=1, 
        hidden_size=200, 
        dropout=0.5,
        target_vocab=data_bundle.get_vocab('target'))
    return model

def getModel_with_Bert():
    model= BiLSTMCRF(
        embed=embed_Bert, 
        num_classes=len(data_bundle.get_vocab('target')),
        num_layers=1, 
        hidden_size=200, 
        dropout=0.5,
        target_vocab=data_bundle.get_vocab('target'))
    return model