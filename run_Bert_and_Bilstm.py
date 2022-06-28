import logging
from fastNLP.core.callback import FitlogCallback
def run_bert():
#数据加载
    from data_dataset_process_loader import myWeiboNERDataSet

    data_bundle=myWeiboNERDataSet().get_data_bundle()

#模型实例化
    import downstream_task_model
    bertBilstmModel=downstream_task_model.getModel_with_Bert()


#bert-fine-tuning实验优化器定义
    from fastNLP import SpanFPreRecMetric
    from torch.optim import Adam
    from fastNLP import LossInForward

    metric = SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'))
    optimizer = Adam(bertBilstmModel.parameters(), lr=2e-5)
    loss = LossInForward()

    logging.warning('--------------------------BERT--------------------------------')
#训练
    from fastNLP import Trainer
    import torch

    device= 0 if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(
        train_data=data_bundle.get_dataset('train'), 
        model=bertBilstmModel,
        loss=loss, 
        optimizer=optimizer,
        batch_size=16,
        dev_data=data_bundle.get_dataset('dev'), 
        metrics=metric, 
        device=device,
        callbacks=[FitlogCallback(data_bundle.get_dataset('test'))])
                    
    trainer.train()
    logging.warning('--------------------------训练完成--------------------------------')

#模型验证
    from fastNLP import Tester
    logging.warning('--------------------------模型测试--------------------------------')
    tester = Tester(data_bundle.get_dataset('test'), bertBilstmModel, metrics=metric)
    tester.test()
