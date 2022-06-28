import logging
from fastNLP.core.callback import FitlogCallback


def run_no():
    #数据加载
    from data_dataset_process_loader import myWeiboNERDataSet

    data_bundle=myWeiboNERDataSet().get_data_bundle()

    #模型实例化
    import downstream_task_model
    noBilstmModel=downstream_task_model.getModel_with_no()


    #对比实验优化器定义
    from fastNLP import SpanFPreRecMetric
    from torch.optim import Adam
    from fastNLP import LossInForward

    metric = SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'))
    optimizer = Adam(noBilstmModel.parameters(), lr=1e-2)
    loss = LossInForward()

    logging.warning('---------------------------No-------------------------------')
    #训练
    from fastNLP import Trainer
    import torch

    device= 0 if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(
        train_data=data_bundle.get_dataset('train'), 
        model=noBilstmModel,
        loss=loss, 
        optimizer=optimizer,
        batch_size=16,
        dev_data=data_bundle.get_dataset('dev'), 
        metrics=metric, 
        device=device,
        callbacks=[FitlogCallback(data_bundle.get_dataset('test'))])
                    
    trainer.train()
    logging.warning('----------------------------------------------------------')

#模型验证
    from fastNLP import Tester
    logging.warning('--------------------------模型测试--------------------------------')
    tester = Tester(data_bundle.get_dataset('test'), noBilstmModel, metrics=metric)
    tester.test()

