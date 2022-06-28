from fastNLP.io import WeiboNERPipe

class myWeiboNERDataSet():
    def __init__(self) -> None:
        self.data_bundle=WeiboNERPipe().process_from_file()
        
        
    def get_data_bundle(self):
        self.data_bundle.rename_field('chars', 'words')  
        return self.data_bundle
