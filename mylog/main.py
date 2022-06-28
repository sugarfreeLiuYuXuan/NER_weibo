
import fitlog

fitlog.commit(__file__)             # auto commit your codes
fitlog.set_log_dir('logs/')         # set the logging directory
fitlog.add_hyper_in_file(__file__)  # record your hyper-parameters
import sys
sys.path.append("..")
import run_word2vec_and_Bilstm
import run_Bert_and_Bilstm
import run_no

"""
Your training code here, you may use these functions to log your result:
    fitlog.add_hyper()
    fitlog.add_loss()
    fitlog.add_metric()
    fitlog.add_best_metric()
    ......
"""
# run_word2vec_and_Bilstm.run_word2vec()
# run_Bert_and_Bilstm.run_bert()
run_no.run_no()
fitlog.finish()                     # finish the logging
