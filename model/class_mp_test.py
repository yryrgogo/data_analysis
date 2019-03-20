import os
HOME = os.path.expanduser('~')
import sys
sys.path.append(f"{HOME}/kaggle/github/library/")
from utils import parallel_process
#  import lightgbm as lgb


class multi_test:

    def __init__(self):
        self.comment_list = ['hello', 'thanks', 'good-bye']
        #  self.model = lgb

    def comment(self, comment):
        print(comment)

    def multi_comment(self, c1, c2, c3, model):
        print(c1)
        print(c2)
        print(c3)
        print(model)

    def com_wrapper(self, args):
        return self.multi_comment(*args)

    def parallel_comment(self, c_list=[]):
        #  parallel_process(self.comment, self.comment_list)
        parallel_process(self.com_wrapper, c_list, cpu_cnt=2)

