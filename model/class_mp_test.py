import os
HOME = os.path.expanduser('~')
import sys
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
from pararell_utils import pararell_process


class multi_test:

    def __init__(self):
        self.comment_list = ['hello', 'thanks', 'good-bye']

    def comment(self, comment):
        print(comment)

    def multi_comment(self, c1, c2, c3, model):
        print(c1)
        print(c2)
        print(c3)
        print(model)

    def com_wrapper(self, args):
        return self.multi_comment(*args)

    def pararell_comment(self, c_list=[]):
        #  pararell_process(self.comment, self.comment_list)
        pararell_process(self.com_wrapper, c_list, cpu_cnt=2)

