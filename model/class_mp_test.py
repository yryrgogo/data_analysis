import os
HOME = os.path.expanduser('~')
import sys
sys.path.append(f"{HOME}/kaggle/github/library/")
from utils import pararell_process


class multi_test:

    def __init__(self):
        self.comment_list = ['hello', 'thanks', 'good-bye']

    def comment(self, comment):
        print(comment)

    def pararell_comment(self, comment_list=[]):
        pararell_process(self.comment, self.comment_list)

