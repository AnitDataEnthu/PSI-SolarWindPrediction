from datetime import datetime
import math

class utilityfunctions():

    def fn_print(string):
        print("\n-- ",string,"            ----->  ",datetime.now().strftime('%H:%M:%S'),"  <-----")

    def frameSizecalculator(window_size,image_size,stride_size):
        d=0
        x=math.ceil((image_size[d] - window_size[d]) / stride_size[d]) + 1
        d=1
        y=math.ceil((image_size[d] - window_size[d]) / stride_size[d]) + 1
        return x * y

