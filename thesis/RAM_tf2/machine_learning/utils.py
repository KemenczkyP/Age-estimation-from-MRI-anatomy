import sys
import time

def print_process(iterator, current_epoch):
    cp = iterator.current_pointer
    bs = iterator.batch_size
    dm = iterator.data_num

    xnum = int((cp+bs)*25/dm)
    spnum = (25-int((cp+bs)*25/dm))
    if(spnum<0):
        spnum = 0
        xnum = 25

    sys.stdout.write('\rEpoch:{:5};{:4}[{}{}]'.format(current_epoch,
                                                      cp,
                                                      xnum*'x',
                                                      spnum*' '))
    sys.stdout.flush()

