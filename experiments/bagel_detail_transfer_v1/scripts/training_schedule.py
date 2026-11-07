"""Randomize task exposure order without changing any image's supervision budget."""
import random
from supervision import make_pack,number


def training_pack(row,arm,epoch):
    cycle,position=divmod(epoch,3)
    order=[0,1,2]
    random.Random(number(f'{row["id"]}/{cycle}/exposure_order')).shuffle(order)
    variant=cycle*3+order[position]
    pack=make_pack(row,arm,variant)
    random.Random(number(f'{row["id"]}/{variant}/question_order')).shuffle(pack['tasks'])
    pack['physical_epoch']=epoch
    pack['supervision_variant']=variant
    return pack
