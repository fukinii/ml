import multiprocessing
from functools import partial
from itertools import product
from itertools import repeat

def func(a, b):
    return '{} & {}'.format(a, b)

if __name__ == '__main__':
    a_args = ['Brown', 'Wilson', 'Bartlett', 'Rivera', 'Molloy', 'Opie']
    second_arg = 1
    with multiprocessing.Pool(processes=3) as pool:
        # results = pool.starmap(merge_names, product(names, repeat=2))
        M = pool.starmap(func, zip(a_args, repeat(second_arg)))
        N = pool.map(partial(func, b=second_arg), a_args)

    print(N)
    print(N)