import numpy as np

def average(lst):
    """ Computes the average of the values in a list of integers
    Args:
        lst: a list of integers.
    Returns:
        A float that represents the average of the values in lst.
    """
    return sum(lst) / float(len(lst))

l= [3, 7, 1 ,2 ,3]
print average(l)
print np.mean(l)



def median(lst):
    """ Computes the median of the values in a list of integers.
    Args:
        lst: a list of integers.
    Returns:
        A float that represents the median of the values in lst.
    """
    return np.median(lst)

print median([7, 2, 3, 10, 3, 30])
print median([0, 8, 9])
