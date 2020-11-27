import pytest 
import random 

def extend(l1,l2):
    l1.extend(l2)
    return l1
def test_extend():
    l1 = [1,2,3]
    l2 = [4,5,6]
    res = extend(l1, l2)
    assert res == [1,2,3,4,5,6]


@pytest.mark.repeat(100)
def test_extend_random():
    l1 = []
    l2 = []
    for _ in range(0,3):
        n = random.randint(1,10)
        l1.append(n)
        n = random.randint(1,10)
        l2.append(n)

    res = extend(l1,l2)
    
    assert res[-(len(l2)):] == l2

    