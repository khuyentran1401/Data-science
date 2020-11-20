def concat(len_list):
    l = []
    for i in range(len_list):
        l = l + [i]
    return l 

def append(len_list):
    l = []
    for i in range(len_list):
        l.append(i)
    return l 

def comprehension(len_list):
    l = [i for i in range(len_list)]
    return l 
    
def list_range(len_list):
    l = list(range(len_list))
    return l 

def test_concat(benchmark):

    len_list = 1000
    res = benchmark.pedantic(concat, kwargs={'len_list': len_list}, iterations=100)
    assert res == list(range(len_list))

def test_append(benchmark):

    len_list = 1000
    res = benchmark.pedantic(append, kwargs={'len_list': len_list}, iterations=100)
    assert res == list(range(len_list))

def test_comprehension(benchmark):

    len_list = 1000
    res = benchmark.pedantic(comprehension, kwargs={'len_list': len_list}, iterations=100)
    assert res == list(range(len_list))

def test_list_range(benchmark):
    len_list = 1000
    res = benchmark.pedantic(list_range, kwargs={'len_list': len_list}, iterations=100)
    assert res == list(range(len_list))

    



