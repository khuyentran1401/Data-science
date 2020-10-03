import os 

def make_noise():
    '''Make noise after finishing executing a code'''
    duration = 1  # seconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

def main():
    even_arr = []
    for i in range(10000):
        if i%2==0:
            even_arr.append(i)
    make_noise()

if __name__=='__main__':
    main()