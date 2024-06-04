import numpy as np

def readcfl(name):
    # get dims from .hdr
    with open(name + ".hdr", "rt") as h:
        h.readline() # skip
        l = h.readline()
    dims = [int(i) for i in l.split()]

    # remove singleton dimensions from the end
    n = np.prod(dims)
    dims_prod = np.cumprod(dims)
    dims = dims[:np.searchsorted(dims_prod, n)+1]

    # load data and reshape into dims
    with open(name + ".cfl", "rb") as d:
        a = np.fromfile(d, dtype=np.complex64, count=n);
    return a.reshape(dims, order='F') # column-major

def float2cplx(float_in):
    return np.array(float_in[...,0]+1.0j*float_in[...,1], dtype='complex64')
    
def cplx2float(cplx_in):
    return np.array(np.stack((cplx_in.real, cplx_in.imag), axis=-1), dtype='float32')


# read cfl file and plot it as image
import matplotlib.pyplot as plt
import sys

def main(cfl, img_path):    
        input_tensor = readcfl(cfl).squeeze()
        plt.imshow(abs(input_tensor), cmap="gray")
        plt.savefig(img_path)

if __name__ == "__main__":
    # parse command line arguments
    if len(sys.argv) != 3:
        print("Usage: python utils.py <cfl> <save_img_path>")
        sys.exit(1)

    cfl = sys.argv[1]
    img_path = sys.argv[2]
    main(sys.argv[1], sys.argv[2])