import numba
import numpy as np


class MICO:
    def __init__(self, height: int, width: int, **param) -> None:
        self.H = height
        self.W = width
        self.ROI = param["ROI"] if "ROI" in param else np.ones((height, width))
        self.N = param["N"] if param.get("N") else 3
        self.Bas, self.GGT = self.getBasisFunc(height, width)
        _, _ = self.apply(np.random.random((height, width)))# 初始化时先编译一次,节省时间

    def getBasisFunc(self, height, width):
        x = np.zeros((height, width))
        y = np.zeros((height, width))
        bais = np.zeros((height, width, 10))
        B = np.zeros_like(bais)
        GGT = np.zeros((10, 10, height, width))
        for i in range(height):
            x[i, :] = np.linspace(-1, 1, width)
        for i in range(width):
            y[:, i] = np.linspace(-1, 1, height)
        bais[:, :, 0] = 1
        bais[:, :, 1] = x
        bais[:, :, 2] = (3*x*x-1)/2
        bais[:, :, 3] = (5*x*x*x - 3*x)/2
        bais[:, :, 4] = y
        bais[:, :, 5] = x*y
        bais[:, :, 6] = y*(3*x*x-1)/2
        bais[:, :, 7] = (3*y*y-1)/2
        bais[:, :, 8] = (3*y*y-1)*x/2
        bais[:, :, 9] = (5*y*y*y-3*y)/2
        for i in range(10):
            r = np.sqrt(np.sum(bais[:, :, i]**2))
            B[:, :, i] = bais[:, :, i]/r
        for i in range(10):
            for j in range(i, 10):
                GGT[i, j, :, :] = B[:, :, i]*B[:, :, j]*self.ROI
                GGT[j, i, :, :] = GGT[i, j, :, :]
        return B, GGT

    def apply(self, img, q: float = 1, max_iteration: int = 5):
        img = img.astype(np.float64)
        bias = np.ones_like(img)
        M = np.random.random((*img.shape, self.N))
        C = 255*np.random.rand(self.N)
        ImgG = calcImgG(img, self.Bas, self.ROI)

        temp = np.sum(M, axis=2)
        for i in range(self.N):
            M[:, :, i] /= temp
        M_max = argmax(M)
        for i in range(self.N):
            M[:, :, i][M_max == i] = 1
            M[:, :, i][M_max != i] = 0

        for i in range(max_iteration):
            C = updateC(img, self.ROI, bias, M, self.N)
            M = updateM(img, M, C, bias, q)
            bias = updateBias(img, M, C, self.Bas, self.GGT, ImgG, q)
        bias_field = bias*self.ROI+1e-10
        bias_corrected = (img/bias_field)*self.ROI
        return bias_field, bias_corrected.astype(np.uint8)


@numba.jit(nopython=True)
def updateBias(img, M, C, Bas, GGT, ImgG, q):
    PC2 = np.zeros_like(img)
    PC = np.zeros_like(img)

    for n in range(M.shape[2]):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                PC2[i, j] += C[n]*C[n]*(M[i, j, n]**q)
                PC[i, j] += C[n] * (M[i, j, n]**q)

    N_basis = Bas.shape[2]
    V = np.zeros((N_basis))
    A = np.zeros((N_basis, N_basis))
    for n in range(N_basis):
        for i in range(PC.shape[0]):
            for j in range(PC.shape[1]):
                V[n] += ImgG[n, i, j]*PC[i, j]

    for n in range(N_basis):
        for m in range(n, N_basis):
            for i in range(PC.shape[0]):
                for j in range(PC.shape[1]):
                    A[n, m] += GGT[n, m, i, j]*PC2[i, j]
            A[m, n] = A[n, m]

    w = np.linalg.solve(A, V)
    bias = np.zeros_like(img)
    for n in range(Bas.shape[2]):
        for i in range(Bas.shape[0]):
            for j in range(Bas.shape[1]):
                bias[i, j] += w[n]*Bas[i, j, n]

    return bias


@numba.jit(nopython=True)
def updateM(img, M, C, bias, q):
    e = np.zeros_like(M)
    for n in range(M.shape[2]):
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                e[i, j, n] = (img[i, j]-C[n]*bias[i, j])**2

    new_M = np.zeros_like(e)
    if q > 1:
        e = e+1e-10
        p = 1/(q-1)
        f = 1/(np.power(e, p))
        f_sum = np.sum(f, axis=2)
        for i in range(3):
            new_M[:, :, i] = f[:, :, i]/f_sum
    elif q == 1:
        E_min = argmin(e)
        for i in range(3):
            for m in range(M.shape[0]):
                for n in range(M.shape[1]):
                    if E_min[m, n] == i:
                        new_M[m, n, i] = 1
                    else:
                        new_M[m, n, i] = 0
    return new_M


@numba.jit(nopython=True)
def updateC(img, ROI, bias, M, N):
    C = np.zeros((N,))
    for n in range(N):
        sN = 0
        sD = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                sN += bias[i, j]*img[i, j]*M[i, j, n]*ROI[i, j]
                sD += bias[i, j]*bias[i, j]*M[i, j, n]*ROI[i, j]
        C[n] = sN/sD if sD else sN
    return C


@numba.jit(nopython=True)
def argmin(arr):
    M = np.empty((arr.shape[0], arr.shape[1]))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            M[i, j] = arr[i, j, :].argmin()
    return M


@numba.jit(nopython=True)
def argmax(arr):
    M = np.empty((arr.shape[0], arr.shape[1]))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            M[i, j] = arr[i, j, :].argmax()
    return M


@numba.jit(nopython=True)
def calcImgG(img, Bas, ROI):
    ImgG = np.zeros((10, img.shape[0], img.shape[1]))
    for n in range(10):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                ImgG[n, i, j] = img[i, j]*Bas[i, j, n]*ROI[i, j]
    return ImgG


if __name__ == '__main__':
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    img = np.array(Image.open("data/brainweb67.tif").convert("L"))
    ROI = np.zeros_like(img)
    ROI[img>=5] = 1
    mico = MICO(img.shape[0],img.shape[1],N=3,ROI=ROI)
    bias_field,output = mico.apply(img,q=1,max_iteration=5)
    fig,ax = plt.subplots(1,3)
    ax[0].imshow(img,'gray')
    ax[1].imshow(bias_field,'gray')
    ax[2].imshow(output,'gray')
    plt.savefig("data/result.png")
    plt.show()