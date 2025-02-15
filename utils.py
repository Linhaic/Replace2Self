import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def angular_neighbors(vec, n):
    """
    Returns the indices of the n closest neighbors (excluding the vector itself)
    given an array of m points with x, y and z coordinates.

    Input : A m x 3 array, with m being the number of points, one per line.
    Each column has x, y and z coordinates for each vector.

    Output : A m x n array. Each line has the n indices of
    the closest n neighbors amongst the m input vectors.

    Note : Symmetries are not considered here so a vector and its opposite sign
    counterpart will be considered far apart, even though in dMRI we consider
    (x, y, z) and -(x, y, z) to be practically identical.

    返回n个最近邻居的索引（不包括向量本身）
    给定具有x、y和z坐标的m个点的阵列。

    输入：一个m x 3的数组，m是点数，每行一个。
    每列都有每个向量的x、y和z坐标。

    输出：m x n阵列。每行有n个索引
    m个输入向量中最接近的n个邻居。

    注：此处不考虑对称性，因此矢量及其相反符号
    即使在核磁共振成像中我们考虑到了
    （x，y，z）和-（x，y，z）实际上是相同的。
    """

    # Sort the values and only keep the n closest neighbors.
    # The first angle is always 0, since _angle always
    # computes the angle between the vector and itself.
    # Therefore we pick the rest of n+1 vectors and exclude the index
    # itself if it was picked, which can happen if we have N repetition of dwis
    # but want n < N angular neighbors

    # arr1 = np.argsort(_angle(vec))[:, :n+1]
    # arr3 = np.argsort(-_angle_cosine(vec))[:, :n + 1]

    arr = np.argsort(-abs_angle(vec))[:, :n + 1]
    #上面操作后返回n个最近邻居的索引

    # arr_angle = np.argsort(_angle(vec))
    # arr_cosine_similarity = np.argsort(-_cosine_similarity(vec))
    #
    # sum=np.sum(arr_angle==arr_cosine_similarity)
    # print(sum)

    # We only want n elements - either we remove an index and return the remainder
    # or we don't and only return the n first indexes.
    output = np.zeros((arr.shape[0], n), dtype=np.int32)
    for i in range(arr.shape[0]):
        cond = i != arr[i]
        output[i] = arr[i, cond][:n]

    return output

#将余弦相似度计算结果的负数值投至正数，DMRI成像特点
def abs_angle(vec):
    return np.abs(_angle_cosine(vec))

#计算余弦相似度
def _angle_cosine(vec):
    vec = np.array(vec)
    if vec.shape[1] != 3:
        raise ValueError("Input must be of shape N x 3. Current shape is {}".format(vec.shape))
    angle=cosine_similarity(vec)
    return angle

def psnr_ssim(target,input,mask=None,multichannel=True):
    if mask is not None:
        target=target*mask
        input=input*mask
    if np.max(target)==np.min(target):
       return -1,-1
    target = (target - np.min(target)) / (np.max(target) - np.min(target))
    input = (input - np.min(input)) / (np.max(input) - np.min(input))
    psnr = compare_psnr(target,input,data_range=(np.max(target)-np.min(target)))
    ssim=compare_ssim(target,input,data_range=(np.max(target)-np.min(target)),multichannel=multichannel)
    return psnr,ssim

class Parser:
    def __init__(self, parser):
        self.__parser = parser
        self.__args = parser.parse_args()

        # set gpu ids
        str_ids = self.__args.gpu_ids.split(',')
        self.__args.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.__args.gpu_ids.append(id)
        # if len(self.__args.gpu_ids) > 0:
        #     torch.cuda.set_device(self.__args.gpu_ids[0])

    def get_parser(self):
        return self.__parser

    def get_arguments(self):
        return self.__args


    def print_args(self, name='PARAMETER TABLES'):
        params_dict = vars(self.__args)

        print('----' * 10)
        print('{0:^40}'.format(name))
        print('----' * 10)
        for k, v in sorted(params_dict.items()):
            if '__' not in str(k):
                print('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)))
        print('----' * 10)