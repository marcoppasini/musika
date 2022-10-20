from parse_encode import parse_args
from models import Models_functions
from utils import Utils_functions
from utils_encode import UtilsEncode_functions

if __name__ == "__main__":

    # parse args
    args = parse_args()

    # initialize networks
    M = Models_functions(args)
    models_ls = M.get_networks()

    # encode samples
    U = UtilsEncode_functions(args)
    U.compress_files(models_ls)

