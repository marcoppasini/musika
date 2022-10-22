import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from parse.parse_decode import parse_args
from models import Models_functions
from utils import Utils_functions

if __name__ == "__main__":

    # parse args
    args = parse_args()

    # initialize networks
    M = Models_functions(args)
    M.download_networks()
    models_ls = M.get_networks()

    # encode samples
    U = Utils_functions(args)
    U.decode_path(models_ls)
