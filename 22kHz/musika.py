from parse_test import parse_args
from models import Models_functions
from utils import Utils_functions

if __name__ == "__main__":

    # parse args
    args = parse_args()

    # initialize networks
    M = Models_functions(args)
    models_ls = M.get_networks()

    # test musika
    U = Utils_functions(args)
    U.render_gradio(models_ls, train=False)
