from parse import parse_args
from data import Data_functions
from models import Models_functions
from utils import Utils_functions
from train import Train_functions

if __name__ == "__main__":

    # parse args
    args = parse_args()

    # create dataset
    D = Data_functions(args)
    ds = D.create_dataset()

    # initialize networks
    M = Models_functions(args)
    models_ls = M.get_networks()

    # test musika in real-time during training
    U = Utils_functions(args)
    U.render_gradio(models_ls)

    # train musika
    T = Train_functions(args)
    T.train(ds, models_ls)
