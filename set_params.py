import argparse


def acm_params():
    parser = argparse.ArgumentParser(description='HeCo')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--dataset', type=str, default="acm")

    parser.add_argument('--device', type=int, default=0, help='GPU device')
    parser.add_argument('--num-hidden', type=int, default=64, help='number of hidden units')
    parser.add_argument('--feat-drop', type=float, default=0.5, help='feature dropout')
    parser.add_argument('--attn-drop', type=float, default=0.5, help='attention dropout')
    parser.add_argument('--tau', type=float, default=0.8, help='temperature parameter')
    parser.add_argument(
        '--lambda', type=float, default=0.5, dest='lambda_',
        help='balance coefficient of contrastive loss'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.5, dest='alpha_',
        help='Balance the weights between different losses'
    )
    parser.add_argument(
        '--gamma', type=float, default=0.8, dest='gamma_',
        help='Balance weights between different embeddings'
    )
    parser.add_argument('--epochs', type=int, default=10000, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0008, help='learning rate')
    parser.add_argument('--beta', type=float, default=3)
    parser.add_argument('--k', type=int, default=5)

    args = parser.parse_args()
    return args


def dblp_params():
    parser = argparse.ArgumentParser(description='HeCo')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=int, default=0, help='GPU device')
    parser.add_argument('--dataset', type=str, default="dblp")

    parser.add_argument('--num-hidden', type=int, default=64, help='number of hidden units')
    parser.add_argument('--feat-drop', type=float, default=0.5, help='feature dropout')
    parser.add_argument('--attn-drop', type=float, default=0.5, help='attention dropout')
    parser.add_argument('--tau', type=float, default=0.8, help='temperature parameter')
    parser.add_argument(
        '--lambda', type=float, default=0.5, dest='lambda_',
        help='balance coefficient of contrastive loss'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.5, dest='alpha_',
        help='Balance the weights between different losses'
    )
    parser.add_argument(
        '--gamma', type=float, default=0.8, dest='gamma_',
        help='Balance weights between different embeddings'
    )
    parser.add_argument('--epochs', type=int, default=10000, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0008, help='learning rate')
    parser.add_argument('--beta', type=float, default=1.5)
    parser.add_argument('--k', type=int, default=3)

    args = parser.parse_args()
    return args


def yelp_params():
    parser = argparse.ArgumentParser(description='HeCo')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=int, default=0, help='GPU device')
    parser.add_argument('--dataset', type=str, default="yelp")

    parser.add_argument('--num-hidden', type=int, default=64, help='number of hidden units')
    parser.add_argument('--feat-drop', type=float, default=0.2, help='feature dropout')
    parser.add_argument('--attn-drop', type=float, default=0.5, help='attention dropout')
    parser.add_argument('--tau', type=float, default=0.8, help='temperature parameter')
    parser.add_argument(
        '--lambda', type=float, default=0.5, dest='lambda_',
        help='balance coefficient of contrastive loss'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.5, dest='alpha_',
        help='Balance the weights between different losses'
    )
    parser.add_argument(
        '--gamma', type=float, default=0.6, dest='gamma_',
        help='Balance weights between different embeddings'
    )
    parser.add_argument('--epochs', type=int, default=10000, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0008, help='learning rate')
    parser.add_argument('--beta', type=float, default=4)
    parser.add_argument('--k', type=int, default=2)

    args = parser.parse_args()
    return args


def aminer_params():
    parser = argparse.ArgumentParser(description='HeCo')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=int, default=0, help='GPU device')
    parser.add_argument('--dataset', type=str, default="aminer")

    parser.add_argument('--num-hidden', type=int, default=64, help='number of hidden units')
    parser.add_argument('--feat-drop', type=float, default=0.2, help='feature dropout')
    parser.add_argument('--attn-drop', type=float, default=0.5, help='attention dropout')
    parser.add_argument('--tau', type=float, default=0.8, help='temperature parameter')
    parser.add_argument(
        '--lambda', type=float, default=0.5, dest='lambda_',
        help='balance coefficient of contrastive loss'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.9, dest='alpha_',
        help='Balance the weights between different losses'
    )
    parser.add_argument(
        '--gamma', type=float, default=0.6, dest='gamma_',
        help='Balance weights between different embeddings'
    )
    parser.add_argument('--epochs', type=int, default=10000, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0008, help='learning rate')
    parser.add_argument('--beta', type=float, default=3)
    parser.add_argument('--k', type=int, default=6)

    args = parser.parse_args()
    return args


def set_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='name of the dataset')
    args = parser.parse_args()
    dataset = args.dataset
    dataset = "yelp"
    if dataset == "acm":
        args = acm_params()
    elif dataset == "dblp":
        args = dblp_params()
    elif dataset == "aminer":
        args = aminer_params()
    elif dataset == "yelp":
        args = yelp_params()
    return args


