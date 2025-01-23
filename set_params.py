import argparse


def acm_params(parser):
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    parser.add_argument('--device', type=int, default=0, help='GPU device')
    parser.add_argument('--num-hidden', type=int, default=64, help='number of hidden units')
    parser.add_argument('--feat-drop', type=float, default=0.5, help='feature dropout')
    parser.add_argument('--attn-drop', type=float, default=0.5, help='attention dropout')
    parser.add_argument('--tau', type=float, default=0.8, help='temperature parameter')
    parser.add_argument(
        '--lambda', type=float, default=0.6, dest='lambda_',
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
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta', type=float, default=3)
    parser.add_argument('--k', type=int, default=5)

    return parser


def dblp_params(parser):
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=int, default=0, help='GPU device')

    parser.add_argument('--num-hidden', type=int, default=64, help='number of hidden units')
    parser.add_argument('--feat-drop', type=float, default=0.5, help='feature dropout')
    parser.add_argument('--attn-drop', type=float, default=0.5, help='attention dropout')
    parser.add_argument('--tau', type=float, default=0.8, help='temperature parameter')
    parser.add_argument(
        '--lambda', type=float, default=0.6, dest='lambda_',
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
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta', type=float, default=3)
    parser.add_argument('--k', type=int, default=5)

    return parser


def yelp_params(parser):
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=int, default=0, help='GPU device')

    parser.add_argument('--num-hidden', type=int, default=64, help='number of hidden units')
    parser.add_argument('--feat-drop', type=float, default=0.2, help='feature dropout')
    parser.add_argument('--attn-drop', type=float, default=0.5, help='attention dropout')
    parser.add_argument('--tau', type=float, default=0.8, help='temperature parameter')
    parser.add_argument(
        '--lambda', type=float, default=0.6, dest='lambda_',
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
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta', type=float, default=3)
    parser.add_argument('--k', type=int, default=5)

    return parser


def aminer_params(parser):
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=int, default=0, help='GPU device')

    parser.add_argument('--num-hidden', type=int, default=64, help='number of hidden units')
    parser.add_argument('--feat-drop', type=float, default=0.2, help='feature dropout')
    parser.add_argument('--attn-drop', type=float, default=0.5, help='attention dropout')
    parser.add_argument('--tau', type=float, default=0.8, help='temperature parameter')
    parser.add_argument(
        '--lambda', type=float, default=0.6, dest='lambda_',
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
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta', type=float, default=3)
    parser.add_argument('--k', type=int, default=5)

    return parser


def set_params():
    """Parse and dynamically set parameters based on dataset."""
    # Base parser for common arguments
    parser = argparse.ArgumentParser(description="HGNN-DB Parameters")
    parser.add_argument('--dataset', type=str, default='acm', help='name of the dataset')
    parser.add_argument('--strategy', type=str, default='rand')# cand

    # Parse the dataset argument first
    args, remaining_args = parser.parse_known_args()

    # Add dataset-specific parameters
    if args.dataset == "acm":
        parser = acm_params(parser)  # Keep `parser` as an ArgumentParser
    elif args.dataset == "dblp":
        parser = dblp_params(parser)  # Ensure dblp_params returns ArgumentParser
    elif args.dataset == "aminer":
        parser = aminer_params(parser)  # Ensure aminer_params returns ArgumentParser
    elif args.dataset == "yelp":
        parser = yelp_params(parser)  # Ensure yelp_params returns ArgumentParser
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Re-parse arguments with the updated parser
    args = parser.parse_args(remaining_args)
    return args
