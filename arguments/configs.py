import argparse
import torch
def get_args():
    # region 参数设置
    parser = argparse.ArgumentParser(description="GraphMAE Model Hyperparameters")

    parser.add_argument('--d_h', type=int, default=256, help='Dimensionality of the hidden layers.')
    parser.add_argument('--hidden_dim', default=512, type=int)


    parser.add_argument('--model', type=str, default='GCMAE', help='model name')


    parser.add_argument('--input_dim', default=256, type=int)

    parser.add_argument('--output_dim', default=256, type=int)
    parser.add_argument('--final_dim', default=256, type=int)


    parser.add_argument('--num_layers', default=3, type=int)#就是lstm
    parser.add_argument('--lstm_layers', default=2, type=int)

    parser.add_argument('--input_adjuster', default=512, type=int)
    parser.add_argument('--final_adjuster', default=256, type=int)
    parser.add_argument('--enc_layers', default=2, type=int)
    parser.add_argument('--n_heads', default=2, type=int)

    # Model parameters
    parser.add_argument('--aggregator_type', type=str, default="mean", choices=["mean", "pool", "gcn"],
                        help='Aggregator type for SAGEConv.')

    # Training parameters
    parser.add_argument('--lr', type=float, default=6.485150366115917e-05, help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train the model.')
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")


    # Dataset parameters

    parser.add_argument('--mask_ratio', default=9.329807949166699e-05, type=int)
    # parser.add_argument('--alpha', default=0.5, type=int)
    parser.add_argument('--drop_ratio', default=4.978969015020039e-06, type=int)
    parser.add_argument('--G_weight', default=0.36140715531215095, type=int)


    parser.add_argument('--neg_ratio', default=2, type=int, choices=[1, 2, 3])
    parser.add_argument('--m_d', default=431, type=int)
    parser.add_argument('--d_d', default=140, type=int)

    parser.add_argument('--res_dir', default='ALRTGCL_results')
    parser.add_argument('--miRNA_sim_dir', default='dataset/dataset1/m-mmatrix.txt')
    parser.add_argument('--drug_sim_dir', default='/dataset1/d-d2matrix.txt')
    parser.add_argument('--association_m_dir', default='/dataset1/guanlianmatrix.txt')


    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--fold', default=5, type=int)



    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    print(args)
