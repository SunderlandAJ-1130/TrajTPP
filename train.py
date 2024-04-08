# coding=utf8
import dpp
import numpy as np
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from dpp.utils import evaluation_tpp, get_total_loss, setup_seed, get_logger
import os
from tqdm import tqdm
import warnings
import time
sns.set_style('whitegrid')
os.environ['KMP_DUPLICATE_LIB_OK']='True'
warnings.filterwarnings('ignore')
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# define paraser
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--max_epochs', type=int, default=5000)
parser.add_argument('--use_marks', type=bool, default=True)
parser.add_argument('--decoder_name', type=str, default='LogNormMix')
parser.add_argument('--rnn_type', type=str, default='GRU')
parser.add_argument('--attention_type', type=str, default='GAU')
parser.add_argument('--use_history', action='store_false', help='use_history or not')
parser.add_argument('--history_size', type=int, default=128)
parser.add_argument('--use_embedding', type=bool, default=False)
parser.add_argument('--embedding_size', type=int, default=32)
parser.add_argument('--trainable_affine', action='store_true', help='trainable_affine or not')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--n_components', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--regularization', type=float, default=1e-5)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--num_classes', type=int, default=1187)
parser.add_argument('--num_heads', type=int, default=1)
parser.add_argument('--mark_embedding_size', type=int, default=128)
parser.add_argument('--log_dir', type=str, default='GRU')
parser.add_argument('--grn_decoder', type=str, default='GAU')
parser.add_argument('--dataset_name', type=str, default='./dataset/geolife')
parser.add_argument('--use_sa', action='store_true', help='use_sa or not')
parser.add_argument('--use_timeofday', action='store_true', help='use_timeofday or not')
parser.add_argument('--use_dayofweek', action='store_true', help='use_dayofweek or not')
parser.add_argument('--use_driver', action='store_true', help='use_driver or not')
parser.add_argument('--pre_embedding', action='store_true', help='Use pre_embedding or not')
parser.add_argument('--joint_type', type=str, default='None', help='Joint for mark and time NLL')
parser.add_argument('--time_threshold', type=int, default=15)
parser.add_argument('--trip_threshold', type=int, default=5)
parser.add_argument('--min_clip', type=float, default=-5.0)
parser.add_argument('--max_clip', type=float, default=3.0)
parser.add_argument('--use_prior', action='store_true', help='use_prior or not')
parser.add_argument('--use_mtl', action='store_true', help='use multi-task learning or not.')
args = parser.parse_args()
device = f'cuda:{str(args.device)}'


def main(args, logger):  # sourcery skip: low-code-quality
    dataset_name = f'{args.dataset_name}-{args.time_threshold}-{args.trip_threshold}/'

    ## General model config
    use_history = args.use_history              # Whether to use RNN to encode history
    history_size = args.history_size            # Size of the RNN hidden vector
    rnn_type = args.rnn_type                    # Which RNN cell to use (other: ['GRU', 'LSTM'])
    use_embedding = args.use_embedding          # Whether to use sequence embedding (should use with 'each_sequence' split)
    embedding_size = args.embedding_size        # Size of the sequence embedding vector
    trainable_affine = args.trainable_affine    # Train the final affine layer
    batch_size = args.batch_size

    ## Decoder config
    decoder_name = args.decoder_name            # other: ['RMTPP', 'FullyNeuralNet', 'Exponential', 'SOSPolynomial', 'DeepSigmoidalFlow']
    n_components = args.n_components            # Number of components for a mixture model
    hypernet_hidden_sizes = []                  # Number of units in MLP generating parameters ([] -- affine layer, [64] -- one layer, etc.)

    ## Flow params
    # Polynomial
    max_degree = 3                              # Maximum degree value for Sum-of-squares polynomial flow (SOS)
    n_terms = 4                                 # Number of terms for SOS flow
    # DSF / FullyNN
    n_layers = 2                                # Number of layers for Deep Sigmoidal Flow (DSF) / Fully Neural Network flow (Omi et al., 2019)
    layer_size = 64                             # Number of mixture components / units in a layer for DSF and FullyNN

    ## Training config
    regularization = args.regularization        # L2 regularization parameter
    learning_rate = args.learning_rate          # Learning rate for Adam optimizer
    max_epochs = args.max_epochs                # For how many epochs to train
    display_step = 1                            # Display training statistics after every display_step
    patience = args.patience                    # After how many consecutive epochs without improvement of val loss to stop training

    logger.info('load data...')

    # load data
    d_train, num_drivers = dpp.data.load_dataset(f'{dataset_name}train', device=device)
    d_val, _ = dpp.data.load_dataset(f'{dataset_name}val', device=device)
    d_test, _ = dpp.data.load_dataset(f'{dataset_name}test', device=device)

    logger.info(
        f'num_drivers: {num_drivers}, d_train: {len(d_train.in_times[0])}, d_val: {len(d_val.in_times[0])}, d_test: {len(d_test.in_times[0])}'
    )

    # Calculate mean and std of the input inter-event times and normalize only input
    mean_in_train, std_in_train = d_train.get_mean_std_in()
    std_out_train = 1.0
    d_train.normalize(mean_in_train, std_in_train, std_out_train)
    d_val.normalize(mean_in_train, std_in_train, std_out_train)
    d_test.normalize(mean_in_train, std_in_train, std_out_train)
    args.mean_in_train = mean_in_train
    args.std_in_train = std_in_train

    # Define model
    logger.info('Building model...')
    logger.info(
        f'mean_log_inter_time: {mean_in_train}, std_log_inter_time: {std_in_train}'
    )

    # Break down long train sequences for faster batch traning and create torch DataLoaders
    d_train.break_down_long_sequences(128)
    collate = dpp.data.collate
    dl_train = torch.utils.data.DataLoader(d_train, batch_size=batch_size, shuffle=True, collate_fn=collate, generator=torch.Generator(device=device))
    dl_val = torch.utils.data.DataLoader(d_val, batch_size=batch_size, shuffle=False, collate_fn=collate, generator=torch.Generator(device=device))
    dl_test = torch.utils.data.DataLoader(d_test, batch_size=1, shuffle=False, collate_fn=collate, generator=torch.Generator(device=device))

    # Set the parameters for affine normalization layer depending on the decoder (see Appendix D.3 in the paper)
    if decoder_name in ['RMTPP', 'FullyNeuralNet', 'Exponential']:
        _, std_out_train = d_train.get_mean_std_out()
        mean_out_train = 0.0
    else:
        mean_out_train, std_out_train = d_train.get_log_mean_std_out()

    # model setup
    # General model config
    general_config = dpp.model.ModelConfig(
        use_history=use_history,
        history_size=history_size,
        rnn_type=rnn_type,
        use_embedding=use_embedding,
        embedding_size=embedding_size,
        num_embeddings=len(d_train)+len(d_val)+len(d_test),
        use_marks=args.use_marks,
        num_classes=args.num_classes,
        mark_embedding_size=args.mark_embedding_size,
        device=device,
        use_sa=args.use_sa,
        num_heads=args.num_heads,
        use_driver=args.use_driver,
        use_timeofday=args.use_timeofday,
        use_dayofweek=args.use_dayofweek,
        num_driver=num_drivers,
        max_clip=args.max_clip,
        min_clip=args.min_clip,
        decoder_name=args.decoder_name,
        attention_type=args.attention_type,
        use_prior=args.use_prior,
        mean_in_train=args.mean_in_train,
        std_in_train=args.std_in_train,
        pre_embedding=args.pre_embedding,
        joint_type=args.joint_type,
        logger=logger
    )

    # Decoder specific config
    decoder = getattr(dpp.decoders, decoder_name)(general_config,
                                                n_components=n_components,
                                                hypernet_hidden_sizes=hypernet_hidden_sizes,
                                                max_degree=max_degree,
                                                n_terms=n_terms,
                                                n_layers=n_layers,
                                                layer_size=layer_size,
                                                shift_init=mean_out_train,
                                                scale_init=std_out_train,
                                                use_marks=args.use_marks,
                                                trainable_affine=trainable_affine)
    decoder = decoder.to(device)

    # Define model
    model = dpp.model.Model(general_config, decoder).to(device)
    model.use_history(general_config.use_history)
    model.use_embedding(general_config.use_embedding)
    logger.info(model)

    # Define optimizer
    opt = torch.optim.Adam(model.parameters(), weight_decay=regularization, lr=learning_rate)

    # model training
    impatient = 0
    best_loss = np.inf
    best_model = deepcopy(model.state_dict())
    training_val_losses = []
    training_test_losses = []

    num_batch = dl_train.__len__()
    
    avg_cost = np.zeros([max_epochs, 2], dtype=np.float32)
    lambda_weight = np.ones([avg_cost.shape[1], max_epochs])
    T = 2.0
    train_batch = len(dl_train)
    val_batch = len(dl_val)

    # Traning
    logger.info('Starting training...')

    for epoch in range(max_epochs):
        cost = np.zeros(avg_cost.shape[1], dtype=np.float32)

        if epoch == 0 or epoch == 1:
            lambda_weight[:, epoch] = 1.0
        else:
            w_1 = avg_cost[epoch - 1, 0] / avg_cost[epoch - 2, 0]
            w_2 = avg_cost[epoch - 1, 1] / avg_cost[epoch - 2, 1]
            lambda_weight[0, epoch] = 2 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))
            lambda_weight[1, epoch] = 2 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))

        model.train()
        with tqdm(total=num_batch, ascii=' =') as pbar:
            for input in dl_train:
                input.in_mark = input.in_mark.to(device)
                input.in_time = input.in_time.to(device)
                input.index = input.index.to(device)
                input.out_mark = input.out_mark.to(device)
                input.out_time = input.out_time.to(device)

                opt.zero_grad()
                
                if args.use_mtl is True:
                    if args.use_marks:
                        log_prob, mark_nll =  model.log_prob(input)
                        log_prob_loss = -model.aggregate(log_prob, input.length)
                        mark_nll_loss = -model.aggregate(mark_nll, input.length)

                        train_loss = [log_prob_loss, mark_nll_loss]
                        loss = sum([lambda_weight[i, epoch] * train_loss[i] for i in range(len(train_loss))])
                    else:
                        log_prob = model.log_prob(input)
                        loss = -model.aggregate(log_prob, input.length)

                    cost[0] = train_loss[0].item()
                    cost[1] = train_loss[1].item()
                    avg_cost[epoch, :2] += cost[:2] / train_batch             
                else:
                    if args.use_marks:
                        log_prob, mark_nll = model.log_prob(input)
                        log_prob += mark_nll
                    else:
                        log_prob = model.log_prob(input)

                    loss = -model.aggregate(log_prob, input.length)

                loss.backward()
                opt.step()
                pbar.update(1)

        model.eval()

        if args.decoder_name == 'FullyNeuralNet':
            loss_val = get_total_loss(model, dl_val, use_marks=args.use_marks)
        else:
            with torch.no_grad():
                if args.use_mtl:
                    loader_log_prob, loader_mark_nll, loader_lengths = [], [], []
                    for input in dl_val:
                        output = model.log_prob(input)
                        loader_log_prob.append(output[0].detach())
                        loader_mark_nll.append(output[1].detach())
                        loader_lengths.append(input.length.detach())
                    loss_val_time = -model.aggregate(loader_log_prob, loader_lengths)
                    loss_val_mark = -model.aggregate(loader_mark_nll, loader_lengths)

                    val_loss = [loss_val_time, loss_val_mark]
                    cost[2] = val_loss[0].item()
                    cost[3] = val_loss[1].item()
                    avg_cost[epoch, 2:] += cost[2: ] / val_batch  
                    
                    loss_val = sum([lambda_weight[i, epoch] * val_loss[i] for i in range(len(val_loss))])   
                else:
                    loss_val = get_total_loss(model, dl_val, use_marks=args.use_marks)
                # loss_test = get_total_loss(model, dl_test, use_marks=args.use_marks)

        training_val_losses.append(loss_val.item())
        # training_test_losses.append(loss_test.item())
        if (best_loss - loss_val) < 0:
            impatient += 1
            if loss_val < best_loss:
                best_loss = loss_val.item()
                best_model = deepcopy(model.state_dict())
        else:
            best_loss = loss_val.item()
            best_model = deepcopy(model.state_dict())
            impatient = 0
        torch.save(best_model, 'best_model.pkl')

        if impatient >= patience:
            logger.info(f'Breaking due to early stopping at epoch {epoch}')
            break

        if (epoch) % display_step == 0:
            if args.use_mtl:
                logger.info('Epoch: %d, loss_train_last_batch: %.4f, loss_val: %.4f, best_loss: %.4f, lambda_weight_0: %.3f, lambda_weight_1: %.3f, impatient: %d' % (
                    epoch+1, loss, loss_val, best_loss, lambda_weight[0, epoch], lambda_weight[1, epoch], impatient))
            else:
                logger.info('Epoch: %d, loss_train_last_batch: %.4f, loss_val: %.4f, best_loss: %.4f, impatient: %d' % (
                    epoch+1, loss, loss_val, best_loss, impatient))

    # evaluation
    model.load_state_dict(best_model)
    model.eval()

    # All training & testing sequences stacked into a single batch
    if args.decoder_name == 'FullyNeuralNet':
        pdf_loss_train = get_total_loss(model, dl_train, use_marks=args.use_marks)
        pdf_loss_val = get_total_loss(model, dl_val, use_marks=args.use_marks)
        pdf_loss_test = get_total_loss(model, dl_test, use_marks=args.use_marks)
    else:
        with torch.no_grad():
            pdf_loss_train = get_total_loss(model, dl_train, use_marks=args.use_marks)
            pdf_loss_val = get_total_loss(model, dl_val, use_marks=args.use_marks)
            pdf_loss_test, test_tau_nll, test_mark_nll = get_total_loss(model, dl_test, use_marks=args.use_marks, return_all=True)

    logger.info(f'Negative log-likelihood:\n'
          f' - Train: {pdf_loss_train:.3f}\n'
          f' - Val:   {pdf_loss_val:.3f}\n'
          f' - Test:  {pdf_loss_test:.3f}\n'
          f'    - Mark NLL: {test_mark_nll:.3f}\n'
          f'    - Time NLL: {test_tau_nll:.3f}')

    training_val_losses = training_val_losses[:-patience] # plot only until early stopping
    fig = plt.figure(dpi=300)
    plt.plot(range(len(training_val_losses)), training_val_losses)
    plt.ylabel('Validation loss')
    plt.xlabel('Epoch')
    plt.title(f'Training on "{dataset_name}" dataset')
    plt.close()
    fig.savefig(f'{args.log_path}training_process.pdf')

    np.save(f'{args.log_path}training_val_losses.npy', arr=training_val_losses)
    np.save(f'{args.log_path}training_test_losses.npy', arr=training_test_losses)
    torch.save(model, f'{args.log_path}model.pkl')

    # eval
    evaluation_tpp(
        model=model,
        test_dataset=dl_test,
        decoder_name=args.decoder_name,
        std_in_train=std_in_train,
        mean_in_train=mean_in_train,
        log_path=args.log_path,
        logger=logger)

    torch.save(model, args.log_path + args.log_filename[:-4]+'.pkl')
    np.save(f'{args.log_path}loss_val.npy', arr=training_val_losses)
    np.save(f'{args.log_path}loss_test.npy', arr=training_test_losses)
    logger.info('finished...')


if __name__ == '__main__':
    for seed in range(args.seed, args.seed+5):
        args.seed = seed
        setup_seed(args.seed)
        log_path = f'./{args.log_dir}-{args.time_threshold}-{args.trip_threshold}/'

        if os.path.exists(log_path) is False:
            os.makedirs(log_path)

        log_filename = 'log-{:s}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
        args.log_filename = log_filename
        args.log_path = log_path

        if os.path.exists(log_path+args.log_filename[:-4]) is False:
            os.makedirs(log_path+args.log_filename[:-4])

        args.log_path += f'{args.log_filename[:-4]}/'

        logger = get_logger(args.log_path, __name__, args.log_filename)
        logger.info(args)

        # select device
        if torch.cuda.is_available():
            cuda = args.device
            # cuda = auto_select_gpu()
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
            logger.info(f"Using GPU {os.environ['CUDA_VISIBLE_DEVICES']}")
            device = torch.device(f'cuda:{cuda}')

            if args.device == 'cpu':            
                device = torch.device('cpu')
        else:
            logger.info('Using CPU')
            device = torch.device('cpu')

        main(args, logger)
        logger = None
