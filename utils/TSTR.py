import torch
from tqdm import tqdm, trange
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

class GeneralRNN(torch.nn.Module):
    """
    A general RNN model for time-series prediction
    """

    def __init__(self, args):
        super(GeneralRNN, self).__init__()

        self.model_type = args['model_type']
        self.input_size = args['in_dim']
        self.hidden_size = args['h_dim']
        self.output_size = args['out_dim']
        self.num_layers = args['n_layers']
        self.dropout = args['dropout']
        self.bidirectional = args['bidirectional']

        self.max_seq_len = args['max_seq_len']

        self.rnn_module = self._get_rnn_module(self.model_type)

        self.rnn_layer = self.rnn_module(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )

        self.linear_layer = torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=self.output_size
        )

    def _get_rnn_module(self, model_type):
        if model_type == "rnn":
            return torch.nn.RNN
        elif model_type == "lstm":
            return torch.nn.LSTM
        elif model_type == "gru":
            return torch.nn.GRU

    def forward(self, X):
        # Dynamic RNN input for ignoring paddings
        H_o, H_t = self.rnn_layer(X)
        logits = self.linear_layer(H_o)

        return logits

def rmse_error(y_true, y_pred):
    """User defined root mean squared error.
    Args:
    - y_true: true labels
    - y_pred: predictions
    Returns:
    - computed_rmse: computed rmse loss
    """
    # Exclude masked labels
    idx = (y_true >= 0) * 1
    # Mean squared loss excluding masked labels
    computed_mse = np.sum(idx * ((y_true - y_pred) ** 2)) / np.sum(idx)
    computed_rmse = np.sqrt(computed_mse)
    return computed_rmse

def one_step_ahead_prediction(train_data, test_data):
    """Use the previous time-series to predict one-step ahead feature values.
    Args:
    - train_data: training time-series
    - test_data: testing time-series
    Returns:
    - perf: average performance of one-step ahead predictions (in terms of AUC or MSE)
    """
    train_data = train_data
    test_data = test_data

    # Parameters
    no, seq_len, dim = 256, 30, 92

    # Set model parameters
    args = {}
    args["device"] = "cuda"
    args["task"] = "regression"
    args["model_type"] = "gru"
    args["bidirectional"] = False
    args["epochs"] = 2000
    args["batch_size"] = 256
    args["in_dim"] = dim
    args["h_dim"] = 30 # dim
    args["out_dim"] = dim
    args["n_layers"] = 3
    args["dropout"] = 0.3
    args["max_seq_len"] = 30  # only 29 is used for prediction
    args["learning_rate"] = 1e-3
    args["grad_clip_norm"] = 5.0

    # Set training features and labels
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args["batch_size"],
        shuffle=True
    )

    # Set testing features and labels
    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=no,
        shuffle=True
    )
    
    # Initialize model
    model = GeneralRNN(args)
    model.to(args["device"])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args["learning_rate"]
    )

    # Train the predictive model
    logger = trange(args["epochs"], desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:
        running_loss = 0.0

        for train_x in train_dataloader:
            train_x = train_x.to(args["device"])
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            train_p = model(train_x[:,0:29,:])
            loss = criterion(train_p, train_x[:,1:30,:])
            # backward
            loss.backward()
            # optimize
            optimizer.step()

            running_loss += loss.item()

        logger.set_description(f"Epoch: {epoch}, Loss: {running_loss:.4f}")

    # Evaluate the trained model
    with torch.no_grad():
        perf = np.zeros((8789,2668))
        ind = 0
        for test_x in test_dataloader:
            test_x = test_x.to(args["device"])
            test_p = model(test_x[:,0:29,:]).cpu()

            test_p = np.reshape(test_p.numpy(), [test_p.numpy().shape[0],-1])
            test_y = np.reshape(test_x[:,1:30,:].cpu().numpy(), [test_x[:,1:30,:].cpu().numpy().shape[0],-1])

            perf[ind:ind+test_x[:,1:30,:].cpu().numpy().shape[0]] = test_p-test_y
            ind = ind+test_x[:,1:30,:].cpu().numpy().shape[0]
            #perf += rmse_error(test_y, test_p)
            #perf2.append(rmse_error(test_y, test_p))
        
        mae = np.abs(perf).mean(1)
        rmse = (perf * perf).mean(1)
        rmse = np.sqrt(rmse)
        
        print(f'MAE : {mae.mean(0)}')
        print(f'RMSE : {rmse.mean(0)}')

    return perf, rmse.mean(0), mae.mean(0)
