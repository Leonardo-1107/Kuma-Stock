from net import *
from utils import *

class KumaModel:

    def __init__(self, model_name, input_size=6, hidden_size=32, output_size=1, num_layers=3, dropout_rate=0.2, is_attention=False):
        """
        Initialize the model with parameters.
        Args:

            hidden_size: (int, default 128) the size of the linear neuron layer in GRU
            output_seze: (int, default 1) the input size of the GRU. Default set to 1, as the last time step of the sequence.
            num_layers: (int, default 3) the num of muli-GRU layers in DRNN.
            dropout_rate: (float, default 0.2) rate of the drop out layer in the model.
            is_attention: (bool, default False) whether to deploy attention layber between grus.

        """
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        self.model_list = ['gru', 'drnn']

        if model_name.lower() == 'gru':
            self.model = GRUModel(input_size, hidden_size, output_size).to(self.device)
        elif model_name.lower() == 'drnn':
            self.model = DRNNModel(input_size, hidden_size, output_size, num_layers, dropout_rate, is_attention).to(self.device)
        else:
            print(f"Model not exist, please choose between {self.model_list}")
            raise NameError

    def set_train_params(self, loss_type='ic', lr=0.001):
        if loss_type.lower() == 'ic':
            self.criterion = ICLoss().to(self.device)
        elif loss_type.lower() == 'mse':
            self.criterion = nn.MSELoss().to(self.device)
        elif loss_type.lower() == 'l1':
            self.criterion = nn.L1Loss().to(self.device)
        else:
            print("Wrong Loss Function!")
            raise TypeError

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr)

    def train_model(self, X, y, seq_length, epochs=200):
        """
        Train the model and update loss with tqdm.

        Args:
            X: The input data.
            y: The target data.
            epochs: The number of epochs to train the model. Default is 200.
        """
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        
        # add random noise to prevent model collapse
        # X = add_noise_4d(X)

        # List to store loss values and set up the real-time plot
        loss_values = []
        
        # record the train process in plots
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlabel('Epochs')
        ax.set_ylabel(f'Loss Value')
        line, = ax.plot([], [], 'r-')
        
        def update_plot(loss_values):
            line.set_xdata(np.arange(len(loss_values)))
            line.set_ydata(loss_values)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()


        X_tensor = torch.tensor(X.reshape(len(y), seq_length, -1), dtype=torch.float).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float).view(-1).to(self.device)
        
        for _ in range(epochs):
            
            # Get data for training
            inputs = X_tensor  # Add batch dimension
            targets = y_tensor

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record the train process
            loss_values.append(loss.item())
            # update_plot(loss_values)

        # plt.ioff()
        # plt.show()
        # plt.close(fig)
    

    def prediction(self, val_X, date, code_list, seq_length):
        """
        To output the factor value of the given time sequence
        val_x: The input time sequence.
        code_list: The codes of stocks included.
        date: Date for prediction results.
        """
        self.model.eval()
        
        df_factor_value = []
        df_code = []
        df_date = []
        for i in tqdm(range(len(val_X))):
            try:
                val_X[i] = np.nan_to_num(val_X[i], nan=0.0)
                x = torch.tensor(np.array(val_X[i]).reshape(len(date[i]), seq_length, -1), dtype=torch.float).to(self.device)
                pred_y = list(self.model(x).detach().cpu().numpy().squeeze())
                df_factor_value.extend(pred_y)
                df_code.extend(code_list[i])
                df_date.extend(date[i])
            except:
                continue

        result_df = pd.DataFrame([df_date, df_code, df_factor_value])
        result_df.index = ['date', 'code', 'gru']
        result_df = result_df.T
        result_df.sort_values(by=['date', 'code'], inplace=True)

        return result_df