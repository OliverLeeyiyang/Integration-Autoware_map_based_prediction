# for neuralNet
import torch
from torchinfo import summary
import torch.nn as nn
from ament_index_python import get_package_share_directory


####################
wholeModelName = 'wholeNetPrototype5'
####################

class TrajectoryPredictionNet(nn.Module):
    def __init__(self):
        super(TrajectoryPredictionNet, self).__init__()
        self.predictor1 = nn.Linear(145, 512)  # Future dx and dy predictor
        self.predictor2 = nn.Linear(512, 256)
        self.predictor3 = nn.Linear(256, 128)
        self.predictor4 = nn.Linear(128, 60)

    def forward(self, input_data):
        predicted_trajectory = self.predictor1(input_data)
        predicted_trajectory = nn.ReLU()(predicted_trajectory)
        predicted_trajectory = self.predictor2(predicted_trajectory)
        predicted_trajectory = nn.ReLU()(predicted_trajectory)
        predicted_trajectory = self.predictor3(predicted_trajectory)
        predicted_trajectory = nn.ReLU()(predicted_trajectory)
        predicted_trajectory = self.predictor4(predicted_trajectory)
        return predicted_trajectory.view(-1, 30 * 2)


##########################
modelName = 'CNNModelFlatten64x1x1'
##########################
encoded_space_dim = 64
outChannels = 4

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, outChannels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(outChannels, outChannels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(outChannels, 2*outChannels, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(2*outChannels, 4*outChannels, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(4*outChannels, 8*outChannels, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(8*outChannels, 16*outChannels, 3, padding=1, stride = 2),
            nn.ReLU()
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(64*3*3, 128),
            nn.ReLU(),
            nn.Linear(128, encoded_space_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.view(-1,1,39,39)
        output = self.encoder_cnn(x)
        # print(output.shape)
        output = self.flatten(output)
        output = self.encoder_lin(output)
        return output


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim,128),
            nn.ReLU(),
            nn.Linear(128, 64*3*3)
        )

        ### Unflatten layer
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64,3,3))

        ### convolutional Section
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(16*outChannels, 8*outChannels, 3, padding=1, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(8*outChannels, 4*outChannels, 3, padding=1, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(4*outChannels, 2*outChannels, 3, padding=1, stride = 2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2*outChannels, outChannels, 3, padding=1, stride = 2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(outChannels, outChannels, 4, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(outChannels, 1, 5, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.view(-1,encoded_space_dim)
        output = self.decoder_lin(x)
        output = self.unflatten(output)
        output = self.decoder_cnn(output)
        return output


class ConvAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(ConvAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x):
        x = x.view(-1,1,39,39)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Model:
    def __init__(self):
        encoder = Encoder()
        decoder = Decoder()
        self.autoencoder = ConvAutoencoder(encoder, decoder)
        self.model = TrajectoryPredictionNet()
    def load(self):
        """
        Load the pre-trained neural network models for trajectory prediction and autoencoder.

        Args:
        - None

        Returns:
        - autoencoder (ConvAutoencoder): The pre-trained autoencoder model.
        - model (TrajectoryPredictionNet): The pre-trained trajectory prediction model.
        """
        abpath = get_package_share_directory('tum_prediction')
        device = 'cpu'
        file_path = abpath + '/encoder_model_weights/CNN_200.pth'
        self.autoencoder = torch.load(file_path, map_location=torch.device('cpu'))
        self.autoencoder.eval()

        file_path = abpath + '/prediction_model_weights/cleanedCombine.csv_wholeNetPrototype5_Epochs_25_ValidationLoss0.442764_lr0.0003_2023_08_16_19:33:00.pth' # l1 + l2 loss (This is the MLP model used in the thesis)
        # file_path = 'linearPredictionModels/db39/mega/cleanedCombine.csv_wholeNetPrototype5_Epochs_25_ValidationLoss0.376590_lr0.0003_2023_08_17_06:15:32.pth' #only l1 loss
        # file_path = 'linearPredictionModels/db39/mega/cleanedCombine.csv_wholeNetPrototype5_Epochs_25_ValidationLoss0.368600_lr0.0003_2023_08_17_19:42:59.pth' # l1 + l2 loss ablation without social tensor

        self.model = torch.load(file_path, map_location=torch.device('cpu'))
        self.model.eval()
        summary(self.model, (1, 145))

        self.autoencoder.to(device)
        self.model.to(device)

        return self.autoencoder, self.model
