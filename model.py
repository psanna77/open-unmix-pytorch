from torch.nn import LSTM, Linear, BatchNorm1d, Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoOp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class STFT(nn.Module):
    def __init__(
            self,
            n_fft=4096,
            n_hop=1024,
            center=False
    ):
        super(STFT, self).__init__()
        self.window = nn.Parameter(
            torch.hann_window(n_fft),
            requires_grad=False
        )
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

    def forward(self, x):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_bins, nb_frames, 2)
        """

        nb_samples, nb_channels, nb_timesteps = x.size()

        # merge nb_samples and nb_channels for multichannel stft
        x = x.reshape(nb_samples * nb_channels, -1)

        # compute stft with parameters as close as possible scipy settings
        stft_f = torch.stft(
            x,
            n_fft=self.n_fft, hop_length=self.n_hop,
            window=self.window, center=self.center,
            normalized=False, onesided=True,
            pad_mode='reflect'
        )

        # reshape back to channel dimension
        stft_f = stft_f.contiguous().view(
            nb_samples, nb_channels, self.n_fft // 2 + 1, -1, 2
        )
        return stft_f


class Spectrogram(nn.Module):
    def __init__(
            self,
            power=1,
            mono=True
    ):
        super(Spectrogram, self).__init__()
        self.power = power
        self.mono = mono

    def forward(self, stft_f):
        """
        Input: complex STFT
            (nb_samples, nb_bins, nb_frames, 2)
        Output: Power/Mag Spectrogram
            (nb_frames, nb_samples, nb_channels, nb_bins)
        """
        stft_f = stft_f.transpose(2, 3)
        # take the magnitude
        stft_f = stft_f.pow(2).sum(-1).pow(self.power / 2.0)

        # downmix in the mag domain
        if self.mono:
            stft_f = torch.mean(stft_f, 1, keepdim=True)

        # permute output for LSTM convenience
        return stft_f.permute(2, 0, 1, 3)


class OpenUnmix(nn.Module):
    def __init__(
            self,
            n_fft=4096,
            n_hop=1024,
            input_is_spectrogram=False,
            hidden_size=512,
            nb_channels=2,
            sample_rate=44100,
            nb_layers=3,
            input_mean=None,
            input_scale=None,
            max_bin=None,
            unidirectional=False,
            power=1,
    ):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
            or (nb_frames, nb_samples, nb_channels, nb_bins)
        Output: Power/Mag Spectrogram
                (nb_frames, nb_samples, nb_channels, nb_bins)
        """

        super(OpenUnmix, self).__init__()

        self.nb_output_bins = n_fft // 2 + 1
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        self.hidden_size = hidden_size

        self.stft = STFT(n_fft=n_fft, n_hop=n_hop)
        self.spec = Spectrogram(power=power, mono=(nb_channels == 1))
        self.register_buffer('sample_rate', torch.tensor(sample_rate))

        if input_is_spectrogram:
            self.transform = NoOp()
        else:
            self.transform = nn.Sequential(self.stft, self.spec)

        self.fc1 = Linear(
            self.nb_bins * nb_channels, hidden_size,
            bias=False
        )

        self.bn1 = BatchNorm1d(hidden_size)

        if unidirectional:
            lstm_hidden_size = hidden_size
        else:
            lstm_hidden_size = hidden_size // 2

        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4,
        )

        self.fc2 = Linear(
            in_features=hidden_size * 5,
            out_features=hidden_size,
            bias=False
        )

        self.bn2 = BatchNorm1d(hidden_size)

        self.fc3 = Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins * nb_channels,
            bias=False
        )

        self.bn3 = BatchNorm1d(self.nb_output_bins * nb_channels)

        if input_mean is not None:
            input_mean = torch.from_numpy(
                -input_mean[:self.nb_bins]
            ).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(
                1.0 / input_scale[:self.nb_bins]
            ).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

        self.output_scale = Parameter(
            torch.ones(self.nb_output_bins).float()
        )
        self.output_mean = Parameter(
            torch.ones(self.nb_output_bins).float()
        )

    def half_forward(self, x):
        # check for waveform or spectrogram
        # transform to spectrogram if (nb_samples, nb_channels, nb_timesteps)
        # and reduce feature dimensions, therefore we reshape
        x = self.transform(x)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        mix = x.detach().clone()

        # crop
        x = x[..., :self.nb_bins]

        # shift and scale input to mean=0 std=1 (across all bins)
        x += self.input_mean
        x *= self.input_scale

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))
        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        # squash range ot [-1, 1]
        x = torch.tanh(x)

        return x

    def forward(self, x, x1, x2, x3, og):
        # check for waveform or spectrogram
        # transform to spectrogram if (nb_samples, nb_channels, nb_timesteps)
        # and reduce feature dimensions, therefore we reshape
        og = self.transform(og)
        nb_frames, nb_samples, nb_channels, nb_bins = og.data.shape

        mix = og.detach().clone()
        #
        # # crop
        # x = x[..., :self.nb_bins]
        #
        # # shift and scale input to mean=0 std=1 (across all bins)
        # x += self.input_mean
        # x *= self.input_scale
        #
        # # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # # and encode to (nb_frames*nb_samples, hidden_size)
        # x = self.fc1(x.reshape(-1, nb_channels*self.nb_bins))
        # # normalize every instance in a batch
        # x = self.bn1(x)
        # x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        # # squash range ot [-1, 1]
        # x = torch.tanh(x)

        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x)

        print(x.shape)
        print(lstm_out[0].shape)
        print(x1.shape)
        print(x2.shape)
        print(x3.shape)
        print("********")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        max_shape = max(max(max(x.shape[0], x1.shape[0]), x2.shape[0]), x3.shape[0])
        og_shape = x.shape[0]

        x_pad, lstm_out_pad, x1_pad, x2_pad, x3_pad = x.clone(), lstm_out[0].clone(), x1.clone(), x2.clone(), x3.clone()
        if x.shape[0] < max_shape:
            x_pad = torch.zeros(max_shape - x.shape[0], x.shape[1], x.shape[2]).to(device)
            x_pad = torch.cat([x, x_pad], dim=0)
        if lstm_out[0].shape[0] < max_shape:
            lstm_out_pad = torch.zeros(max_shape - x.shape[0], x.shape[1], x.shape[2]).to(device)
            lstm_out_pad = torch.cat([lstm_out[0], lstm_out_pad], dim=0)
        if x1.shape[0] < max_shape:
            x1_pad = torch.zeros(max_shape - x1.shape[0], x.shape[1], x.shape[2]).to(device)
            x1_pad = torch.cat([x1, x1_pad], dim=0)
        if x2.shape[0] < max_shape:
            x2_pad = torch.zeros(max_shape - x2.shape[0], x.shape[1], x.shape[2]).to(device)
            x2_pad = torch.cat([x2, x2_pad], dim=0)
        if x3.shape[0] < max_shape:
            x3_pad = torch.zeros(max_shape - x3.shape[0], x.shape[1], x.shape[2]).to(device)
            x3_pad = torch.cat([x3, x3_pad], dim=0)

        print(x_pad.shape)
        print(lstm_out_pad.shape)
        print(x1_pad.shape)
        print(x2_pad.shape)
        print(x3_pad.shape)
        print("HHHHH")

        # lstm skip connection
        x = torch.cat([x_pad, lstm_out_pad, x1_pad, x2_pad, x3_pad], -1)

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)

        x = F.relu(x)

        # second dense stage + layer norm
        x = self.fc3(x)
        x = self.bn3(x)

        # reshape back to original dim
        x = x[:og_shape, :, :]
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix

        return x


class Separator(nn.Module):
    def __init__(self,
                 n_fft=4096,
                 n_hop=1024,
                 input_is_spectrogram=False,
                 hidden_size=512,
                 nb_channels=2,
                 sample_rate=44100,
                 nb_layers=3,
                 input_mean=None,
                 input_scale=None,
                 max_bin=None,
                 unidirectional=False,
                 power=1,
                 device='cpu'
                 ):
        super(Separator, self).__init__()
        # Vocals
        self.unmix_v = OpenUnmix(
            input_mean=input_mean,
            input_scale=input_scale,
            nb_channels=nb_channels,
            hidden_size=hidden_size,
            n_fft=n_fft,
            n_hop=n_hop,
            max_bin=max_bin,
            sample_rate=sample_rate
        ).to(device)

        # Drums
        self.unmix_d = OpenUnmix(
            input_mean=input_mean,
            input_scale=input_scale,
            nb_channels=nb_channels,
            hidden_size=hidden_size,
            n_fft=n_fft,
            n_hop=n_hop,
            max_bin=max_bin,
            sample_rate=sample_rate
        ).to(device)

        # Bass
        self.unmix_b = OpenUnmix(
            input_mean=input_mean,
            input_scale=input_scale,
            nb_channels=nb_channels,
            hidden_size=hidden_size,
            n_fft=n_fft,
            n_hop=n_hop,
            max_bin=max_bin,
            sample_rate=sample_rate
        ).to(device)

        # Others
        self.unmix_o = OpenUnmix(
            input_mean=input_mean,
            input_scale=input_scale,
            nb_channels=nb_channels,
            hidden_size=hidden_size,
            n_fft=n_fft,
            n_hop=n_hop,
            max_bin=max_bin,
            sample_rate=sample_rate
        ).to(device)

    def forward(self, x):
        Y_hat_vocals = self.unmix_v.half_forward(x)
        Y_hat_drums = self.unmix_d.half_forward(x)
        Y_hat_bass = self.unmix_b.half_forward(x)
        Y_hat_other = self.unmix_o.half_forward(x)

        Y_hat_vocals_f = self.unmix_v(Y_hat_vocals.clone(), Y_hat_drums, Y_hat_bass, Y_hat_other, x)
        Y_hat_drums_f = self.unmix_d(Y_hat_drums.clone(), Y_hat_vocals, Y_hat_bass, Y_hat_other, x)
        Y_hat_bass_f = self.unmix_b(Y_hat_bass.clone(), Y_hat_vocals, Y_hat_drums, Y_hat_other, x)
        Y_hat_other_f = self.unmix_o(Y_hat_other.clone(), Y_hat_vocals, Y_hat_drums, Y_hat_bass, x)

        return Y_hat_vocals_f, Y_hat_drums_f, Y_hat_bass_f, Y_hat_other_f
