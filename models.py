import torch

encoder_dim = 1024
predictor_dim = 1024
joiner_dim = 1024
audio_dim = 80
audio_frames_per_video_frame = 10 #4
#mctct

# adapted from https://github.com/lorenlugosch/transducer-tutorial/blob/main/transducer_tutorial_example.ipynb
class Encoder(torch.nn.Module):
		def __init__(self):
				super(Encoder, self).__init__()
				self.combine = torch.nn.Linear(audio_dim*audio_frames_per_video_frame + 128, encoder_dim)
				# self.audio_encoder = 
				self.video_encoder = torch.nn.Conv2d(in_channels=3, out_channels=128, stride=7, kernel_size=7)
				#self.video_dropout = torch.nn.Dropout(p=0.5)
				self.rnn = torch.nn.LSTM(input_size=encoder_dim, hidden_size=encoder_dim, num_layers=3, batch_first=True, bidirectional=False, dropout=0.15)
				self.linear = torch.nn.Linear(encoder_dim, joiner_dim)

		def forward(self, audio, video):
				batch_size = video.shape[0]
				video_T = video.shape[1]
				audio_features = audio #self.audio_encoder(audio)
				audio_T = audio_features.shape[1]
				pad_ = (video_T*audio_frames_per_video_frame) - audio_T
				audio_features = torch.nn.functional.pad(audio_features, (0,0,0,pad_))
				audio_features = audio_features.reshape(batch_size, video_T, audio_dim*audio_frames_per_video_frame)

				video = video.reshape(-1, 3, 630, 300)
				video_features = self.video_encoder(video)
				video_features = video_features.max(dim=2)[0].max(dim=2)[0]
				#video_features = self.video_dropout(video_features)
				video_features = video_features.reshape(batch_size, video_T, 128)
				out = torch.cat([audio_features, video_features], dim=2)
				out = self.combine(out)
				out = self.rnn(out)[0]
				out = self.linear(out)
				return out

class Predictor(torch.nn.Module):
		def __init__(self, output_dim):
				super(Predictor, self).__init__()
				self.embed = torch.nn.Linear(output_dim, predictor_dim)
				self.rnn = torch.nn.LSTMCell(input_size=predictor_dim, hidden_size=predictor_dim)
				self.linear = torch.nn.Linear(predictor_dim, joiner_dim)
				self.initial_state_h = torch.nn.Parameter(torch.randn(predictor_dim))
				self.initial_state_c = torch.nn.Parameter(torch.randn(predictor_dim))
				self.start_token = torch.tensor([-1, -1, -1.])

		def forward_one_step(self, input, previous_state):
				embedding = self.embed(input)
				state = self.rnn.forward(embedding, previous_state)
				out = self.linear(state[0])
				return out, state

		def forward(self, y):
				batch_size = y.shape[0]
				U = y.shape[1]
				outs = []
				state = (torch.stack([self.initial_state_h] * batch_size), #.to(y.device)
								torch.stack([self.initial_state_c] * batch_size))
				for u in range(U): # need U+1 to get null output for final timestep
						decoder_input = y[:,u-1]
						out, state = self.forward_one_step(decoder_input, state)
						outs.append(out)
				out = torch.stack(outs, dim=1)
				return out

class Joiner(torch.nn.Module):
		def __init__(self, num_outputs):
				super(Joiner, self).__init__()
				self.linear = torch.nn.Linear(joiner_dim, num_outputs)

		def forward(self, encoder_out, predictor_out):
				out = encoder_out + predictor_out
				out = torch.nn.functional.relu(out)
				out = self.linear(out)
				return out
