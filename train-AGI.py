# python train-AGI.py hparams.yaml --batch_size=2 --data_folder=/home/mila/l/lugoschl/timers-and-such-backup
import torch
import speechbrain as sb
import numpy as np
import sys
from hyperpyyaml import load_hyperpyyaml
import struct
import torchvision

encoder_dim = 1024
predictor_dim = 1024
joiner_dim = 1024
audio_dim = 80
audio_frames_per_video_frame = 10 #4
#mctct

def greedy_search(encoder_out, predictor, joiner):
	y_batch = []
	B = encoder_out.shape[0]; T = encoder_out.shape[1]
	for b in range(B):
		t = 0; u = 0; y = [predictor.start_token]; predictor_state = (predictor.initial_state_h.unsqueeze(0), predictor.initial_state_c.unsqueeze(0))
		U_max = T * 2
		while t < T and u < U_max:
			print(y[-1])
			predictor_input = y[-1].unsqueeze(0).to(encoder_out.device)
			g_u, predictor_state = predictor.forward_one_step(predictor_input, predictor_state)
			f_t = encoder_out[b, t]
			h_t_u = joiner.forward(f_t, g_u)
			transition = h_t_u[0,0].item() < 0.
			if transition:
				t += 1
			else: # argmax == a label
				u += 1
				DOWN = h_t_u[0,1].item() < 0
				y_u = torch.tensor([
					not DOWN,
					h_t_u[0,2].item() if DOWN else -1.,
					h_t_u[0,3].item() if DOWN else -1.
				])
				print(h_t_u); print(y_u); sys.exit()
				y.append(y_u)
		y_batch.append(y[1:]) # remove start symbol
	return y_batch

# adapted from https://github.com/lorenlugosch/transducer-tutorial/blob/main/transducer_tutorial_example.ipynb
class Encoder(torch.nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.combine = torch.nn.Linear(audio_dim*audio_frames_per_video_frame + 128, encoder_dim)
		# self.audio_encoder = 
		self.video_encoder = torch.nn.Conv2d(in_channels=3, out_channels=128, stride=7, kernel_size=7)
		self.rnn = torch.nn.LSTM(input_size=encoder_dim, hidden_size=encoder_dim, num_layers=3, batch_first=True, bidirectional=False, dropout=0.15)
		self.linear = torch.nn.Linear(encoder_dim, joiner_dim)

	def forward(self, audio,video):
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

class AGI(sb.Brain):
	def compute_aligned_features(self, encoder_out, predictor_out, aligned_gestures):
		batch_size = encoder_out.shape[0]
		T = encoder_out.shape[1]
		aligned_encoder_out = []
		aligned_predictor_out = []
		for b in range(batch_size):
			t = 0; u = 0
			t_u_indices = []
			for step in aligned_gestures[b, :, 0]:
				t_u_indices.append((t,u))
				if int(step.item()) == 0: # right (null)
					t += 1
				if int(step.item()) == 1: # down (label)
					u += 1
			# t_u_indices.append((T-1,U))
			t_indices = [min(t,T-1) for (t,u) in t_u_indices]
			u_indices = [u for (t,u) in t_u_indices]
			encoder_out_expanded = encoder_out[b, t_indices]
			predictor_out_expanded = predictor_out[b, u_indices]
			aligned_encoder_out.append(encoder_out_expanded)
			aligned_predictor_out.append(predictor_out_expanded)
		aligned_encoder_out = torch.stack(aligned_encoder_out)
		aligned_predictor_out = torch.stack(aligned_predictor_out)
		return aligned_encoder_out, aligned_predictor_out

	def compute_forward(self, batch, stage):
		batch = batch.to(self.device)
		video = batch.vid[0].float() / 255.
		audio,audio_lens = batch.sig
		audio = self.hparams.compute_features(audio)
		audio = self.modules.normalize(audio,audio_lens)
		encoder_out = self.modules.encoder(audio,video)

		greedy_search(encoder_out, self.modules.predictor, self.modules.joiner)

		unaligned_gestures = batch.unaligned_gestures[0]
		batch_size = unaligned_gestures.shape[0]
		start_pad = torch.stack([self.modules.predictor.start_token] * batch_size).unsqueeze(1).to(self.device)
		unaligned_gestures_in = torch.cat([start_pad, unaligned_gestures], dim=1)
		predictor_out = self.modules.predictor(unaligned_gestures_in)

		aligned_gestures = batch.aligned_gestures[0]
		aligned_encoder_out, aligned_predictor_out = self.compute_aligned_features(encoder_out, predictor_out, aligned_gestures)
		joiner_out = self.modules.joiner(aligned_encoder_out, aligned_predictor_out)
		return joiner_out

	def compute_objectives(self, predictions, batch, stage):
		joiner_out = predictions
		aligned_gestures = batch.aligned_gestures[0]
		batch_size = joiner_out.shape[0]

		# fix padding - should be -1 instead of 0
		zeros = aligned_gestures[:,:,2:] == 0. # x,y will never be 0
		aligned_gestures[:,:,2:][zeros] = -1
		aligned_gestures[:,:,1][zeros.prod(2).bool()] = -1
		aligned_gestures[:,:,0][zeros.prod(2).bool()] = -1

		mask = (aligned_gestures != -1)
		transition_losses = torch.nn.functional.binary_cross_entropy_with_logits(input=joiner_out[:,:,0], target=aligned_gestures[:,:,0], reduction="none") * mask[:,:,0]
		type_emission_losses = torch.nn.functional.binary_cross_entropy_with_logits(input=joiner_out[:,:,1], target=aligned_gestures[:,:,1], reduction="none") * mask[:,:,1]
		location_emission_losses = torch.nn.functional.mse_loss(input=joiner_out[:,:,2:], target=aligned_gestures[:,:,2:], reduction="none") * mask[:,:,2:]

		transition_accuracy = (mask[:,:,0]*(aligned_gestures[:,:,0].long() == (joiner_out[:,:,0] > 0).long())).sum()/mask[:,:,0].sum()
		print("transition accuracy: %f" % transition_accuracy)
		return (transition_losses.sum() + type_emission_losses.sum() + location_emission_losses.sum()) / batch_size

	def on_stage_end(self, stage, stage_loss, epoch):
		print("yay!")

def prepare_data(hparams):
	data_folder = hparams["data_folder"]
	train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
		csv_path=hparams["train_csv"], #replacements={"data_root": data_folder},
	)
	datasets = [train_data]
	# audio
	@sb.utils.data_pipeline.takes("wav")
	@sb.utils.data_pipeline.provides("sig")
	def audio_pipeline(wav):
		sig = sb.dataio.dataio.read_audio(data_folder + "/" + wav)
		return sig
	sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

	# screen
	@sb.utils.data_pipeline.takes("screens")
	@sb.utils.data_pipeline.provides("vid")
	def screens_pipeline(screens):
		vid = torchvision.io.read_video(data_folder + "/" + screens, output_format="TCHW")[0]
		return vid
	sb.dataio.dataset.add_dynamic_item(datasets, screens_pipeline)

	# gestures
	@sb.utils.data_pipeline.takes("touch_events")
	@sb.utils.data_pipeline.provides("aligned_gestures", "unaligned_gestures")
	def gestures_pipeline(touch_events):
		gesture_path = data_folder + "/" + touch_events
		FORMAT = '2IHHi'
		EVENT_SIZE = struct.calcsize(FORMAT)
		events = []
		with open(gesture_path, "rb" ) as f:
			while 1:
				data = f.read(EVENT_SIZE)
				if not data:
					break
				events.append(struct.unpack(FORMAT,data))
		gestures = []
		event_idx = 0
		g = {"timestamp": -1, "code": -1, "x": -1, "y": -1}
		while event_idx < len(events):
			# g = {"timestamp": -1, "code": -1, "x": -1, "y": -1}
			done = False
			# loop through events until 0 0 0 encountered
			while not done:
				e = events[event_idx]
				if e[2] == 0:
					done = True
				else:
					if e[3] == 57:
						g["code"] = e[4]
						if g["code"] == -1: # 0 = down, 1 = up, 2 = no op
							g["code"] = 1 # reserve -1 for "no label"
					if e[3] == 53:
						g["x"] = e[4] / 32768. # == 2**15 because signed 2-byte int
					if e[3] == 54:
						g["y"] = e[4] / 32768.
				event_idx += 1
			if done:
				timestamp = e[0] + e[1]/1000000
				g["timestamp"] = timestamp
				g_ = g.copy()
				if g_["code"] == 1:
					g_["x"] = -1
					g_["y"] = -1
				gestures.append(g_)
		#
		timestamp_path = gesture_path.replace(".gestures", "_timestamps.npy")
		timestamps = np.load(timestamp_path)
		frames = []
		t_prev = -1
		for t in timestamps:
			frame = []
			for g in gestures:
				t_g = g["timestamp"]
				if t_prev <= t_g and t_g < t:
					frame.append(g)
			t_prev = t
			frames.append(frame)
		# print(frames)
		#
		aligned_gestures = []
		unaligned_gestures = []
		NULL_ = [0, -1, -1, -1]
		# label: [1, 0/1, x, y]
		for frame in frames:
			for gesture in frame:
				label = [gesture["code"], gesture["x"], gesture["y"]]
				unaligned_gestures.append(label)
				aligned_gestures.append([1] + label)
			aligned_gestures.append(NULL_) # must transition to next timestep for each frame
		unaligned_gestures = torch.tensor(unaligned_gestures)
		aligned_gestures = torch.tensor(aligned_gestures)
		yield aligned_gestures
		yield unaligned_gestures
	sb.dataio.dataset.add_dynamic_item(datasets, gestures_pipeline)

	sb.dataio.dataset.set_output_keys(
		datasets,
		["id", "sig", "vid", "aligned_gestures", "unaligned_gestures"]
	)
	return train_data


hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
with open(hparams_file) as fin:
	hparams = load_hyperpyyaml(fin, overrides)
train_data = prepare_data(hparams)

# modules = {"model": torch.nn.Linear(in_features=10 , out_features=10) }
modules = {"encoder": Encoder(), "predictor": Predictor(output_dim=3), "joiner": Joiner(num_outputs=4), "normalize": sb.processing.features.InputNormalization(norm_type="global")}
brain = AGI(modules, lambda x: torch.optim.Adam(x, 3e-4), hparams=hparams, run_opts=run_opts)
brain.modules.encoder = torch.load("encoder.pt"); brain.modules.predictor = torch.load("predictor.pt"); brain.modules.joiner = torch.load("joiner.pt"); brain.modules.normalize = torch.load("normalize.pt")
brain.fit(epoch_counter=range(15), train_set=train_data, train_loader_kwargs=hparams["train_dataloader_opts"])
#brain.modules.encoder = torch.load("encoder.pt"); brain.modules.predictor = torch.load("predictor.pt"); brain.modules.joiner = torch.load("joiner.pt"); brain.modules.normalize = torch.load("normalize.pt")

