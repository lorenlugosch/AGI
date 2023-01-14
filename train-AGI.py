# python train-AGI.py hparams.yaml --batch_size=2 --output_folder=<whatever>
import torch
import speechbrain as sb
import numpy as np
import sys
from hyperpyyaml import load_hyperpyyaml
import struct
import torchvision

#encoder_dim = 1024
#predictor_dim = 1024
#joiner_dim = 1024
#audio_dim = 80
#audio_frames_per_video_frame = 10 #4
#mctct

def greedy_search(encoder_out, predictor, joiner):
	y_batch = []
	B = encoder_out.shape[0]; T = encoder_out.shape[1]
	for b in range(B):
		t = 0; u = 0; y = [predictor.start_token]; predictor_state = (predictor.initial_state_h.unsqueeze(0), predictor.initial_state_c.unsqueeze(0))
		U_max = T * 2
		while t < T and u < U_max:
			predictor_input = y[-1].unsqueeze(0).to(encoder_out.device)
			g_u, predictor_state = predictor.forward_one_step(predictor_input, predictor_state)
			f_t = encoder_out[b, t]
			h_t_u = joiner.forward(f_t, g_u)
			transition = h_t_u[0,0].item() < 0.
			if transition:
				t += 1
			else: # argmax == a label
				u += 1
				#DOWN = h_t_u[0,1].item() > 0 # should be < 0? something wrong
				DOWN = h_t_u[0,1].item() < 0
				y_u = torch.tensor([
					not DOWN,
					h_t_u[0,2].item() if DOWN else -1.,
					h_t_u[0,3].item() if DOWN else -1.
				])
				if b == 0: print(t); print(DOWN); print(h_t_u);
				y.append(y_u)
		#print(y)
		y_batch.append(y[1:]) # remove start symbol
	return y_batch

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

		#print(batch.unaligned_gestures[0][0])
		#greedy_search(encoder_out, self.modules.predictor, self.modules.joiner)
		#sys.exit()

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

		#print(joiner_out[0,:,1][mask[0,:,1]])
		#print(aligned_gestures[0,:,1].long()[mask[0,:,1]])
		type_emission_accuracy = (mask[:,:,1]*(aligned_gestures[:,:,1].long() == (joiner_out[:,:,1] > 0).long())).sum()/mask[:,:,1].sum()
		#print("type emission accuracy: %f" % type_emission_accuracy)
		transition_accuracy = (mask[:,:,0]*(aligned_gestures[:,:,0].long() == (joiner_out[:,:,0] > 0).long())).sum()/mask[:,:,0].sum()
		#print("transition accuracy: %f" % transition_accuracy)
		return (transition_losses.sum() + type_emission_losses.sum() + location_emission_losses.sum()) / batch_size

	def on_stage_start(self, stage, epoch):
		"""Gets called at the beginning of each epoch"""
		self.batch_count = 0

		# if stage != sb.Stage.TRAIN:
		# 	self.cer_metric = self.hparams.cer_computer()
		# 	self.wer_metric = self.hparams.error_rate_computer()

	def on_stage_end(self, stage, stage_loss, epoch):
		"""Gets called at the end of a epoch."""
		# Compute/store important stats
		stage_stats = {"loss": stage_loss}
		if stage == sb.Stage.TRAIN:
			self.train_stats = stage_stats
		# else:
		# 	stage_stats["CER"] = self.cer_metric.summarize("error_rate")
		# 	stage_stats["WER"] = self.wer_metric.summarize("error_rate")
		# 	stage_stats["SER"] = self.wer_metric.summarize("SER")

		# Perform end-of-iteration things, like annealing, logging, etc.
		if stage == sb.Stage.VALID:
			# old_lr, new_lr = self.hparams.lr_annealing(stage_stats["SER"])
			# sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
			self.hparams.train_logger.log_stats(
				# stats_meta={"epoch": epoch, "lr": old_lr},
				stats_meta={"epoch": epoch},
				train_stats=self.train_stats,
				valid_stats=stage_stats,
			)
			self.checkpointer.save_and_keep_only(
				# meta={"SER": stage_stats["SER"]}, min_keys=["SER"],
				meta={"loss": stage_stats["loss"]}, min_keys=["loss"],
			)
		elif stage == sb.Stage.TEST:
			self.hparams.train_logger.log_stats(
				stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
				test_stats=stage_stats,
			)
			# with open(self.hparams.wer_file, "w") as w:
			# 	self.wer_metric.write_stats(w)

def prepare_data(hparams):
	data_folder = hparams["data_folder"]
	train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
		csv_path=data_folder + "/" + hparams["train_csv"], #replacements={"data_root": data_folder},
	)
	valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
                csv_path=data_folder + "/" + hparams["valid_csv"], #replacements={"data_root": data_folder},
        )
	test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
                csv_path=data_folder + "/" + hparams["test_csv"], #replacements={"data_root": data_folder},
        )
	datasets = [train_data, valid_data, test_data]
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
	return train_data, valid_data, test_data


hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
with open(hparams_file) as fin:
	hparams = load_hyperpyyaml(fin, overrides)
train_data, valid_data, test_data = prepare_data(hparams)

sb.create_experiment_directory(
	experiment_directory=hparams["output_folder"],
	hyperparams_to_save=hparams_file,
	overrides=overrides,
)

#modules = {"encoder": Encoder(), "predictor": Predictor(output_dim=3), "joiner": Joiner(num_outputs=4), "normalize": sb.processing.features.InputNormalization(norm_type="global")}
brain = AGI(
	hparams["modules"],
	lambda x: torch.optim.Adam(x, 3e-4),
	hparams=hparams,
	run_opts=run_opts,
	checkpointer=hparams["checkpointer"],
)
#brain.modules.encoder = torch.load("encoder.pt"); brain.modules.predictor = torch.load("predictor.pt"); brain.modules.joiner = torch.load("joiner.pt"); brain.modules.normalize = torch.load("normalize.pt")
brain.fit(epoch_counter=brain.hparams.epoch_counter, train_set=train_data, valid_set=valid_data, train_loader_kwargs=hparams["train_dataloader_opts"])
#brain.modules.encoder = torch.load("encoder.pt"); brain.modules.predictor = torch.load("predictor.pt"); brain.modules.joiner = torch.load("joiner.pt"); brain.modules.normalize = torch.load("normalize.pt")

