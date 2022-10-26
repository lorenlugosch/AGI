# python train-AGI.py hparams.yaml --batch_size=2 --data_folder=/home/mila/l/lugoschl/timers-and-such-backup
import torch
import speechbrain as sb
import numpy as np
import sys
from hyperpyyaml import load_hyperpyyaml
import struct
import torchvision

conv = torch.nn.Conv2d(in_channels=3, out_channels=128, stride=7, kernel_size=7)

class AGI(sb.Brain):
	def compute_forward(self, batch, stage):
		# print(batch.sig[0].shape)
		vid = batch.vid[0].reshape(-1, 3, 630, 300)
		print(batch.unaligned_gestures[0].shape)
		print(batch.aligned_gestures[0].shape)
		return self.modules.conv(vid)
	def compute_objectives(self, predictions, batch, stage):
		vid = batch.vid[0].reshape(-1, 3, 630, 300)
		return torch.nn.functional.l1_loss(predictions, predictions) #blah)
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
		video_path = data_folder + "/" + screens
		vid = torchvision.io.read_video(video_path, output_format="TCHW")[0].float() / 255.
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
		# return frames 
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
modules = {"conv": conv }
brain = AGI(modules, lambda x : torch.optim.SGD(x, 0.1) )
# data = [{"input": np.random.rand(10 , 10).astype("float32") , "target": np.random.rand(10, 10).astype("float32") }]
brain.fit(epoch_counter=range(15), train_set=train_data, train_loader_kwargs=hparams["train_dataloader_opts"])



# # type a 1 in the calculator
# adb shell sendevent /dev/input/event2 3 57 0; adb shell sendevent /dev/input/event2 3 53 3253; adb shell sendevent /dev/input/event2 3 54 21553; adb shell sendevent /dev/input/event2 3 58 1024; adb shell sendevent /dev/input/event2 0 0 0;   adb shell sendevent /dev/input/event2 3 58 0;  adb shell sendevent /dev/input/event2 3 57 -1; adb shell sendevent /dev/input/event2 0 0 0
# adb shell sendevent /dev/input/event2 3 57 0; adb shell sendevent /dev/input/event2 3 53 3253; adb shell sendevent /dev/input/event2 3 54 21553; adb shell sendevent /dev/input/event2 3 58 1024; adb shell sendevent /dev/input/event2 0 0 0;   adb shell sendevent /dev/input/event2 3 58 0;  adb shell sendevent /dev/input/event2 3 57 -1; adb shell sendevent /dev/input/event2 3 53 3253; adb shell sendevent /dev/input/event2 3 54 21553; adb shell sendevent /dev/input/event2 0 0 0
# adb shell sendevent /dev/input/event2 3 57 0; adb shell sendevent /dev/input/event2 3 53 3253; adb shell sendevent /dev/input/event2 3 54 21553; adb shell sendevent /dev/input/event2 3 58 1024; adb shell sendevent /dev/input/event2 0 0 0;  adb shell sendevent /dev/input/event2 3 57 -1; adb shell sendevent /dev/input/event2 3 53 3253; adb shell sendevent /dev/input/event2 3 54 21553;  adb shell sendevent /dev/input/event2 3 58 0; adb shell sendevent /dev/input/event2 0 0 0

# (1666202942, 553795, 3, 57, 0)
# (1666202942, 553795, 3, 53, 3253)
# (1666202942, 553795, 3, 54, 21553)
# (1666202942, 553795, 3, 58, 1024)
# (1666202942, 553795, 0, 0, 0)
# (1666202943, 76135, 3, 58, 0)
# (1666202943, 76135, 3, 57, -1)
# (1666202943, 76135, 0, 0, 0)

