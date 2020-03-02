from __future__ import print_function
import librosa
from pydub import AudioSegment
import numpy as np
import h5py
import sys
import os
from collections import defaultdict

class Dataset:
    def __init__(self, path:str, reinitialize:bool):
        self.instruments = [
            "pia",
            "gac",
            "sax"
        ]
        self.features = [
            "melspectogram",
            "rms",
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_rolloff",
            "zero_crossing_rate"
        ]
        self.dataset = {}
        self.train_path = "train.h5"
        if reinitialize or not os.path.exists(self.train_path):
            self.SaveTrainData(path)

    def SaveTrainData(self, path:str):
        for feature in self.features:
            self.dataset[feature] = defaultdict(list)

        for instrument in self.instruments:
            instrument_path = os.path.join(path, instrument)
            tracks = os.listdir(instrument_path)
            count = 0
            for track in tracks:
                count += 1
                try:
                    y, sr = librosa.load(os.path.join(instrument_path, track))
                    self.dataset["melspectogram"][instrument].append(librosa.amplitude_to_db(librosa.feature.melspectrogram(y, sr)))
                    self.dataset["rms"][instrument].append(librosa.feature.rms(y))
                    self.dataset["spectral_centroid"][instrument].append(librosa.feature.spectral_centroid(y, sr))
                    self.dataset["spectral_bandwidth"][instrument].append(librosa.feature.spectral_bandwidth(y, sr))
                    self.dataset["spectral_rolloff"][instrument].append(librosa.feature.spectral_rolloff(y, sr))
                    self.dataset["zero_crossing_rate"][instrument].append(librosa.feature.zero_crossing_rate(y, sr))
                except (KeyboardInterrupt, SystemExit):
                    raise
                except:
                    print("Missed " + track)
            
            print(instrument + " has " + str(count) + " tracks")
        
        train_file = h5py.File(self.train_path, "w")
        for feature in self.features:
            for instrument in self.instruments:
                print(feature + "-" + instrument)
                train_file.create_dataset(feature + "-" + instrument, data=np.asarray(self.dataset[feature][instrument]))
        
        train_file.attrs["vector_size"] = (128, 130)
        train_file.close()
    
    def __call__(self):
        print("call")

    def compare_melspectogram(self):
        with h5py.File(self.train_path, "r") as dataset:
            for instrument in self.instruments:
                data = dataset["melspectogram-" + instrument]
                average = np.average(data, axis=(0))
                print(average.shape)
                for instrument2 in self.instruments:
                    data2 = dataset["melspectogram-" + instrument2]
                    average2 = np.average(data2, axis=0)
                    with open("feature_comparison/melspectogram/" + instrument + "-" + instrument2, "w") as f:
                        difference = average - average2
                        for i in range(0, difference.shape[0]):
                            for j in range(0, difference.shape[1]):
                                f.write(str(difference[i][j]) + " ")
                            f.write("\n");
    
    def compare_features_with_one_feature_array(self):
        features_with_one_feature_array = [
            "rms",
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_rolloff",
            "zero_crossing_rate"
        ]
        import matplotlib.pyplot as plt
        with h5py.File(self.train_path, "r") as dataset:
            for feature in features_with_one_feature_array:
                for instrument in self.instruments:
                    data = dataset[feature + "-" + instrument]
                    average = np.average(data, axis=0)
                    print(average.shape)
                    for instrument2 in self.instruments:
                        data2 = dataset[feature + "-" + instrument2]
                        average2 = np.average(data2, axis=0)
                        with open("feature_comparison/" + feature + "/" + instrument + "-" + instrument2, "w") as f:
                            difference = average - average2
                            for i in range(0, difference.shape[0]):
                                for j in range(0, difference.shape[1]):
                                    f.write(str(difference[i][j]) + " ")
                                f.write("\n");
                            
                            x_axis = [i for i in range (average.shape[1])]
                            plt.plot(x_axis, average[0], alpha=0.25, label=instrument, color='green')
                            plt.plot(x_axis, average2[0], alpha=0.25, label=instrument2, color='blue')
                            plt.title(feature)
                            plt.legend()
                            plt.savefig("feature_comparison/" + feature + "/" + instrument + "-" + instrument2 + ".png")
                            plt.close()



if __name__ == '__main__':
    dataset = Dataset(path="dataset", reinitialize=True)
    dataset.compare_melspectogram()
    dataset.compare_features_with_one_feature_array()
    print("Nice")
                

