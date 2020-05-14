import librosa
import numpy as np
import h5py
import sys
import os
import itertools

from pydub import AudioSegment
from collections import defaultdict


class Dataset:
    def __init__(self, path:str, reinitialize:bool):
        self.instruments = [
            "bas",
            "gac",
            "key",
            "org"
        ]
        self.features = [
            "chroma_stft",
            "chroma_cqt",
            "chroma_cens",
            "melspectogram",
            "rms",
            "spectral_bandwidth",
            "spectral_centroid",
            "spectral_contrast",
            "spectral_flatness",
            "spectral_rolloff",
            "poly_features",
            "tonnetz",
            "zero_crossing_rate",

            "tempogram",
            "fourier_tempogram",
             
            "track"
        ]
        self.dataset = {}
        self.train_path = "train.h5"
        if reinitialize or not os.path.exists(self.train_path):
            self.SaveTrainData(path)

    def SaveTrainData(self, path:str):
        tracks_count = 0
        for feature in self.features:
            self.dataset[feature] = defaultdict(list)

        for instrument in self.instruments:
            instrument_path = os.path.join(path, instrument)
            tracks = os.listdir(instrument_path)
            count = 0
            for track in tracks:
                try:
                    y, sr = librosa.load(os.path.join(instrument_path, track))

                    self.dataset["chroma_stft"][instrument].append(librosa.feature.chroma_stft(y, sr))
                    self.dataset["chroma_cqt"][instrument].append(librosa.feature.chroma_cqt(y, sr))
                    self.dataset["chroma_cens"][instrument].append(librosa.feature.chroma_cens(y, sr))                 
                    self.dataset["melspectogram"][instrument].append(librosa.feature.melspectrogram(y, sr))
                    self.dataset["rms"][instrument].append(librosa.feature.rms(y))
                    self.dataset["spectral_centroid"][instrument].append(librosa.feature.spectral_centroid(y, sr))
                    self.dataset["spectral_bandwidth"][instrument].append(librosa.feature.spectral_bandwidth(y, sr))
                    self.dataset["spectral_contrast"][instrument].append(librosa.feature.spectral_contrast(y, sr))
                    self.dataset["spectral_flatness"][instrument].append(librosa.feature.spectral_flatness(y))
                    self.dataset["spectral_rolloff"][instrument].append(librosa.feature.spectral_rolloff(y, sr))
                    self.dataset["poly_features"][instrument].append(librosa.feature.poly_features(y, sr))
                    self.dataset["tonnetz"][instrument].append(librosa.feature.tonnetz(y, sr))
                    self.dataset["zero_crossing_rate"][instrument].append(librosa.feature.zero_crossing_rate(y, sr))
                    self.dataset["tempogram"][instrument].append(librosa.feature.tempogram(y, sr))
                    self.dataset["fourier_tempogram"][instrument].append(librosa.feature.tempogram(y, sr))
                    self.dataset["track"][instrument].append(y)

                    count += 1
                    if count == 1000:
                        break
                except (KeyboardInterrupt, SystemExit):
                    raise
                except:
                    print("Missed " + track)
            
            print(instrument + " has " + str(count) + " tracks")
            tracks_count += count
                
        train_file = h5py.File(self.train_path, "w")
        for feature in self.features:
            for instrument in self.instruments:
                print(feature + "-" + instrument)
                train_file.create_dataset(feature + "-" + instrument, data=np.asarray(self.dataset[feature][instrument]))
        
        train_file.attrs["tracks_count"] = tracks_count
        train_file.close()
    
    def __call__(self, features, _2d = False):
        with h5py.File(self.train_path, "r") as dataset:
            X = []
            Y = []
            XY_assigned = False
            
            for feature in features:
                if feature not in self.features:
                    continue
                i = 0
                for instrument in self.instruments:
                    data = dataset[feature + "-" + instrument]
                    for track_features in data:
                        if not XY_assigned:    
                            X.append(track_features.flatten())
                            Y.append(instrument)
                        else:
                            X[i] = np.append(X[i], track_features.flatten())
                            i += 1
                XY_assigned = True

            if _2d:
                for i in range(len(X)):
                    X[i] = np.reshape(X[i], (-1, 130))
                    
            X = np.asarray(X)
            Y = np.asarray(Y)
            assert(X.shape[0] == Y.shape[0])
            return X, Y


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
    dataset = Dataset(path="dataset", reinitialize=False)
    dataset.compare_melspectogram()
    dataset.compare_features_with_one_feature_array()
    print("Nice")