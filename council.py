import math

from text_analyzer import TextAnalyzer
from constants import *
from utilities import *

from gimli import Gimli
from pippin import Pippin
from SimpleNB_fordo import NB as Frodo

class Council(TextAnalyzer):
    def process(self,nb_max=1000):
        if not self.silent:
            print_color("Council is convening.",COLORS.GREEN)

        classes={"gimli":Gimli,"pippin":Pippin,"frodo":Frodo}
        tas={key:classes[key](self.source_csv,
            interval=self.interval,
            validation_ratio=self.validation_ratio,
            balanced=self.balanced) for key in classes}

        tas["gimli"].process(ngram_max=3,filter_threshold=1)
        tas["pippin"].process(ngram_max=3,k=500)
        tas["frodo"].process(nb_max=nb_max)

        self.predictors=tas

        self.fn_scores={
                "gimli":(71.7,65.5,87.1,58.7),
                "pippin": (97.4,16.8,44.6,8.3),
                "frodo":(60.6,63.1,86.1,40.9)}
        self.tp_scores={
                "gimli":(76.54,70.84,74.75,61.51),
                "pippin":(40.21,76.05,85.81,71.54),
                "frodo":(74.15,53.41,67.04,56.32)}
        self.confidences={
                "gimli":(46.97,49.25,57.2,45.18),
                "pippin":(40.05,35.14,40.12,37.78),
                "frodo":(87.13,83.32,90.15,85.98)}

        self.has_processed=True

    def get_prediction_tuple(self,text):
        tuples={key:self.predictors[key].get_prediction_tuple(text) for key in self.predictors}
        prediction=[0 for i in range(OPTION_COUNT)]

        for key in tuples:
            p=tuples[key]
            p=[item/self.confidences[key][i] for i,item in enumerate(p)]
            p=[item*self.tp_scores[key][i] for i,item in enumerate(p)]
            prediction=[prediction[i]+p[i] for i in range(OPTION_COUNT)]

        return [item/sum(prediction) for item in prediction]

