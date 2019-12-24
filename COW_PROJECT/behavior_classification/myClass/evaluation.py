#-*- encoding:utf-8 -*-

""" このクラスは主に評価を行うクラス（今のところfunctions/evaluation_of_classifier.pyとは関係がないが吸収するかたちで拡充していきたい） """
class Evaluation:
    prediction_list = []
    answer_list = []

    def __init__(self, pred, ans):
        self.prediction_list = pred
        self.answer_list = ans

    def evaluate(self, target):
        TP, TN, FP, FN = 0, 0, 0, 0
        for pred, ans in zip(self.prediction_list, self.answer_list):
            if (pred == target and ans == target):
                TP += 1
            elif (pred == target and ans != target):
                FP += 1
            elif (pred != target and ans == target):
                TN += 1
            else:
                FN += 1
        return (TP, TN, FP, FN)