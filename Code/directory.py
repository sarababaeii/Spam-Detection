from enum import Enum
import os


class Directory(Enum):
    Current = os.getcwd()
    StopWords = Current + '/StopWords.txt'
    DataSet = Current + '/emails'
    SpamTraining = DataSet + '/spamtraining'
    SpamTesting = DataSet + '/spamtesting'
    HamTraining = DataSet + '/hamtraining'
    HamTesting = DataSet + '/hamtesting'

    def __str__(self):
        return str(self.value)