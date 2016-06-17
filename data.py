from collections import Counter, defaultdict
from itertools import chain

import numpy as np

class WordForm:
    def __init__(self, task, source, target=None,
                 source_feats=None, target_feats=None):

        self.task = task                    # in (1, 2, 3)
        self.source = source                # str
        self.target = target                # str or None
        self.source_feats = source_feats    # str or None
        self.target_feats = target_feats    # str
        self.source_feats_dict = WordForm.parse_features(source_feats)
        self.target_feats_dict = WordForm.parse_features(target_feats)

        self.pos = self.target_feats_dict['pos']
        self.forms = [x for x in (self.source, self.target) if not x is None]
        self.features = [
                x for x in (self.source_feats, self.target_feats)
                if not x is None]
        self.feature_dicts = [
                x for x in (self.source_feats_dict, self.target_feats_dict)
                if not x is None]


    def __str__(self):
        if self.task in (1, 3):
            return '\t'.join(
                    x for x in (self.source, self.target_feats, self.target)
                    if not x is None)
        elif self.task == 2:
            return '\t'.join(
                    x for x in (self.source_feats, self.source,
                                self.target_feats, self.target)
                    if not x is None)


    def recapitalize(self, s):
        if not s: return s
        elif self.source.islower(): return s
        elif self.source.isupper() and len(self.source) > 1: return s.upper()
        else: return s[0].upper() + s[1:]


    @staticmethod
    def parse_features(feats):
        if feats is None: return None
        return dict(pair.split('=') for pair in feats.split(','))

    @staticmethod
    def task1(line):
        row = line.strip().split('\t')
        return WordForm(1,
                        source=row[0],
                        target_feats=row[1],
                        target=row[2] if len(row) == 3 else None)

    @staticmethod
    def task2(line):
        row = line.strip().split('\t')
        return WordForm(2,
                        source_feats=row[0],
                        source=row[1],
                        target_feats=row[2],
                        target=row[3] if len(row) == 4 else None)

    @staticmethod
    def task3(line):
        row = line.strip().split('\t')
        return WordForm(3,
                        source=row[0],
                        target_feats=row[1],
                        target=row[2] if len(row) == 3 else None)


class MorphonData:
    def __init__(self, prefix):
        def read_task(task, part):
            task_funs = [WordForm.task1, WordForm.task2, WordForm.task3]
            filename = '%s-task%d-%s' % (prefix, task, part)
            with open(filename, 'r', encoding='utf-8') as f:
                data = [task_funs[task-1](line) for line in f]
            return data

        self.data = [{part: read_task(task, part)
                      for part in ('train', 'dev', 'test-covered')}
                     for task in (1,2,3)]

        self.alphabet = sorted(set(chain.from_iterable(
            ''.join(wf.forms).lower() for task in self.data
                                      for part in task.values()
                                      for wf in part)))
        
        self.features = sorted(set(chain.from_iterable(
            wf.features for task in self.data
                        for part in task.values()
                        for wf in part)))

        self.feature_idx = {s: i for i,s in enumerate(self.features)}

        self.max_length = max(
            len(form) for task in self.data
                      for part in task.values()
                      for wf in part
                      for form in wf.forms)

        # Note: the Morphon model will add padding and beginning/end of
        # sentence symbols, so don't use this map
        #self.alphabet_idx = {c: i for i,c in enumerate(self.alphabet)}

        self.feature_values = defaultdict(set)
        for task in self.data:
            for part in task.values():
                for wf in part:
                    for feats in wf.feature_dicts:
                        for feat,value in feats.items():
                            self.feature_values[feat].add(value)

        self.feature_values = sorted([
                [feature, sorted(values)]
                for feature, values in self.feature_values.items()])

        self.feature_values_idx = {
                feature: {value: i for i,value in enumerate(values)}
                for feature, values in self.feature_values}

        self.feature_idx = {
                feature: i for i,(feature,_) in enumerate(self.feature_values)}

        self.feature_vector_length = sum(
                len(values) for _,values in self.feature_values)

    def encode_features(self, features):
        return np.array(
                [features.get(feature) == value
                 for feature,values in self.feature_values
                 for value in values],
                dtype=np.int8)


if __name__ == '__main__':
    import sys
    from pprint import pprint

    morphon_data = MorphonData(sys.argv[1])
    print('Alphabet: %s (%d items)' % (
            ''.join(morphon_data.alphabet),
            len(morphon_data.alphabet)))
    print('Number of features: %d' % len(morphon_data.features))
    pprint(morphon_data.features)
    #print(morphon_data.feature_values)

