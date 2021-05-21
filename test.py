import re, io, gzip, urllib.request, os.path
from collections import namedtuple
from typing import TextIO, AnyStr

from featselect import *


# Simple Bayes Net implementation for the purposes of testing:

class Node(Variable):
    """A node in a Bayes network."""

    def __init__(self, variable: Variable):
        super().__init__(variable.id, variable.values)
        # These values are possibly unknown during the creation of this object:
        self.parents: Optional[Iterable['Node']] = None
        self.children: Optional[Iterable['Node']] = None
        self.cpt: Optional[np.ndarray] = None  # A multidimensional Conditional Probability Table.

    def _distribution(self, parent_values: Event) -> np.ndarray:
        return self.cpt[event_index(parent_values, self.parents)]

    def probability(self, evidence: Event, value: Optional[Value] = None) -> Probability:
        if value is None:
            value = evidence[self]
        return self._distribution(evidence)[self.values.index(value)]

    def sample(self, evidence: Event, k=1) -> list[Value]:
        return random.choices(population=self.values, weights=self._distribution(evidence), k=k)


BayesNet = Iterable[Node]


def markov_blanket(targets: Iterable[Node]) -> set[Node]:
    """Find a Markov blanket set."""
    result = set()
    for node in targets:
        for parent in node.parents:
            result.add(parent)
        for child in node.children:
            result.add(child)
            for spouse in child.parents:
                result.add(spouse)
    for node in targets:
        if node in result:
            result.remove(node)
    return result


def topological_order(bn: BayesNet) -> BayesNet:
    found = {node: False for node in bn}
    marked = {node: False for node in bn}

    def dfs_visit(node):
        if found[node]:
            return
        if marked[node]:
            raise ValueError("Graph must be DAG")
        marked[node] = True
        for parent in node.parents:
            for _result in dfs_visit(parent):
                yield _result
        marked[node] = False
        found[node] = True
        yield node

    start = None
    while not all(found.values()):
        for node, _found in found.items():
            if not _found:
                start = node
                break
        for result in dfs_visit(start):
            yield result


def prior_sampling(bn: BayesNet, sort=True, evidence=EmptyEvent) -> Iterable[Event]:
    """Return observations that are randomly distributed in accordance with the Bayes net."""
    ordered = bn if not sort else list(topological_order(bn))
    ordered = [n for n in ordered if n not in evidence]
    while True:
        result = dict(evidence)
        for node in ordered:
            result[node] = node.sample(result)[0]
        yield result


_bif_var_pattern = re.compile(r"variable\s+(.*)\s*{\s*type\s+discrete\s+\[\s*(.*)\s*]\s*{\s*(.*)\s*};\s*}")
_bif_table_pattern = re.compile(r"probability\s+\(\s*(.*)\s*\)\s*{\s*table\s+(.*);\s*}")
_bif_cpt_pattern = re.compile(r"probability\s+\(\s*(.*)\s*\|\s*(.*?)\s*\)\s*{([\s\S]*?)}")
_bif_cpt_row_pattern = re.compile(r"\s*\(\s*(.*)\s*\)\s*(.*)\s*;")


def load_bif(bif: AnyStr, load_cpt=True) -> list[Node]:
    """Deserialize from Bayesian Interchange Format."""
    nodes: list[Node] = []
    for match in _bif_var_pattern.finditer(bif):
        node = Node(Variable(match[1].strip(), [s.strip() for s in match[3].split(',')]))
        node.parents, node.children = [], []
        nodes.append(node)
    if not load_cpt:
        return nodes
    by_id = {n.id: n for n in nodes}
    for match in _bif_table_pattern.finditer(bif):
        node = by_id[match[1].strip()]
        node.cpt = np.array([float(s.strip()) for s in match[2].split(',')])
    for match in _bif_cpt_pattern.finditer(bif):
        node = by_id[match[1].strip()]
        node.parents = [by_id[s.strip()] for s in match[2].split(',')]
        for p in node.parents:
            p.children.append(node)
        node.cpt = np.zeros(tuple(n.arity for n in node.parents) + (len(node.values),))
        for submatch in _bif_cpt_row_pattern.finditer(match[3]):
            index = tuple(n.values.index(s.strip()) for n, s in zip(node.parents, submatch[1].split(',')))
            values = [float(s.strip()) for s in submatch[2].split(',')]
            node.cpt[index + (slice(None),)] = values
    return nodes


def load_bif_file(path):
    with open(path, 'rt') as fp:
        return load_bif(fp.read())


def event_values(event: Event, variables: Iterable[Variable]) -> Tuple[Optional[Value], ...]:
    """Return the value of each assignment in the order of the given variable sequence."""
    return tuple(event.get(var) for var in variables) if event else tuple()


def save_bif(bn: BayesNet, output: Optional[TextIO] = None) -> Optional[str]:
    """Serialize to Bayesian Interchange Format."""
    return_string = False
    if output is None:
        output = io.StringIO()
        return_string = True
    output.write("network unknown {\n}\n")
    for n in bn:
        values = (str(v) for v in n.values)
        output.write(f"variable {n.id} {{\n  type discrete [ {n.arity} ] {{ {', '.join(values)} }};\n}}\n")
    for n in bn:
        if not n.parents and n.cpt is not None:
            probabilities = (str(v) for v in n.cpt)
            output.write(f"probability ( {n.id} ) {{\n  table {', '.join(probabilities)};\n}}\n")
        if n.parents and n.cpt is not None:
            parents = (str(p.id) for p in n.parents)
            output.write(f"probability ( {n.id} | {', '.join(parents)} ) {{\n")
            for ev in all_events(n.parents):
                values = (str(v) for v in event_values(ev, n.parents))
                probabilities = (str(v) for v in n.cpt[event_index(ev, n.parents) + (slice(None),)])
                output.write(f"  ({', '.join(values)}) {', '.join(probabilities)};\n")
            output.write("}\n")
    if return_string:
        output.seek(0)
        return output.read()


def load_bn(path):
    with open(path, 'rt') as fp:
        return load_bif(fp.read())


def sample(bn: BayesNet, n=1000) -> Dataset:
    bn = list(bn)
    observations = np.empty((n, len(bn)), dtype=np.uint8)
    for i, event in enumerate(itertools.islice(prior_sampling(bn), n)):
        observations[i, :] = event_index(event, bn)
    return Dataset(bn, observations)


# Testing related code:

def download_gz(url, file):
    if os.path.exists(file):
        return
    with urllib.request.urlopen(url) as response, open(file, 'wb') as out_file:
        out_file.write(gzip.decompress(response.read()))


def prepare_tests():
    # Download BNs from the bnlearn repository https://www.bnlearn.com/bnrepository/

    download_gz('https://www.bnlearn.com/bnrepository/asia/asia.bif.gz', 'asia.bif')
    bn = load_bn('asia.bif')
    print("Sampling Asia BN")
    random.seed(123)
    ds = sample(bn, 10000)
    ds.save("asia.npz")

    download_gz('https://www.bnlearn.com/bnrepository/alarm/alarm.bif.gz', 'alarm.bif')
    bn = load_bn('alarm.bif')
    print("Sampling Alarm BN")
    random.seed(123)
    ds = sample(bn, 1000000)
    ds.save("alarm.npz")

    download_gz('https://www.bnlearn.com/bnrepository/hailfinder/hailfinder.bif.gz', 'hailfinder.bif')
    bn = load_bn('hailfinder.bif')
    print("Sampling Hailfinder BN")
    random.seed(123)
    ds = sample(bn, 1000000)
    ds.save("hailfinder.npz")

    bn = load_bn('large_bn.bif')
    print("Sampling the large BN")
    random.seed(123)
    ds = sample(bn, 1000000)
    ds.save('large_bn.npz')


ConfusionMatrix = namedtuple('ConfusionMatrix', ['tp', 'fp', 'fn', 'tn'])


def confusion_matrix(population: Iterable, predicted_positives: Container, actual_positives: Container):
    tp, fp, fn, tn = 0, 0, 0, 0
    for x in population:
        if x in predicted_positives:
            if x in actual_positives:
                tp += 1
            else:
                fp += 1
        else:
            if x in actual_positives:
                fn += 1
            else:
                tn += 1
    return ConfusionMatrix(tp, fp, fn, tn)


def test_all_markov_blankets(ds: Dataset, bn: BayesNet) -> list[ConfusionMatrix]:
    """Find all Mbs from a dataset. Return the validation results based on the BN"""

    mbs = list(sorted(((n, markov_blanket((n,))) for n in bn), key=lambda nmb: len(nmb[1])))
    results = []
    for i in range(0, len(mbs)):
        n, mb = mbs[i]
        heuristic = SmartIAMB()  # SensitiveIAMB(0.009)
        result = find_markov_blanket((n,), set(ds.variables), ds, heuristic)
        cm = confusion_matrix(bn, result, mb)
        results.append(cm)
        print(i, n, cm)
    return results


def test_metrics(results: list[ConfusionMatrix]):
    """Group the results by Mb size. Print the average TNR, TPR for each size."""
    grouped_results = dict()
    for cm in results:
        mb_size = cm.tp + cm.fn
        if mb_size not in grouped_results:
            grouped_results[mb_size] = list()
        grouped_results[mb_size].append(cm)

    for mb_size in grouped_results:
        cms = grouped_results[mb_size]
        for i, cm in enumerate(cms):
            tnr = cm.tn / (cm.fp + cm.tn)
            tpr = cm.tp / (cm.tp + cm.fn)
            cms[i] = (tnr, tpr)

    size = max(grouped_results.keys()) + 1
    result_tnr_tpr: MutableSequence[Optional] = [None] * size
    for mb_size in grouped_results:
        arr = np.array(grouped_results[mb_size]).transpose()
        mean_tnr, mean_tpr = np.mean(arr[0]), np.mean(arr[1])
        result_tnr_tpr[mb_size] = (mean_tnr, mean_tpr)
    return result_tnr_tpr


def test_iamb():
    """Find all MBs in these four BNs."""

    ds, bn = Dataset.load('asia.npz'), load_bn('asia.bif')
    print("Testing Asia BN")
    asia_results = test_all_markov_blankets(ds, bn)

    ds, bn = Dataset.load('alarm.npz'), load_bn('alarm.bif')
    print("Testing Alarm BN")
    alarm_results = test_all_markov_blankets(ds, bn)

    ds, bn = Dataset.load('hailfinder.npz'), load_bn('hailfinder.bif')
    print("Testing Hailfinder BN")
    hailfinder_results = test_all_markov_blankets(ds, bn)

    ds, bn = Dataset.load("large_bn.npz"), load_bn('large_bn.bif')
    print("Testing the large BN")
    large_results = test_all_markov_blankets(ds, bn)

    joint_results = asia_results + alarm_results + hailfinder_results + large_results

    print(test_metrics(joint_results))


def main():
    test_iamb()


if __name__ == '__main__':
    main()
