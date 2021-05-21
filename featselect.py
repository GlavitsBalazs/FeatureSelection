import itertools, random
from abc import ABC, abstractmethod
from typing import AbstractSet, Any, Callable, Container, FrozenSet, Hashable, Iterable, Mapping, MutableMapping, \
    MutableSequence, MutableSet, Optional, Sequence, Tuple, Union

import numpy as np

Value = Any  # Type alias for the value of a Variable.


class Variable:
    """A random variable with finite range."""

    def __init__(self, id_: Hashable, values: Sequence[Value]):
        self.id = id_
        self.values = values

    @property
    def arity(self):
        return len(self.values)

    def __eq__(self, other: Optional['Variable']):
        return other is not None and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        """Print the ID for debug purposes."""
        if isinstance(self.id, str):
            return self.id
        else:
            return f"Var_{self.id}"


Event = Mapping[Variable, Value]  # A set of statements assigning values to variables.
EmptyEvent: Event = dict()


def event_index(event: Event, variables: Iterable[Variable]) -> Tuple[Union[int, slice], ...]:
    """Return a numpy index pointing to the given event in
    a multidimensional probability table of the given variables."""

    def _items():
        for var in variables:
            val = event.get(var)
            if val is not None:
                yield var.values.index(val)
            else:
                yield slice(None)

    return tuple(_items())


def all_events(variables: Iterable[Variable]) -> Iterable[Event]:
    """Return all events observable from the given variables."""
    for values in itertools.product(*(var.values for var in variables)):
        yield {var: val for var, val in zip(variables, values)}


Probability = float
Information = float  # In bits. Mind the logarithm.


class ProbabilityTable(Mapping[Event, Probability]):
    """A unnormalized multidimensional probability table."""

    def __init__(self, variables: Sequence[Variable], table: np.ndarray):
        self.variables = variables
        self.table = table

    def __getitem__(self, key: Event) -> Probability:
        return self.table[event_index(key, self.variables)]

    def __setitem__(self, key: Event, value: Probability):
        self.table[event_index(key, self.variables)] = value

    def __len__(self):
        return np.prod(self.table.shape, dtype=np.uint32)

    def __iter__(self):
        return all_events(self.variables)

    def __eq__(self, other):
        return object.__eq__(self, other)

    def sum_out(self, targets: Container[Variable]):
        if not targets:
            return self
        other_variables = [v for v in self.variables if v not in targets]
        target_axes = tuple(i for i, v in enumerate(self.variables) if v in targets)
        return ProbabilityTable(other_variables, np.sum(self.table, axis=target_axes))

    def select(self, targets: Iterable[Variable]):
        to_be_removed = set(self.variables)
        for t in targets:
            to_be_removed.remove(t)
        return self.sum_out(to_be_removed)

    def normalize(self) -> 'ProbabilityTable':
        norm = np.sum(self.table)
        if norm != 0:
            self.table /= norm
        return self

    def entropy(self, targets: Optional[AbstractSet[Variable]] = None) -> 'Information':
        pruned = self.select(targets) if targets is not None else self
        norm = np.sum(pruned.table)
        probabilities = pruned.table.reshape(-1)  # flatten
        logs = np.zeros(len(probabilities), dtype=np.float64)
        np.log2(probabilities, out=logs, where=probabilities > 0)
        return np.log2(norm) - np.dot(probabilities, logs) / norm


class Dataset(Sequence[Event]):
    """A collection of samples from variables."""

    # The strategy is chosen depending on whether the Numba dependency is satisfied.
    _count_strategy: Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray] = None

    def __init__(self, variables: Sequence[Variable], observations: np.ndarray):
        """observations[i, j] codes the value that the j-th variable takes in the i-th sample."""
        self.variables = variables
        self.observations = observations
        self._column_of = {var: col for col, var in enumerate(variables)}
        self._entropy_cache: MutableMapping[FrozenSet[Variable], Information] = {frozenset(): 0}

    def __getitem__(self, i: int) -> Event:
        obs = self.observations[i]
        return {var: var.values[obs[j]] for j, var in enumerate(self.variables)}

    def __len__(self) -> int:
        return len(self.observations)

    def select(self, variables: Sequence[Variable]) -> 'Dataset':
        """Select the given columns."""
        if variables == self.variables:
            return self
        columns = tuple(self._column_of[var] for var in variables)
        return Dataset(variables, self.observations[:, columns])

    def save(self, file, compressed=False):
        savez_func = np.savez_compressed if compressed else np.savez
        var_copy = [Variable(v.id, v.values) for v in self.variables]
        savez_func(file=file, variables=np.array(var_copy), observations=self.observations)

    @staticmethod
    def load(file) -> 'Dataset':
        # deserializing pickle could open an arbitrary code execution exploit
        npz = np.load(file=file, allow_pickle=True)
        return Dataset(npz['variables'], npz['observations'])

    def count(self, targets: Sequence[Variable]) -> ProbabilityTable:
        """Count the number occurrences of each event to a ProbabilityTable.
        A special hash function is used that produces no collisions. Each coded event from the Dataset is treated
        as a sequence of digits in a positional number system. Indexing the hash table is based on the
        numeric values of these digit sequences."""
        arities = tuple(v.arity for v in targets)
        positions = np.ones(len(targets), dtype=np.uint32)
        for i in range(1, len(targets)):
            m = arities[i]
            for j in range(i):
                positions[j] *= m
        bins = positions[0] * arities[0]
        columns = np.array([self._column_of[var] for var in targets])
        counts = Dataset._count_strategy(self.observations, columns, positions, bins)
        return ProbabilityTable(targets, counts.reshape(arities))

    def entropies(self, *target_sets: FrozenSet[Variable]) -> Sequence[Information]:
        """Compute multiple multiple entropy values with only one count operation. Cache the results."""
        result: MutableSequence[Information] = [0.0] * len(target_sets)
        to_be_counted = set()
        for i, target in enumerate(target_sets):
            if target in self._entropy_cache:
                result[i] = self._entropy_cache[target]
            else:
                for v in target:
                    to_be_counted.add(v)
        if to_be_counted:
            counts = self.count(list(to_be_counted))
            for i, target in enumerate(target_sets):
                if target not in self._entropy_cache:
                    h = counts.entropy(target)
                    self._entropy_cache[target] = h
                    result[i] = h
        return result

    def mutual_information(self, x: FrozenSet[Variable], y: FrozenSet[Variable],
                           z: Optional[FrozenSet[Variable]] = None) -> Information:
        """The mutual information between the variable sets x and y conditioned on z."""
        if not z:
            h_x, h_y, h_xy = self.entropies(x, y, x | y)
            return h_x + h_y - h_xy
        h_xz, h_yz, h_xyz, h_z = self.entropies(x | z, y | z, x | y | z, z)
        return h_xz + h_yz - h_xyz - h_z


try:
    from numba import njit  # https://numba.pydata.org/


    @njit
    def _fast_count(observations: np.ndarray, columns: np.ndarray, positions: np.ndarray, bins: int) -> np.ndarray:
        counts = np.zeros(bins, dtype=np.uint32)
        h, w = len(observations), len(columns)
        for i in range(h):
            v = 0
            obs = observations[i]
            for j in range(w):
                v += obs[columns[j]] * positions[j]
            counts[v] += 1
        return counts


    Dataset._count_strategy = _fast_count
except ModuleNotFoundError:
    # The optional dependency is not satisfied.
    print("debug: running without Numba")


    def _slow_count(observations: np.ndarray, columns: np.ndarray, positions: np.ndarray, bins: int) -> np.ndarray:
        return np.bincount(np.dot(observations[:, tuple(columns)], positions), minlength=bins)


    Dataset._count_strategy = _slow_count


class IAMBHeuristic(ABC):
    @abstractmethod
    def dependent_heuristic(self, cmi_values: Mapping[Variable, Information]) -> Optional[Variable]:
        """Given their CMI values, find a variable that is conditionally dependent"""
        pass

    @abstractmethod
    def independent_heuristic(self, cmi_values: Mapping[Variable, Information]) -> Optional[Variable]:
        """Given their CMI values, find a variable that is conditionally independent"""
        pass


class SensitiveIAMB(IAMBHeuristic):
    """Uses a constant threshold for conditional mutual information tests."""

    def __init__(self, sensitivity: Information, k: Optional[Probability] = None):
        super().__init__()
        self._sensitivity = sensitivity
        self._k = k

    def _k_select(self, sample: Sequence):
        """This function is used to implement KIAMB from https://doi.org/10.1016/j.ijar.2006.06.008"""
        if self._k is not None and 0.0 < self._k < 1.0:
            size = max(1, int(len(sample) * self._k))
            return random.sample(sample, size)
        return sample

    def dependent_heuristic(self, cmi_values: Mapping[Variable, Information]) -> Optional[Variable]:
        candidates = [(x, cmi) for x, cmi in cmi_values.items() if cmi >= self._sensitivity]
        if not candidates:
            return None
        candidates = self._k_select(candidates)
        return max(candidates, key=lambda x_cmi: x_cmi[1])[0]

    def independent_heuristic(self, cmi_values: Mapping[Variable, Information]) -> Optional[Variable]:
        candidates = [(x, cmi) for x, cmi in cmi_values.items() if cmi < self._sensitivity]
        if not candidates:
            return None
        return min(candidates, key=lambda x_cmi: x_cmi[1])[0]


class SmartIAMB(IAMBHeuristic):
    """Tries to guess the point when there are no more conditionally dependent variables."""

    def __init__(self):
        super().__init__()
        self._dep_threshold: Optional[Information] = None
        self._indep_threshold = 0.0001
        self._stop_next = False
        self._stop_next2 = False

    def dependent_heuristic(self, cmi_values: Mapping[Variable, Information]) -> Optional[Variable]:
        cmi_sorted = list(cmi_values.items())
        cmi_sorted.sort(key=lambda x_cmi: x_cmi[1], reverse=True)
        max_x, max_cmi = cmi_sorted[0]

        # Handle the trivial cases.
        if max_cmi == 0:
            return None
        if self._dep_threshold is None:
            self._dep_threshold = max_cmi
            return max_x

        # Make some observations about the current situation.
        cmis = list(cmi for _, cmi in cmi_sorted if cmi > 0)
        logs = np.log(cmis)
        p01 = np.percentile(logs, 1)
        p99 = np.percentile(logs, 99)
        outliers_removed = [log_cmi for log_cmi in logs if p01 <= log_cmi <= p99]
        stddev = None
        enough_vars = len(outliers_removed) > 1
        if enough_vars:
            stddev = np.std(outliers_removed)
        good_round = enough_vars and stddev > 1.3
        very_bad_round = enough_vars and stddev < 0.5
        bad_round = enough_vars and stddev < 1.0 and not very_bad_round
        obvious_victory = self._dep_threshold / max_cmi > 100
        threshold_ran_out = self._dep_threshold > max_cmi
        very_uncertain = max_cmi < 0.0005

        # Should we exit?
        if good_round:
            self._stop_next = False
        if self._stop_next:
            return None
        if very_bad_round:
            return None
        if threshold_ran_out:
            if obvious_victory:
                return None
            if very_uncertain:
                return None
            if bad_round:
                self._stop_next = True
            self._dep_threshold = max_cmi

        # Just return the best guess.
        candidates = [(x, cmi) for x, cmi in cmi_sorted if cmi >= self._dep_threshold]
        return max(candidates, key=lambda x_cmi: x_cmi[1])[0]

    def independent_heuristic(self, cmi_values: Mapping[Variable, Information]) -> Optional[Variable]:
        cmi_sorted = list(cmi_values.items())
        cmi_sorted.sort(key=lambda x_cmi: x_cmi[1])
        min_x, min_cmi = cmi_sorted[0]
        if min_cmi < self._indep_threshold:
            return min_x
        return None


class GreedyIAMB(IAMBHeuristic):
    """Always include more variables."""

    def __init__(self, sensitivity: Information):
        super().__init__()
        self._sensitivity = sensitivity

    def dependent_heuristic(self, cmi_values: Mapping[Variable, Information]) -> Optional[Variable]:
        cmi_sorted = list(cmi_values.items())
        cmi_sorted.sort(key=lambda x_cmi: x_cmi[1], reverse=True)
        max_x, max_cmi = cmi_sorted[0]
        return max_x

    def independent_heuristic(self, cmi_values: Mapping[Variable, Information]) -> Optional[Variable]:
        cmi_sorted = list(cmi_values.items())
        cmi_sorted.sort(key=lambda x_cmi: x_cmi[1])
        min_x, min_cmi = cmi_sorted[0]
        if min_cmi < self._sensitivity:
            return min_x
        return None


def _iamb_grow(result: MutableSet[Variable], external_variables: AbstractSet[Variable],
               targets: AbstractSet[Variable], prior_blanket: AbstractSet[Variable], ds: Dataset,
               dependent_heuristic: Callable[[Mapping[Variable, Information]], Optional[Variable]]):
    """
    Growing phase of IAMB.

    :param result: Found Mb variables are collected here.
    :param prior_blanket: Variables that are known Mb members.
    :param dependent_heuristic: See IAMBHeuristic.dependent_heuristic
    """
    frozen_targets = frozenset(targets)
    remaining_external_variables = set(external_variables)
    while True:
        mb_set = frozenset(result) | prior_blanket
        cmi_values = {x: ds.mutual_information(frozen_targets, frozenset((x,)), mb_set)
                      for x in remaining_external_variables}
        dependent = dependent_heuristic(cmi_values)
        if dependent is None:
            break
        result.add(dependent)
        remaining_external_variables.remove(dependent)
        if len(result | prior_blanket) > 16:  # The set grew too much. CMI computation became too slow.
            print("IAMB: timeout!")
            break


def _iamb_shrink(blanket: MutableSet[Variable], targets: AbstractSet[Variable], ds: Dataset,
                 independent_heuristic: Callable[[Mapping[Variable, Information]], Optional[Variable]],
                 not_to_be_removed: Optional[AbstractSet[Variable]] = frozenset()):
    """
    Shrinking phase of IAMB.

    :param blanket: Erroneous Mb variables are removed from here.
    :param not_to_be_removed: Variables that are known Mb members.
    :param independent_heuristic: See IAMBHeuristic.independent_heuristic
    """
    unchecked = list(blanket)
    frozen_targets = frozenset(targets)
    while len(unchecked) > 0:
        cmi_values = {x: ds.mutual_information(frozen_targets, frozenset((x,)),
                                               frozenset(blanket - {x}) | not_to_be_removed)
                      for x in unchecked}
        independent = independent_heuristic(cmi_values)
        if independent is None:
            break
        blanket.remove(independent)
        unchecked.remove(independent)


def _iamb(targets: AbstractSet[Variable], external_variables: AbstractSet[Variable],
          ds: Dataset, heuristic: IAMBHeuristic) -> set[Variable]:
    """
    Incremental Association Markov Blanket algorithm.

    As seen in https://www.aaai.org/Papers/FLAIRS/2003/Flairs03-073.pdf
    :param targets: The set of variables whose Mb is desired.
    :param external_variables: Variables out of which the Mb is to be chosen.
    :param ds: Dataset of observations.
    :param heuristic: Heuristic that determines conditional independence.
    :return: The Markov blanket set.
    """
    mb = set()
    if heuristic is None:
        heuristic = SmartIAMB()
    _iamb_grow(result=mb, external_variables=external_variables, targets=targets,
               prior_blanket=set(), ds=ds, dependent_heuristic=heuristic.dependent_heuristic)
    _iamb_shrink(blanket=mb, targets=targets, ds=ds,
                 independent_heuristic=heuristic.independent_heuristic)
    return mb


def _iambs(targets_1: AbstractSet[Variable], targets_2: AbstractSet[Variable],
           boundary_1: AbstractSet[Variable], boundary_2: AbstractSet[Variable],
           variables: AbstractSet[Variable], ds: Dataset, heuristic: IAMBHeuristic) \
        -> set[Variable]:
    """
    Incremental Association Markov Blanket Supplement algorithm.

    As seen in https://jmlr.org/papers/v19/14-033.html
    :param targets_1: One set of targets.
    :param targets_2: Other, distinct set of targets.
    :param boundary_1: The Mb of targets_1
    :param boundary_2: The Mb of targets_2
    :param variables: All variables.
    :param ds: Dataset of observations.
    :param heuristic: Heuristic that determines conditional independence.
    :return: The Mb of targets_1 | targets_2.
    """
    target_union = targets_1 | targets_2
    mb_union = set((boundary_1 | boundary_2) - target_union)
    external_variables = variables - mb_union - target_union
    supplement: MutableSet[Variable] = set()
    _iamb_grow(result=supplement, external_variables=external_variables, targets=targets_2,
               prior_blanket=mb_union, ds=ds, dependent_heuristic=heuristic.dependent_heuristic)
    _iamb_shrink(blanket=supplement, targets=targets_2, ds=ds,
                 independent_heuristic=heuristic.independent_heuristic, not_to_be_removed=mb_union)
    _iamb_shrink(blanket=mb_union, targets=target_union, ds=ds,
                 independent_heuristic=heuristic.independent_heuristic, not_to_be_removed=supplement)
    return mb_union | supplement


def _miamb(targets: Sequence[Variable], variables: AbstractSet[Variable],
           ds: Dataset, heuristic: IAMBHeuristic) -> set[Variable]:
    """
    Multivariable Incremental Association Markov Blanket algorithm.

    As seen in https://jmlr.org/papers/v19/14-033.html
    :param targets: The set of target variables. Ordering of increasing Mb size may yield faster computation.
    :param variables: All variables.
    :param ds: Dataset of observations.
    :param heuristic: Heuristic that determines conditional independence.
    :return: The Markov blanket set.
    """
    individual_blankets = [_iamb({var}, variables - {var}, ds, heuristic) for var in targets]
    result = individual_blankets[0]
    for i in range(1, len(individual_blankets)):
        result = _iambs(set(targets[:i]), {targets[i]}, result, individual_blankets[i],
                        variables, ds, heuristic)
    return result


def find_markov_blanket(targets: Sequence[Variable], variables: AbstractSet[Variable], ds: Dataset,
                        heuristic: Optional[IAMBHeuristic] = None) -> set[Variable]:
    """
    Find the Markov blanket subset.
    
    The target variables are conditionally independent of all other variables given the Markov blanket set.
    :param targets: The set of target variables. Ordering of increasing Mb size may yield faster computation.
    :param variables: All variables.
    :param ds: Dataset of observations.
    :param heuristic: Heuristic that determines conditional independence. Defaults to SmartIAMB.
    :return: The Markov blanket set.
    """
    if heuristic is None:
        heuristic = SmartIAMB()
    return _miamb(targets, variables, ds, heuristic)
