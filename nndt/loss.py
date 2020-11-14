import numpy as np
import torch

import sympy
from sympy.logic.boolalg import to_cnf
from pysdd.sdd import CompilerOptions, Fnf, SddManager, Vtree
from semantic_loss_pytorch import SemanticLoss, ConstraintsToCnf
from pathlib import Path

from sklearn.tree import _tree, DecisionTreeClassifier
from collections import namedtuple


class _DistillationLoss:
    """Implements a distillation loss on inputs, outputs, and explanations."""

    def set_cnf(self, cnf, n_inputs, n_targets):
        self.n_inputs, self.n_targets = n_inputs, n_targets

        BASENAME = 'distillation-loss'

        # TODO avoid dumping stuff on disk
        with open(BASENAME + '.sympy', 'wt') as fp:
            fp.write(f'shape [{2 * n_inputs + n_targets}, {2}]\n')
            for clause in cnf.args:
                fp.write(f'{str(clause)}\n')

        ConstraintsToCnf.expression_to_cnf(BASENAME + '.sympy',
                                           BASENAME + '.dimacs',
                                           1)

        fnf = Fnf.from_cnf_file(bytes(Path(BASENAME + '.dimacs')))
        vtree = Vtree(var_count=fnf.var_count, vtree_type='balanced')
        manager = SddManager.from_vtree(vtree)
        sdd = manager.fnf_to_sdd(fnf)
        sdd.save(bytes(Path(BASENAME + '.sdd')))
        manager.vtree().save(bytes(Path(BASENAME + '.vtree')))

        self.sl = SemanticLoss(BASENAME + '.sdd', BASENAME + '.vtree')

    def encode(self):
        """Returns a sympy CNF, n_inputs, n_outputs."""
        raise NotImplementedError('use a derived class')

    def sync(self):
        self.set_cnf(*self.encode())

    def loss(self, data, logprobs, relevance=None):
        """Evaluates the Semantic Loss on a batch."""
        if relevance is not None:
            # TODO add support for input gradients
            raise NotImplementedError()

        probs = torch.exp(logprobs)

        n_examples = len(data)
        expanded_probs = torch.zeros((n_examples, self.n_targets, 2))
        expanded_probs[:, 0, 0] = 1 - probs[:, 0]
        expanded_probs[:, 0, 1] = probs[:, 0]
        expanded_probs[:, 1, 0] = 1 - probs[:, 1]
        expanded_probs[:, 1, 1] = probs[:, 1]

        distillation_probs = torch.cat((
            torch.ones((n_examples, self.n_inputs, 2)) * 0.5, # P(X)
            torch.ones((n_examples, self.n_inputs, 2)) * 0.5, # P(Z|X)
            expanded_probs, # P(Y|Z,X)
        ), dim=1)

        return self.sl(probabilities=distillation_probs)


class DecisionTreeLoss(_DistillationLoss, DecisionTreeClassifier):
    """Distillation loss relative to a decision tree."""
    def __init__(self, dataset, use_relevance=False):
        DecisionTreeClassifier.__init__(self)
        self.n_inputs = dataset.data.shape[1]
        self.n_targets = len(dataset.target_names)
        self.use_relevance = use_relevance


    def _extract_leaves(self):
        Leaf = namedtuple('Leaf', ['conds', 'klass'])

        tree_ = self.tree_

        def recurse(conds, node):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                i = tree_.feature[node]
                assert i != _tree.TREE_UNDEFINED
                threshold = tree_.threshold[node]
                assert threshold == 0.5, 'non-binary input!'
                leaves_l = recurse(conds | {(i, 0)}, # input i == 0
                                   tree_.children_left[node])
                leaves_r = recurse(conds | {(i, 1)}, # input i == 1
                                   tree_.children_right[node])
                return leaves_l + leaves_r
            else:
                label_probs = tree_.value[node].ravel()
                return [Leaf(conds, label_probs.argmax())]

        return recurse(set(), 0)


    def _encode_mutex(self, outputs):
        """One and only one label can be predicted at any one time."""
        n_outputs = len(outputs)

        formulas = []
        for i in range(n_outputs):
            no_other = sympy.And(*[outputs[j, 0] for j in range(n_outputs)
                                   if i != j])
            formulas.append(outputs[i, 1] & no_other)
        return to_cnf(sympy.Or(*formulas))


    def _encode_leaf_to_y(self, leaves, inputs, outputs):
        """Each leaf entails one and only one label."""
        formulas = []
        for leaf in leaves:
            cond = sympy.And(*[inputs[i, value] for i, value in leaf.conds])
            formulas.append(cond & outputs[leaf.klass, 1])
        return to_cnf(sympy.Or(*formulas))


    def _encode_leaf_to_z(self, leaves, inputs, attributions):
        """Each leaf entails one and only one feature usage mask."""
        n_inputs = inputs.shape[0]

        formulas = []
        for leaf in leaves:
            relevant = {i for i, _ in leaf.conds}
            cond = sympy.And(*[inputs[i, value] for i, value in leaf.conds])
            mask = sympy.And(*[attributions[i, 1 if i in relevant else 0]
                               for i in range(n_inputs)])
            formulas.append(cond & mask)
        return to_cnf(sympy.Or(*formulas))


    def encode(self):
        """Converts this decision tree to a sympy CNF formula."""
        n_inputs, n_targets = self.n_inputs, self.n_targets

        x_symbols = np.array([(sympy.Symbol(f'X{i}.0'), sympy.Symbol(f'X{i}.1'))
                               for i in range(0, n_inputs)])
        z_symbols = np.array([(sympy.Symbol(f'X{i}.0'), sympy.Symbol(f'X{i}.1'))
                               for i in range(n_inputs, 2 * n_inputs)])
        y_symbols = np.array([(sympy.Symbol(f'X{i}.0'), sympy.Symbol(f'X{i}.1'))
                               for i in range(2 * n_inputs, 2 * n_inputs + n_targets)])

        leaves = self._extract_leaves()

        constraints = [
            self._encode_mutex(y_symbols),
            self._encode_leaf_to_y(leaves, x_symbols, y_symbols),
        ]
        if self.use_relevance:
            constraints.append(self._encode_leaf_to_z(leaves, x_symbols, z_symbols))

        # XXX THIS IS FUCKING SLOW
        constraint = to_cnf(sympy.And(*constraints))

        return constraint, n_inputs, n_targets

