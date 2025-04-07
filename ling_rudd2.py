"""
Anonymous version for review
"""
from copy import deepcopy
from collections import defaultdict

class Participant:
    """ simple class to hold participant data """
    bpa = None
    s = None
    sources = None
    disjoint = None
    shared = None

    def __init__(self, _bpa:dict, _s:float, _sources:set):
        """ Constructor """
        self.bpa = _bpa
        self.s = _s
        self.sources = _sources

    def __str__(self):
        """ string representation """
        return (
            f"Participant(bpa={self.bpa}"
            f", s={self.s}, sources={self.sources}, disjoint={self.disjoint}, shared={self.shared})"
        )

    def __repr__(self):
        """ debug representation """
        return (
            f"Participant(bpa={self.bpa}"
            f", s={self.s}, sources={self.sources}, disjoint={self.disjoint}, shared={self.shared})"
        )
    
    def get_hypothesis(self, i=0):
        """
        * store the hypothesis in h 
        * should be done AFTER separation into simple support functions
        """
        return list(self.bpa.keys())[i]

def dempster_combination(m1, m2):
    """
    Combine two mass functions using Dempster's Rule of Combination.
    This is for INDEPENDENT EXPERTS

    Reproduces the results presented by Li and Rudd (1989) Equation 1:
    - Example 3 if s1 & s2 set to 1
    - Examples 4 & 5 if s1 & s2 set to 1, 0.7

    Parameters:
    - m1: Dictionary representing the first mass function {subset: mass}.
    - m2: Dictionary representing the second mass function {subset: mass}.
    - frame: Set of all elements in the frame of discernment.

    Returns:
    - A dictionary representing the combined mass function.
    """
    combined_mass = {}
    conflict = 0

    # Loop through all pairs of focal elements from m1 and m2
    for A, m1_val in m1.items():
        for B, m2_val in m2.items():

            # compute the intersection for each combination
            intersection = A.intersection(B)

            # Disjoint sets contribute to the conflict
            if not intersection:  
                conflict += m1_val * m2_val

            # Otherwise, calculate the combined mass for the intersection
            else:  
                combined_mass[intersection] = combined_mass.get(intersection, 0) + m1_val * m2_val

    # invert conflict to get the denominator
    normalization_factor = 1 - conflict
    if normalization_factor == 0:
        raise ValueError("Conflict is total; no combination possible.")

    # Normalize the combined mass function & return
    for subset in combined_mass:
        combined_mass[subset] /= normalization_factor
    return combined_mass

def compute_belief(mass_function, subset):
    """
    Compute the belief of a subset in the mass function.
    Reproduces Rudd and Li (1989) Equation 4

    This is actually unnecessary because we are separating everything into 
        simple support functions, but I have left it in for completeness.

    Parameters:
    mass_function (dict): The mass function.
    subset (frozenset): The subset for which to calculate belief.

    Returns:
    float: The belief value.
    """
    return sum(mass for s, mass in mass_function.items() if s.issubset(subset))

def compute_plausibility(mass_function, subset):
    """
    Compute the plausibility of a subset in the mass function.

    Parameters:
    mass_function (dict): The mass function.
    subset (frozenset): The subset for which to calculate plausibility.

    Returns:
    float: The plausibility value.
    """
    return sum(mass for s, mass in mass_function.items() if s & subset)

def to_simple_support_functions(bpa):
    """
    Decompose a list of simple and separable BPAs into canonical simple support functions (SSFs).
    
    Parameters:
    - bpa (dict): The original BPA with focal elements and their masses.
    - frame (set): The frame of discernment.
    
    Returns:
    - List of SSFs, where each SSF is a dict.
    """
    return [{focal_element: mass} for focal_element, mass in bpa.items()]

def compute_decomposition(m1, w_shared, w_disjoint):
    """
    Decompose evidence into that which is shared and that which is disjoint
    As per Ling & Rudd (1989) Theorem 2
    """
    # compute total w
    total_w = w_disjoint + w_shared
    
    # if total is 0 then we have no information on which to split, so allocate to shared
    if total_w == 0:
        return 0, m1
    
    # calculate dependence
    dependence = w_shared / (total_w)
    
    # if it is entirely dependent, then allocate everything to shared and return
    if dependence == 1:
        return 0, m1
    
    # otherwise, decompose as per Ling & Rudd
    k = dependence / (1 - dependence)
    disjoint_evidence = 1 - (1 - m1) ** (1 / (k + 1))
    shared_evidence = 1 - (1 - m1) ** (1 - 1 / (k + 1))
    return disjoint_evidence, shared_evidence

def reconcile_disagreement(weights, shared_masses):
    """
    Combine m12 and m12 into a single level of belief (basically a weighted average)
    As per Ling & Rudd (1989) Eq 19

    Modified to accommodate a list rather than a pair

    s1 & s2 are the weights (confidence levels) in evidence 1 & 2
    m11/22 is disjoint
    m12 is proportion of evidence in 1 shared with 2
    m21 is proportion of evidence in 2 shared with 1
    """
    weighted_sum = sum(value * weight for value, weight in zip(shared_masses, weights))
    total_weight = sum(weights)
    return weighted_sum / total_weight

def lr_combination(participants, report=False):
    """
    Combine non-independent evidence in the form of simple and/or seperable support functions using 
        a modified version of Ling and Rudd's method (1989)
    """

    # set frame of discernment (can't calculate as we don't always have all hypotheses)
    frame = frozenset({'trees', 'no trees'})

    ''' STEP 1: Decompose into simple support functions and overlapping & disjoint evidence '''

    if report:
        print("\nStage 1: Decomposition into simple support functions and overlapping & disjoint evidence")

    # loop through all participants to combine
    ss_participants = []
    for i, p in enumerate(participants):

        # get the number of sources for all other participants and the number that are shared
        all_other_sources = []
        for ii, pp in enumerate(participants):
            if ii != i:
                all_other_sources += pp.sources
        all_other_sources = set(all_other_sources)
        n_shared_sources = len(p.sources.intersection(all_other_sources))

        # separate into simple support functions if required
        if len(p.bpa) > 1:
            for ssf in to_simple_support_functions(p.bpa):
                q = deepcopy(p)
                q.bpa = ssf

                # decompose and append
                q.disjoint, q.shared = compute_decomposition(q.bpa[q.get_hypothesis()], n_shared_sources, len(q.sources) - n_shared_sources)
                ss_participants.append(q)

                if report:
                    print(f"disjoint = {q.disjoint:.4f}, shared = {q.shared:.4f}")
        
        # just decompose and append if not required
        else:
            p.disjoint, p.shared = compute_decomposition(p.bpa[p.get_hypothesis()], n_shared_sources, len(p.sources) - n_shared_sources)
            ss_participants.append(p)

            if report:
                print(f"disjoint = {p.disjoint:.4f}, shared = {p.shared:.4f}")

    ''' STEP 2: Reconcile disagreement in overlapping evidence '''

    # gather the Participants evidence for each hypothesis
    shared_m = defaultdict(list)
    for p in ss_participants:
        shared_m[p.get_hypothesis()].append(p)

    # loop through each hypothesis and the associated participants to reconcile the shared evidence for each hypothesis
    # TODO: make m dynamically based on frame
    trees = shared_m[frozenset({'trees'})]
    no_trees = shared_m[frozenset({'no trees'})]
    m_shared = {
        frozenset({"trees"}): reconcile_disagreement([x.s for x in trees] + [x.s for x in no_trees], [x.shared for x in trees] + [0]*len(no_trees)),
        frozenset({"no trees"}): reconcile_disagreement([x.s for x in no_trees] + [x.s for x in trees], [x.shared for x in no_trees] + [0]*len(trees))
    }
    if report:
        print("\nStage 2: Reconciling disagreement in overlapping evidence")
        for subset, mass in m_shared.items():
            print(f"m'({set(subset)}) = {mass:.4f}")

    # add ignorance values
    m_shared[frozenset(frame)] = 1 - sum(m_shared.values())

    ''' STEP 3: Weight expert opinion '''

    # apply weights to disjoint expert opinion
    mass_functions = [{p.get_hypothesis(): p.disjoint * p.s, frame: 1 - (p.disjoint * p.s)} for p in ss_participants]

    if report:
        print("\nStage 3: Weighting expert opinion")
        for m1 in mass_functions:
            for subset, mass in m1.items():
                print(f"m({set(subset)}) = {mass:.4f}")

    ''' STEP 4: Combination '''

    # combine our shared and disjoint bpas
    result = m_shared
    for m in mass_functions:
        result = dempster_combination(result, m)
    
    # add missing hypotheses to result
    if frozenset({"trees"}) not in result:
        result[ frozenset({"trees"})] = 0
    if frozenset({"no trees"}) not in result:
        result[ frozenset({"no trees"})] = 0

    if report:
        print("\nStage 4: Combination:")
        for subset, mass in result.items():
            print(f"m({set(subset)}) = {mass:.4f}")

    # return 
    return result


# Example usage with two foci in each source
if __name__ == "__main__":
    '''
    Example configuration.
    This setup is the setup used for the example in the paper - it results in:

        Stage 1: Decomposition into simple support functions and overlapping & disjoint evidence
        disjoint = 0.3306, shared = 0.5519
        disjoint = 0.2929, shared = 0.2929
        disjoint = 0.2254, shared = 0.2254

        Stage 2: Reconciling disagreement in overlapping evidence
        m'({'trees'}) = 0.2600
        m'({'no trees'}) = 0.0811

        Stage 3: Weighting expert opinion
        m({'trees'}) = 0.2314
        m({'no trees', 'trees'}) = 0.7686
        m({'trees'}) = 0.2636
        m({'no trees', 'trees'}) = 0.7364
        m({'no trees'}) = 0.2029
        m({'no trees', 'trees'}) = 0.7971

        Stage 4: Combination:
        m({'trees'}) = 0.5096
        m({'no trees'}) = 0.1424
        m({'no trees', 'trees'}) = 0.3481

        Belief and Plausibility
        Bel(frozenset({'trees'}))=0.5096        Pl(frozenset({'trees'}))=0.8576
        Bel(frozenset({'no trees'}))=0.1424     Pl(frozenset({'no trees'}))=0.4904
    '''
    out = lr_combination(participants = [
        Participant({frozenset(["trees"]): 0.7 }, 0.7, set(['biodiversity', 'flooding', 'climate change'])),
        Participant({frozenset(["trees"]): 0.5, frozenset(["no trees"]): 0.4 }, 0.9, set(['biodiversity', 'flooding', 'farming', 'access'])),
    ], report=True)

    print(f"\nBelief, Plausibility and Probability")
    bel_pl = {}
    probs = {}
    for h, v in out.items():
        if len(h) == 1:
            bp = {'bel':compute_belief(out, h), 'pl':compute_plausibility(out, h)}
            bel_pl[h] = bp
            probs[h] = bp['bel'] + (bp['pl'] / 2)  # NB: 2 is the number of hypotheses
            print(f"Bel({h})={bp['bel']:.4f}\tPl({h})={bp['pl']:.4f}")

    print(f"\nProbabilities")
    s = sum(probs.values())
    for h, p in probs.items():
        print(f"P({h})={p/s:.4f}")