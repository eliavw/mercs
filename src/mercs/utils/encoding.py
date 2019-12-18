import numpy as np


def encode_attribute(att, desc, targ):
    """
    Encode the 'role' of an attribute in a model.

    This is an important function, since here resides the precise encoding.
    If this is changed, a whole lot of other things will change too. Since everything
    that relies on the encoding should refer here, everything should proceed normally.

    `Role` means:
        - Descriptive attribute (input)
        - Target attribute (output)
        - Missing attribute (not relevant to the model)
    """

    check_desc = att in desc
    check_targ = att in targ

    code_int = check_targ * 2 + check_desc - 1

    return code_int


# Ugly but it just needs to be done once so deal with it.
DESC_ENCODING = encode_attribute(1, [1], [2])
TARG_ENCODING = encode_attribute(2, [1], [2])
MISS_ENCODING = encode_attribute(0, [1], [2])

ENCODING = dict(desc=DESC_ENCODING, targ=TARG_ENCODING, miss=MISS_ENCODING)


def code_to_query(code, return_sets=False, return_list=False):

    desc = get_att(code, kind="desc")
    targ = get_att(code, kind="targ")
    miss = get_att(code, kind="miss")

    if return_sets:
        return set(desc), set(targ), set(miss)
    elif return_list:
        return list(desc), list(targ), list(miss)
    else:
        return desc, targ, miss


def get_att(code, kind="desc"):
    return np.where(code == ENCODING[kind])[0].astype(int)


def get_att_2d(codes, kind="desc"):
    return np.unique(np.where(codes == ENCODING[kind])[1].astype(int))


def query_to_code(q_desc, q_targ, q_miss=None, attributes=None):

    if attributes is None:
        attributes = determine_attributes(q_desc, q_targ, q_miss)

    code = [encode_attribute(a, q_desc, q_targ) for a in attributes]

    return np.array(code)


def change_role(
    q_code, source=ENCODING["miss"], target=ENCODING["targ"], inplace=False
):
    if not inplace:
        q_code = q_code.copy()

    q_code[q_code == source] = target
    return q_code


def determine_attributes(desc, targ, miss=None):
    """
    Determine the entire list of attributes.
    """
    if miss is None:
        miss = []

    atts = list(set(desc).union(targ).union(miss))
    atts.sort()
    return atts
