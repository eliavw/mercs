NODE_PREFIXES = dict(
    data="d",
    vote="v",
    prob="p",
    merge="M",
    imputation="I",
    imputer="I",
    function="f",
    composition="c",
    model="f")


# Utils, used only in 'inference_legacy.py'
def get_ids(g, kind="desc"):

    if kind in {"s", "src", "source", "d", "desc", "descriptive"}:
        r = get_desc(g, ids=True)
    elif kind in {"t", "tgt", "targ", "target"}:
        r = get_targ(g, ids=True)
    else:
        msg = """
        Did not recognize kind:   {}
        """.format(
            kind
        )
        raise ValueError(msg)

    return r


def get_desc(g, ids=False):
    data_nodes = get_nodes(g, kind="data")

    r = {n for n in data_nodes if len(g.in_edges(n)) == 0}

    if ids:
        r = {g.nodes[n]["idx"] for n in r}

    return r


def get_targ(g, ids=False):
    data_nodes = get_nodes(g, kind="data")

    r = {n for n in data_nodes if len(g.out_edges(n)) == 0}

    if ids:
        r = {g.nodes[n]["idx"] for n in r}

    return r


def get_nodes(g, kind="data"):
    prefix = NODE_PREFIXES[kind]
    r = {n for n in g.nodes if n.startswith(prefix)}
    return r
