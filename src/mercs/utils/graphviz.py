import os
import networkx as nx


def fix_layout(G):
    for n in G.nodes:
        node = G.nodes(data=True)[n]
        if node["kind"] in {"model"}:
            node["shape"] = '"square"'
            node["width"] = "1"
        elif node["kind"] in {"data", "prob"}:
            node["shape"] = '"circle"'
        elif node["kind"] in {"vote", "merge"}:
            node["shape"] = '"triangle"'
        elif node["kind"] in {"imputation"}:
            node["shape"] = '"invtriangle"'
        else:
            pass

    return G


def to_dot(
    g,
    dname="tmp",
    fname="test",
    extension=".dot",
    return_fname=False,
    ortho=False,
    fi_labels=False,
):
    """
    Convert a graph to a dot file.
    """

    # Layout
    if fi_labels:
        for e in g.edges():
            g.edges()[e]["label"] = "{0:.2f}".format(g.edges()[e].get("fi", 0))

    dot = nx.drawing.nx_pydot.to_pydot(g)
    dot.set("rankdir", "BT")

    if ortho:
        dot.set("splines", "ortho")

    # To file
    full_fname = os.path.join(dname, fname + extension)

    with open(full_fname, "w") as f:
        print(dot.to_string(), file=f)

    if return_fname:
        return full_fname
    else:
        return
