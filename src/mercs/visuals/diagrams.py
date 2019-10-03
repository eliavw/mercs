import networkx as nx
import pydot
import tempfile
import warnings
from shutil import copyfile

from IPython.display import display, Image, SVG


def diagram_to_dotstring(g, fi_labels=False, ortho=False, rankdir="BT"):
    # Layout
    if fi_labels:
        for e in g.edges():
            g.edges()[e]["label"] = "{0:.2f}".format(g.edges()[e].get("fi", 0))

    dot = nx.drawing.nx_pydot.to_pydot(g)

    dot.set("rankdir", rankdir)

    if ortho:
        dot.set("splines", "ortho")

    dotstring = dot.to_string()

    return dotstring


def dotstring_to_image(dotstring):
    # Create temporary dotfile
    f_dotstring = tempfile.NamedTemporaryFile("w", delete=False, suffix=".dot")
    f_dotstring.write(dotstring)
    f_dotstring.close()

    # Read in PyDot
    (graph,) = pydot.graph_from_dot_file(f_dotstring.name)

    # Create temporary image file
    f_img = tempfile.NamedTemporaryFile("w", delete=False, suffix=".png")
    f_img.close()

    f_vec = tempfile.NamedTemporaryFile("w", delete=False, suffix=".svg")
    f_vec.close()

    (graph,) = pydot.graph_from_dot_file(f_dotstring.name)
    graph.write_png(f_img.name)
    graph.write_svg(f_vec.name)

    return f_img, f_vec


def show_diagram(diagram, kind="svg", fi=False, ortho=False, **kwargs):

    try:
        get_ipython()
        dotstring = diagram_to_dotstring(diagram, fi_labels=fi, ortho=ortho)

        f_img, f_vec = dotstring_to_image(dotstring)

        if kind in {"vector", "vec", "svg"}:
            return display(SVG(f_vec.name))
        elif kind in {"image", "img", "png"}:
            return display(Image(f_img.name, **kwargs))
        else:
            msg = """
            Did not recognize kind: {}
            Supported: ['vector', 'image']
            """.format(
                kind
            )
            raise NotImplementedError(msg)
    except BaseException as e:
        msg = """
        We can only show these diagrams in the ipython shell if you are calling this from the ipython shell.
        Exception was: {}
        """.format(e)
        warnings.warn(msg)


def save_diagram(diagram, fname, kind="svg", fi=False, ortho=False):
    dotstring = diagram_to_dotstring(diagram, fi_labels=fi, ortho=ortho)
    f_img, f_vec = dotstring_to_image(dotstring)

    if kind in {"vector", "vec", "svg"}:
        copyfile(f_vec.name, fname)
    elif kind in {"image", "img", "png"}:
        copyfile(f_img.name, fname)
    else:
        msg = """
        Did not recognize kind: {}
        Supported: ['vector', 'image']
        """.format(
            kind
        )
        raise NotImplementedError(msg)
    return
