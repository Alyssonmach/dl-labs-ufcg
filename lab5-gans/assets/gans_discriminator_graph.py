import graphviz

dot_data = r"""
digraph PatchGAN {

    graph [
        rankdir=TB,
        splines=ortho,
        nodesep=0.5,
        ranksep=0.7,
        bgcolor="transparent",
        fontname="Arial",
        pad=0.2
    ]

    node [
        shape=box,
        style="rounded,filled",
        fontname="Arial",
        fontsize=11,
        margin="0.18,0.12"
    ]

    edge [
        color="#555555",
        penwidth=1.5,
        arrowsize=0.7
    ]

    input [
        label="Entrada para o discriminador\n[L | AB]\nShape: (128, 128, 3)",
        fillcolor="#E8F1FB"
    ]

    patchgan [
        label="Discriminador\nPatchGAN",
        fillcolor="#FFF2CC"
    ]

    output [
        label="Grade de patches\n(N, N, 1)\n\nCada posição representa uma decisão:\nReal ou Falso por região",
        fillcolor="#E2F0D9"
    ]

    input -> patchgan
    patchgan -> output
}
"""

graph = graphviz.Source(dot_data, format="svg")

output_path = graph.render(
    filename="patchgan_discriminator",
    directory=".",
    cleanup=True
)