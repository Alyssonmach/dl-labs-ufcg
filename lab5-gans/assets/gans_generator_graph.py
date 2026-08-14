import graphviz

dot_data = r"""
digraph UNetGenerator {

    graph [
        rankdir=LR,
        splines=ortho,
        nodesep=0.35,
        ranksep=1.0,
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
        label="Entrada L\n128×128×1",
        fillcolor="#E8F1FB"
    ]

    enc1 [
        label="Enc1\n64×64×64",
        fillcolor="#DDEBF7"
    ]

    enc2 [
        label="Enc2\n32×32×128",
        fillcolor="#DDEBF7"
    ]

    enc3 [
        label="Enc3\n16×16×256",
        fillcolor="#DDEBF7"
    ]

    enc4 [
        label="Enc4\n8×8×512",
        fillcolor="#DDEBF7"
    ]

    bottleneck [
        label="Bottleneck\n4×4×512",
        fillcolor="#FFF2CC"
    ]


    dec2 [
        label="Dec2\n16×16×512",
        fillcolor="#E2F0D9"
    ]

    dec3 [
        label="Dec3\n32×32×256",
        fillcolor="#E2F0D9"
    ]

    dec4 [
        label="Dec4\n64×64×128",
        fillcolor="#E2F0D9"
    ]

    dec5 [
        label="Dec5\n128×128×64",
        fillcolor="#E2F0D9"
    ]

    output [
        label="Saída AB\n128×128×2",
        fillcolor="#FCE4D6"
    ]



    input -> enc1 -> enc2 -> enc3 -> enc4 -> bottleneck
    bottleneck -> dec2 -> dec3 -> dec4 -> dec5 -> output


    enc4 -> dec2 [
        style=dashed,
        color="#888888",
        constraint=false
    ]

    enc3 -> dec3 [
        style=dashed,
        color="#888888",
        constraint=false
    ]

    enc2 -> dec4 [
        style=dashed,
        color="#888888",
        constraint=false
    ]

    enc1 -> dec5 [
        style=dashed,
        color="#888888",
        constraint=false
    ]


    { rank=same; enc1; enc2; enc3; enc4 }
    { rank=same; dec2; dec3; dec4; dec5 }
}
"""

graph = graphviz.Source(dot_data, format="svg")

output_path = graph.render(
    filename="unet_generator",
    directory=".",
    cleanup=True
)